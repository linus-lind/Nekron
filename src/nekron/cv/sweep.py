"""The cross-validation driver: one model per fold, one nested MLflow run per model.

A run is a sweep, not a fit. :func:`run_sweep` walks the schedule cut by
:mod:`nekron.data_adapter.splitting`, hands each fold to a model-supplied callback,
and records the sweep as a tree: the parent run carries the configuration and the
aggregate, and each fold is a nested run underneath it carrying its own training
curve, its own model and its own per-period rows. A single split is a schedule with
one fold, so this is also the ordinary path — nothing special-cases it.

Nothing here knows what a model is
----------------------------------
The driver never constructs a model, never names a metric and never interprets a
column. It owns the parts that are the same for every model and easy to get subtly
wrong — the seeding order, the nesting, the error policy, the pooling discipline —
and delegates everything else to a :class:`FoldBody` the model package supplies,
usually as a closure over its own panel and configuration. That is a callback and
not a base class on purpose: the two models this was extracted from share no state
worth inheriting, :func:`dataclasses.replace` does not type-check against an
abstract config, and a protected method is an invitation to override exactly the
invariants below.

Independence between folds
--------------------------
Each fold builds a fresh model and a fresh trainer, and the driver reseeds
*before* the body runs — weight initialization draws from the global RNG, and a
trainer that reseeds inside ``fit`` only does so once the model already exists, so
a fold that relied on that would be initialized from whatever state the fold
before it left behind. Under the default ``cv.seed_mode="fixed"`` every fold
therefore starts from identical weights, and the spread across folds is a property
of the data rather than of the initialization.

Pooled, never averaged
----------------------
A fold's rows are kept and concatenated; the sweep's headline number is computed
from the concatenation by a model-supplied :class:`Pooler`. It is never the mean
of the folds' scores. Folds differ in how many rows they carry and in how much
variation those rows hold, and a ratio-of-sums statistic cannot be recovered from
a set of per-fold ratios at all. The per-fold spread is reported *alongside* the
pooled number, as a second, different summary.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from nekron.data_adapter.splitting import FoldBounds, FoldWindow, Segment
from nekron.nn import set_seed
from nekron.nn.training import FOLD_FLAGS, FOLD_MEDIANS, FOLD_PREFIX, FOLD_TOTALS
from nekron.tracking import MlflowTracker

from .config import CrossValidationConfig

logger = logging.getLogger(__name__)

SUMMARY_FILE = "folds.parquet"
PERIODS_FILE = "test_periods.parquet"
HISTORY_FILE = "history.parquet"
COLUMNS_FILE = "columns.json"

STATUS_TAG = "fold_status"
"""Marks a fold run that ran to completion, so a later reader can skip the rest."""

FOLD_INDEX = "fold_index"
"""The column and parameter naming which fold a row or a run belongs to."""


class SweepError(Exception):
    """Raised when a sweep produces no usable fold at all."""


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldOutcome:
    """What one fold hands back. The driver reads exactly these four things.

    Parameters
    ----------
    scores:
        Named scalars copied verbatim into this fold's summary row, so the *model*
        chooses the column names and the driver never has to know what a metric
        means.
    metrics:
        What to record on MLflow, for the fold's own run and for the parent's
        per-fold series. Empty means "the same as :attr:`scores`". It is separate
        because a results column and a metric name are not the same namespace — a
        column is ``test_total_r2`` and the metric it holds is ``test/total_r2`` —
        and collapsing them would rename one or the other.
    rows:
        The fold's per-period rows, or ``None``. They are concatenated across
        folds and handed to the :class:`Pooler`. The driver takes ownership: it
        labels them with :data:`FOLD_INDEX` **in place** rather than copying a
        frame that may be large, so a body must hand over a frame it does not
        itself keep a reference to. A body that has already labeled them is left
        alone.
    best_epoch, best_metric:
        What the fit selected, for the summary row. ``best_metric`` is reported
        as-is — whether higher or lower is better is the model's business, and a
        driver that assumed either would silently select the worst epoch for the
        model that disagreed.
    history:
        Every epoch's metrics, or ``None``. Attached to the fold's run whole,
        because the curve that reaches the tracker as metrics is subsampled by
        ``mlflow.log_every_n_epochs`` and the rows in between are gone once the
        process exits. Whether a fit was still improving when it stopped is a
        question about those rows.
    """

    scores: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    rows: pd.DataFrame | None = None
    best_epoch: int = -1
    best_metric: float = float("nan")
    history: Sequence[Mapping[str, float]] | None = None


class FoldBody(Protocol):
    """Fit and score one fold, inside the nested run already opened for it.

    Implemented by each model package, normally as a closure over its panel and
    configuration. The driver has already reseeded and opened ``run`` when this is
    called, and will record the outcome and tag the run afterwards; the body's
    only jobs are to build its model, fit it, score it, and log whatever is
    specific to it.
    """

    def __call__(
        self, bounds: FoldBounds, window: FoldWindow, run: MlflowTracker, num_folds: int, /
    ) -> FoldOutcome:
        """Run fold ``bounds.index``; raise to fail it.

        Positional-only, so an implementation is free to name its parameters
        whatever reads best at the call site it lives in.
        """
        ...


class Pooler(Protocol):
    """Reduce every fold's concatenated rows to the sweep's headline scalars."""

    def __call__(self, rows: pd.DataFrame, /) -> Mapping[str, float]:
        """Named scalars over ``rows``; an empty frame must still return the keys."""
        ...


@dataclass(frozen=True)
class SweepReport:
    """How a sweep describes itself in one paragraph of prose.

    Purely presentational, and separate from the numbers so that
    :meth:`SweepResult.describe` can name a model's own statistic without the
    driver knowing anything about it.
    """

    headline: str = "score"
    """What the pooled numbers are, e.g. ``"test R^2"``."""

    spread_column: str = ""
    """Summary column the per-fold spread is taken over; empty omits that line."""

    spread_label: str = ""
    """What that column is, e.g. ``"total R^2"``."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldRecord:
    """One fold's outcome: where it sat, what it scored, where its model went."""

    index: int
    window: FoldWindow
    run_id: str | None
    outcome: FoldOutcome | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.error is None

    @property
    def scores(self) -> Mapping[str, float]:
        """The fold's named scalars; empty for a fold that failed."""
        return self.outcome.scores if self.outcome is not None else {}

    @property
    def metrics(self) -> Mapping[str, float]:
        """What this fold recorded on MLflow; falls back to :attr:`scores`."""
        if self.outcome is None:
            return {}
        return self.outcome.metrics or self.outcome.scores

    @property
    def rows(self) -> pd.DataFrame | None:
        return self.outcome.rows if self.outcome is not None else None

    def to_row(self) -> dict[str, object]:
        """One row of the sweep's summary table.

        ``FoldWindow.to_dict`` is string-valued because that is what MLflow tags
        store; a results table wants the counts back as numbers, so the sizes and
        the fold index are re-typed here.
        """
        bounds = self.window.to_dict()
        rows = self.rows
        row: dict[str, object] = {
            **bounds,
            **{name: int(bounds[name]) for name in ("train_size", "val_size", "test_size")},
            "status": "completed" if self.completed else "failed",
            "run_id": self.run_id or "",
            "error": self.error or "",
            "best_epoch": self.outcome.best_epoch if self.outcome else -1,
            "best_metric": self.outcome.best_metric if self.outcome else float("nan"),
            **dict(self.scores),
            "test_periods": 0 if rows is None else int(len(rows)),
        }
        # Written last so it replaces the string form to_dict() put there.
        row[FOLD_INDEX] = self.index
        return row


@dataclass(frozen=True)
class SweepResult:
    """Every fold of a sweep, plus the two ways of summarizing it.

    :attr:`pooled` is the sweep's headline score, computed over every scored row
    of every fold at once. :attr:`summary` carries one row per fold, which is the
    sample a distribution across folds is drawn from, and :attr:`periods` carries
    one row per scored period, which is the (much larger) sample a distribution
    across periods is drawn from.
    """

    folds: tuple[FoldRecord, ...]
    summary: pd.DataFrame
    periods: pd.DataFrame
    pooled: Mapping[str, float]
    report: SweepReport = field(default_factory=SweepReport)

    @property
    def completed(self) -> tuple[FoldRecord, ...]:
        return tuple(fold for fold in self.folds if fold.completed)

    def describe(self) -> str:
        """A few lines naming the pooled score and the spread behind it."""
        pooled = " ".join(f"{name}={value:.4f}" for name, value in self.pooled.items())
        lines = [
            f"folds: {len(self.completed)} completed of {len(self.folds)} planned",
            f"pooled {self.report.headline} -> {pooled} over {len(self.periods)} periods",
        ]
        column = self.report.spread_column
        if column and column in self.summary.columns:
            scores = self.summary.loc[self.summary["status"] == "completed", column]
            if len(scores) > 1:
                lines.append(
                    f"per-fold {self.report.spread_label} -> median={scores.median():.4f} "
                    f"min={scores.min():.4f} max={scores.max():.4f} sd={scores.std():.4f}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-fold configuration
# --------------------------------------------------------------------------- #


def fold_seed(base: int, index: int, seed_mode: str) -> int:
    """The seed fold ``index`` is initialized from."""
    return base + index if seed_mode == "per_fold" else base


def fold_checkpoint_dir(base: str, index: int, num_folds: int) -> str:
    """Where fold ``index`` writes its checkpoint.

    A one-fold schedule keeps the base directory, so an ordinary single-split run
    still writes its checkpoint exactly where it always has. With more than one
    fold the checkpoint *name* is a constant inside each trainer, so without a
    per-fold directory every fold would overwrite the one before it.
    """
    return base if num_folds == 1 else str(Path(base) / f"fold-{index:02d}")


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def run_sweep(
    bounds: Sequence[FoldBounds],
    dates: pd.Index,
    body: FoldBody,
    *,
    policy: CrossValidationConfig,
    tracker: MlflowTracker,
    seed: int,
    pool: Pooler,
    spread_columns: Sequence[str] = (),
    report: SweepReport | None = None,
    tags: Mapping[str, str] | None = None,
) -> SweepResult:
    """Run every fold of ``bounds``, each inside its own nested run under ``tracker``.

    Parameters
    ----------
    bounds:
        The schedule, from :func:`nekron.cv.schedule.plan_schedule`.
    dates:
        The sequence the fold positions index — the periods that survived, not the
        panel's dates.
    body:
        The model's per-fold callback.
    policy:
        Seeding and failure policy.
    tracker:
        The parent run. Must be inside its own ``with`` block: MLflow parents a
        nested run to whatever run is active on the thread.
    seed:
        The run's base seed. Each fold is seeded from it *before* ``body`` runs.
    pool:
        Reduces the concatenated per-period rows to the headline scalars.
    spread_columns:
        Summary columns to report a per-fold mean/median/min/max/std/IQR for.
    report:
        How :meth:`SweepResult.describe` should phrase itself.
    tags:
        Searchable tags for the parent run, e.g. from :func:`sweep_tags`.
    """
    if not bounds:
        raise SweepError("the fold schedule is empty; nothing to fit.")
    warn_on_overlap(bounds)
    if tags:
        tracker.set_tags(dict(tags))

    records: list[FoldRecord] = []
    for fold in bounds:
        # Everything that can fail sits inside the guard, resolving the fold's
        # dates included: a schedule cut over the wrong sequence fails there, and
        # under "skip" that should cost one fold rather than the whole sweep.
        window: FoldWindow | None = None
        run_id: str | None = None
        try:
            window = fold.window(dates)
            logger.info("%s", window.describe())
            with tracker.child(f"fold-{fold.index:02d}", tags=window.to_dict()) as run:
                # Captured before anything can throw, so a failed fold still points
                # at the run holding its partial logs.
                run_id = run.run_id
                records.append(_run_fold(fold, window, run, body, policy, seed, len(bounds)))
        except Exception as exc:  # noqa: BLE001 - policy is the caller's, recorded below
            if policy.on_fold_error == "raise":
                raise
            logger.warning("fold %d failed and was skipped: %s", fold.index, exc)
            records.append(
                FoldRecord(
                    index=fold.index,
                    window=window if window is not None else unplaced_window(fold.index),
                    run_id=run_id,
                    error=str(exc),
                )
            )

    return _finish(tuple(records), tracker, pool=pool, spread=spread_columns, report=report)


def _run_fold(
    bounds: FoldBounds,
    window: FoldWindow,
    run: MlflowTracker,
    body: FoldBody,
    policy: CrossValidationConfig,
    seed: int,
    num_folds: int,
) -> FoldRecord:
    """One fold, start to finish, inside the nested run already opened for it."""
    this_seed = fold_seed(seed, bounds.index, policy.seed_mode)
    # Before the model exists: initialization draws from the global RNG, and a
    # trainer reseeds only after its model has been constructed.
    set_seed(this_seed)
    outcome = body(bounds, window, run, num_folds)

    rows = outcome.rows
    if rows is not None and FOLD_INDEX not in rows.columns:
        rows.insert(0, FOLD_INDEX, bounds.index)
        outcome = replace(outcome, rows=rows)

    run.log_params({FOLD_INDEX: str(bounds.index), "seed": str(this_seed)})
    run.log_metrics(
        {
            **{name: float(v) for name, v in (outcome.metrics or outcome.scores).items()},
            "fold/best_epoch": float(outcome.best_epoch),
            # A metric and not only a summary column: the score early stopping
            # selected on is what a later query ranks folds by, and a value that
            # lives solely in a Parquet artifact has to be downloaded to be read.
            "fold/best_metric": float(outcome.best_metric),
            "fold/test_periods": float(0 if rows is None else len(rows)),
        },
        step=0,
    )
    if rows is not None:
        run.log_dataframe(rows, PERIODS_FILE)
    if outcome.history:
        run.log_dataframe(pd.DataFrame(list(outcome.history)), HISTORY_FILE)
    # Last, so it marks a fold that got all the way here. A fold that raised leaves
    # a run without it, which is how a later reader tells the two apart.
    run.set_tags({STATUS_TAG: "completed"})
    return FoldRecord(index=bounds.index, window=window, run_id=run.run_id, outcome=outcome)


def _finish(
    records: tuple[FoldRecord, ...],
    tracker: MlflowTracker,
    *,
    pool: Pooler,
    spread: Sequence[str],
    report: SweepReport | None,
) -> SweepResult:
    """Pool the folds, log the aggregate on the parent run, and assemble the result."""
    completed = [fold for fold in records if fold.completed]
    if not completed:
        raise SweepError(
            f"every one of the {len(records)} folds failed; the first said: {records[0].error}"
        )

    summary = pd.DataFrame([fold.to_row() for fold in records])
    frames = [fold.rows for fold in completed if fold.rows is not None]
    # The explicit column list on the empty branch is not decoration: an empty
    # frame built by inference would carry object-dtype columns that promote every
    # fold's dtypes on concatenation.
    periods = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[FOLD_INDEX])
    pooled = dict(pool(periods))

    tracker.log_metrics(_aggregate(summary, periods, records, pooled, spread), step=0)
    # A series indexed by fold, so the tracking UI plots the sweep as a curve.
    for fold in completed:
        if fold.metrics:
            tracker.log_metrics(dict(fold.metrics), step=fold.index)
    tracker.log_dataframe(summary, SUMMARY_FILE)
    if len(periods):
        tracker.log_dataframe(periods, PERIODS_FILE)

    return SweepResult(
        folds=records,
        summary=summary,
        periods=periods,
        pooled=pooled,
        report=report or SweepReport(),
    )


def _aggregate(
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    records: tuple[FoldRecord, ...],
    pooled: Mapping[str, float],
    spread: Sequence[str],
) -> dict[str, float]:
    """The parent run's metrics: the pooled score, and the spread across folds."""
    metrics = {
        **{f"test/pooled_{name}": float(value) for name, value in pooled.items()},
        "folds/planned": float(len(records)),
        "folds/completed": float(sum(fold.completed for fold in records)),
        "folds/failed": float(sum(not fold.completed for fold in records)),
        "folds/test_periods": float(len(periods)),
    }
    metrics.update(aggregate_spread(summary, spread))
    metrics.update(aggregate_convergence(records))
    return metrics


def aggregate_convergence(records: Sequence[FoldRecord]) -> dict[str, float]:
    """How the folds terminated, rolled up to one row on the parent run.

    A sweep's headline score says nothing about whether the fits behind it
    finished. These say how many folds ran out of epochs rather than converging,
    how many produced no finite score at all, and what the whole thing cost — the
    three questions that decide whether a configuration's number is admissible
    evidence before it is a good one. Counts rather than fractions, because "two
    of eleven folds hit the cap" is the actionable form and a fraction hides the
    denominator.

    Reads the per-fold metrics rather than the summary table, so it works for any
    model whose fold reports what :meth:`~nekron.nn.FitResult.convergence_metrics`
    produces and stays silent for one that does not.
    """
    completed = [fold for fold in records if fold.completed]
    if not completed:
        return {}
    metrics: dict[str, float] = {}
    for key in FOLD_FLAGS:
        values = [float(fold.metrics[key]) for fold in completed if key in fold.metrics]
        if values:
            metrics[f"folds/{key.removeprefix(FOLD_PREFIX)}"] = float(sum(values))
    for key in FOLD_TOTALS:
        values = [float(fold.metrics[key]) for fold in completed if key in fold.metrics]
        if values:
            metrics[f"fit/{key.removeprefix(FOLD_PREFIX)}_total"] = float(sum(values))
    for key in FOLD_MEDIANS:
        values = [float(fold.metrics[key]) for fold in completed if key in fold.metrics]
        if values:
            metrics[f"fit/{key.removeprefix(FOLD_PREFIX)}_median"] = float(np.median(values))
    return metrics


def aggregate_spread(summary: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Per-fold count, mean, median, min, max, standard deviation, SEM and IQR.

    The second of the two summaries a sweep produces, and the one that says
    whether a pooled number rests on agreement between folds or on one of them.
    Failed folds are excluded rather than counted as zeros, which is why the count
    is reported: a dispersion over nine folds and one over three are different
    evidence and otherwise indistinguishable in the recorded metrics.

    The standard error is the one of these a comparison between two runs is
    actually built on, and it is not a substitute for the standard deviation — the
    deviation describes how much folds disagree, the error how precisely their
    mean is known. Neither exists for a one-fold schedule, so both are omitted
    there rather than reported as zero.

    A column's namespace is its first underscore-separated word, so
    ``test_total_r2`` is reported under ``test/total_r2_*`` and ``val_total_r2``
    under ``val/total_r2_*``: the summary table and the metric tree stay in step
    without either of them being told about the other.
    """
    if not columns or "status" not in summary.columns:
        return {}
    done = summary["status"] == "completed"
    metrics: dict[str, float] = {}
    for column in columns:
        if column not in summary.columns:
            continue
        scores = summary.loc[done, column].astype("float64").dropna()
        if scores.empty:
            continue
        namespace, _, name = column.partition("_")
        if not name:
            namespace, name = "fold", column
        metrics[f"{namespace}/{name}_n"] = float(len(scores))
        metrics[f"{namespace}/{name}_mean"] = float(scores.mean())
        metrics[f"{namespace}/{name}_median"] = float(scores.median())
        metrics[f"{namespace}/{name}_min"] = float(scores.min())
        metrics[f"{namespace}/{name}_max"] = float(scores.max())
        if len(scores) > 1:
            deviation = float(scores.std())
            metrics[f"{namespace}/{name}_std"] = deviation
            metrics[f"{namespace}/{name}_sem"] = deviation / math.sqrt(len(scores))
            metrics[f"{namespace}/{name}_iqr"] = float(
                np.subtract(*np.percentile(scores.to_numpy(), [75, 25]))
            )
    return metrics


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def unplaced_window(index: int) -> FoldWindow:
    """A window for a fold that could not even be located in the date sequence."""
    blank = Segment(start=None, end=None, size=0)
    return FoldWindow(index=index, train=blank, val=blank, test=blank)


def sweep_tags(split: object, num_folds: int) -> dict[str, str]:
    """Searchable tags describing the schedule a sweep ran.

    Takes the :class:`~nekron.data_adapter.config.SplitConfig` itself rather than a
    model's root configuration, because the two models disagree about where that
    config lives — one nests it under a data pipeline, the other inherits it from
    another model's run.
    """
    scheme = str(getattr(split, "scheme", ""))
    walk_forward = getattr(split, "walk_forward", None)
    return {
        "cv_scheme": scheme,
        "cv_mode": str(getattr(walk_forward, "mode", "")) if scheme == "walk_forward" else "",
        "cv_folds": str(num_folds),
        "cv_purge": str(getattr(walk_forward, "purge", "")) if scheme == "walk_forward" else "",
        "cv_burn_in": str(getattr(split, "burn_in", "")),
    }


def warn_on_overlap(bounds: Sequence[FoldBounds]) -> None:
    """Say so when test windows overlap, because pooling then double-counts dates.

    Only possible with an explicit ``step`` below ``test_size``. It is a legitimate
    thing to configure — more folds out of a short sample — but the pooled score
    stops being an average over distinct dates, and the per-period table will carry
    a date more than once.
    """
    overlapping = sum(
        1 for a, b in zip(bounds, bounds[1:], strict=False) if b.test.start < a.test.stop
    )
    if overlapping:
        logger.warning(
            "%d pairs of consecutive test windows overlap (split.walk_forward.step is smaller "
            "than test_size); pooled statistics count the shared dates once per fold that "
            "covers them.",
            overlapping,
        )
