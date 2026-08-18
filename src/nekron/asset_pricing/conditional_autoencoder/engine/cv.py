"""Cross-validated fitting of the conditional autoencoder.

The sweep itself — the schedule, the nested runs, the seeding order, the failure
policy, the summary table and the pooling discipline — is model-agnostic and lives
in :mod:`nekron.cv`. What is here is only what that driver cannot know: how this
model's periods are built, how one fold is fitted and scored, and what its numbers
are called.

Why a model per fold is kept
----------------------------
Scores could be logged as numbers and the models thrown away. Loadings could not:
every question of the form "what did the beta network do with this characteristic,
in this period" needs that fold's weights back. The model is small — a few tens of
thousands of parameters — so keeping one per fold costs a fraction of a megabyte
and is what makes :mod:`nekron.asset_pricing.conditional_autoencoder.analysis`
possible at all.

Two namespaces, on purpose
--------------------------
A fold's numbers appear twice under different names: as MLflow metrics
(``test/total_r2``) and as columns of the results table (``test_total_r2``). Those
are not the same namespace — one is a path, the other an identifier — so
:class:`~nekron.cv.FoldOutcome` carries both rather than renaming either.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd

from nekron.cv import (
    COLUMNS_FILE,
    PERIODS_FILE,
    STATUS_TAG,
    SUMMARY_FILE,
    FoldOutcome,
    FoldRecord,
    SweepError,
    SweepReport,
    SweepResult,
    fold_checkpoint_dir,
    fold_seed,
    plan_schedule,
    run_sweep,
    sweep_tags,
)
from nekron.data_adapter import assemble_panel, dataset_fingerprint
from nekron.data_adapter.splitting import FoldBounds, FoldWindow
from nekron.tracking import MlflowTracker

from ..config import ConditionalAutoencoderConfig
from ..data.dataset import CrossSectionPanel, build_cross_section_panel
from ..metrics import pooled_r_squared
from ..model import ConditionalAutoencoder
from .trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = [
    "SUMMARY_FILE",
    "PERIODS_FILE",
    "COLUMNS_FILE",
    "STATUS_TAG",
    "DATA_KEY_PARAM",
    "pooled_segment",
    "pooled_validation",
    "CrossValidationError",
    "CrossValidationResult",
    "FoldResult",
    "build_schedule",
    "run_folds",
]

DATA_KEY_PARAM = "data_key"
"""Parameter and tag naming the panel a run trained on."""

CrossValidationError = SweepError
"""The sweep's failure type, under this package's historical name."""

CrossValidationResult = SweepResult
"""A finished sweep. See :class:`nekron.cv.SweepResult`."""

FoldResult = FoldRecord
"""One fold of a sweep. See :class:`nekron.cv.FoldRecord`."""

REPORT = SweepReport(headline="test R^2", spread_column="test_total_r2", spread_label="total R^2")

SPREAD_COLUMNS = (
    "val_total_r2",
    "val_predictive_r2",
    "test_total_r2",
    "test_predictive_r2",
)
"""Summary columns a per-fold count/mean/median/min/max/std/SEM/IQR is reported for.

Both windows are here because they answer different questions about a fold. The
held-out window is what the fold *scored*, and so what one configuration is
compared with another on. The validation window is what early stopping *selected
the epoch* on, which makes its spread a diagnostic of the stopping policy rather
than of the model: a validation score far above the held-out one says the epoch
was chosen on noise.
"""

SEGMENT_SUMS = ("sse_total", "sst_total", "sse_pred", "sst_pred")
"""Per-period error and total columns a pooled R-squared is reassembled from.

Pooling is a ratio of summed errors to summed totals over every scored period, so
it needs the sums and not the folds' own ratios — a set of per-fold R-squareds
cannot be combined into the R-squared of their union at all.
"""

VALIDATION_SUMS = tuple(f"val_{name}" for name in SEGMENT_SUMS)
"""Summary columns the pooled validation R-squared is reassembled from."""

TEST_SUMS = tuple(f"test_{name}" for name in SEGMENT_SUMS)
"""Summary columns the pooled held-out R-squared is reassembled from.

Redundant with the per-period table the sweep already writes, and deliberately so:
a comparison between two runs needs one small artifact per run rather than the
full period-level table, and the two agree exactly because both are the same ratio
of the same sums.
"""


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #


def build_schedule(
    cfg: ConditionalAutoencoderConfig,
) -> tuple[CrossSectionPanel, tuple[FoldBounds, ...]]:
    """Assemble the panel, build every usable period once, and cut the schedule.

    The two steps are ordered this way on purpose. Periods whose cross-section is
    too small to price are dropped when the sections are built, so the sequence a
    fold indexes is shorter than the panel's date index;
    :func:`~nekron.cv.plan_schedule` is what keeps the boundaries where they belong
    given that.
    """
    panel = assemble_panel(cfg.data_pipeline)
    split = cfg.data_pipeline.split
    panel_dates = pd.Index(sorted(panel.index.get_level_values(split.date_level).unique()))
    sections = build_cross_section_panel(panel, cfg.data)
    bounds = plan_schedule(
        sections.dates,
        panel_dates,
        split,
        what=f"cross-section of at least data.min_cross_section={cfg.data.min_cross_section}",
    )
    return sections, bounds


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def run_folds(
    panel: CrossSectionPanel,
    bounds: tuple[FoldBounds, ...],
    cfg: ConditionalAutoencoderConfig,
    tracker: MlflowTracker,
) -> CrossValidationResult:
    """Fit one model per fold, each inside its own nested run under ``tracker``."""
    tracker.log_config(cfg)
    # The digest of the data the run will see, which the configuration parameters
    # do not imply: the source file can be replaced without its path changing.
    fingerprint = dataset_fingerprint(cfg.data_pipeline)
    tracker.log_params({DATA_KEY_PARAM: fingerprint})
    tracker.set_tags({DATA_KEY_PARAM: fingerprint})
    tracker.log_json(
        {
            "beta_columns": list(panel.beta_columns),
            "portfolio_columns": list(panel.portfolio_columns),
        },
        COLUMNS_FILE,
    )

    def body(
        fold: FoldBounds, window: FoldWindow, run: MlflowTracker, num_folds: int
    ) -> FoldOutcome:
        return _run_fold(panel, fold, window, cfg, run, num_folds)

    result = run_sweep(
        bounds,
        panel.dates,
        body,
        policy=cfg.cv,
        tracker=tracker,
        seed=cfg.train.seed,
        pool=pool_r_squared,
        spread_columns=SPREAD_COLUMNS,
        report=REPORT,
        tags=sweep_tags(cfg.data_pipeline.split, len(bounds)),
    )
    validation = pooled_validation(result.summary)
    tracker.log_metrics({f"val/pooled_{k}": v for k, v in validation.items()}, step=0)
    logger.info(
        "pooled validation R^2 -> total=%.5f predictive=%.5f",
        validation["total_r2"],
        validation["predictive_r2"],
    )
    return result


def pooled_segment(summary: pd.DataFrame, prefix: str) -> dict[str, float]:
    """One segment's R-squared over every period of every completed fold.

    The counterpart of :func:`pool_r_squared` for a window whose per-period rows a
    fold does not return, pooled the same way: the ratio of the summed errors to
    the summed totals, never the mean of the folds' ratios.

    Reassembled from the four sums each fold carries in the summary table, so
    ``prefix`` is ``"val"`` or ``"test"``. Failed folds are excluded rather than
    contributing zeros, which would silently lower the errors *and* the totals.
    """
    columns = [f"{prefix}_{name}" for name in SEGMENT_SUMS]
    if "status" not in summary.columns or not set(columns) <= set(summary.columns):
        return {"total_r2": float("nan"), "predictive_r2": float("nan")}
    done = summary.loc[summary["status"] == "completed", columns]
    return {
        "total_r2": _ratio(done[f"{prefix}_sse_total"].sum(), done[f"{prefix}_sst_total"].sum()),
        "predictive_r2": _ratio(done[f"{prefix}_sse_pred"].sum(), done[f"{prefix}_sst_pred"].sum()),
    }


def pooled_validation(summary: pd.DataFrame) -> dict[str, float]:
    """The validation R-squared over every scored validation period of every fold.

    A diagnostic, not a selection metric: the validation window is what early
    stopping chose the epoch on, so this number is a maximum over epochs and is
    optimistic by an amount that varies with how noisy a configuration's training
    curve is. Read it against the held-out number, never in place of it.
    """
    return pooled_segment(summary, "val")


def _ratio(sse: float, sst: float) -> float:
    """``1 - sse/sst``, or ``nan`` where there is no variation to explain."""
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def _segment_sums(rows: pd.DataFrame, prefix: str) -> dict[str, float]:
    """One window's four error and total sums, named for the summary table."""
    return {f"{prefix}_{name}": float(rows[name].sum()) for name in SEGMENT_SUMS}


def pool_r_squared(rows: pd.DataFrame) -> dict[str, float]:
    """The sweep's headline R-squared, over every scored period of every fold.

    The ratio of the summed errors to the summed totals, never the mean of the
    folds' ratios: periods differ in how many stocks they price and in how much
    return variation they carry, so a set of per-fold R-squareds cannot be combined
    into the R-squared of their union.
    """
    if not len(rows):
        return {"total_r2": float("nan"), "predictive_r2": float("nan")}
    pooled = pooled_r_squared(rows)
    return {"total_r2": pooled.total, "predictive_r2": pooled.predictive}


def _run_fold(
    panel: CrossSectionPanel,
    bounds: FoldBounds,
    window: FoldWindow,
    cfg: ConditionalAutoencoderConfig,
    run: MlflowTracker,
    num_folds: int,
) -> FoldOutcome:
    """One fold, start to finish. The driver has already reseeded and opened ``run``."""
    fold_cfg = _fold_config(cfg, bounds.index, num_folds)
    splits = panel.splits(bounds)
    model = ConditionalAutoencoder.from_config(
        fold_cfg,
        num_beta_columns=panel.num_beta_columns,
        num_portfolios=panel.num_portfolios,
    )
    trainer = Trainer(model, fold_cfg, splits, run)
    fit = trainer.fit()

    # One forecast walks the fold in calendar order — training, then validation,
    # then test — each window opening with the state the one before it ended in, so
    # the test window is scored against everything this fold was allowed to see.
    # Chained rather than summed: under ``factor_forecast.mode: ewma`` the state
    # discounts the past and cannot be assembled from per-split summaries. The
    # training pass is factor-network only; advancing a forecast needs no loadings.
    after_train = trainer.advance_forecast(splits.train)
    val_score = trainer.score(splits.val, forecast=after_train)
    test_score = trainer.score(splits.test, forecast=val_score.forecast)
    periods = test_score.rows
    # Both windows pooled from their own rows, so the two scores are the same
    # function of the same table rather than two routes to nearly the same float.
    val = pooled_r_squared(val_score.rows)
    test = pooled_r_squared(periods)

    run.log_json(
        # window.to_dict() carries fold_index too; listing it after would replace
        # the int with its string form.
        {
            "beta_columns": list(panel.beta_columns),
            "portfolio_columns": list(panel.portfolio_columns),
            **window.to_dict(),
            "fold_index": bounds.index,
        },
        COLUMNS_FILE,
    )
    best = fit.best_row
    return FoldOutcome(
        scores={
            "val_total_r2": val.total,
            "val_predictive_r2": val.predictive,
            "val_periods": float(len(val_score.rows)),
            **_segment_sums(val_score.rows, "val"),
            "test_total_r2": test.total,
            "test_predictive_r2": test.predictive,
            **_segment_sums(periods, "test"),
        },
        metrics={
            "test/total_r2": test.total,
            "test/predictive_r2": test.predictive,
            "val/final_total_r2": val.total,
            "val/final_predictive_r2": val.predictive,
            # The in-sample minus out-of-sample fit at the epoch that was kept. Its
            # level is not readable — ``train/total_r2`` accumulates while the
            # weights move — but its movement between configurations is what says
            # whether a change bought generalization or memorization.
            "fold/gap_total_r2": best.get("train/total_r2", float("nan"))
            - best.get("val/total_r2", float("nan")),
            **fit.convergence_metrics(),
        },
        rows=periods,
        best_epoch=fit.best_epoch,
        best_metric=fit.best_metric,
        history=fit.history,
    )


def _fold_config(
    cfg: ConditionalAutoencoderConfig, index: int, num_folds: int
) -> ConditionalAutoencoderConfig:
    """``cfg`` as fold ``index`` sees it: its own seed, its own checkpoint directory.

    A single-fold schedule under the default seeding is handed the configuration
    unchanged — the same object, not a copy — so an ordinary run still writes
    ``checkpoints/conditional_autoencoder_best.pt`` where it always has.

    The two-line :func:`dataclasses.replace` stays in this package rather than
    moving to :mod:`nekron.cv`: ``replace`` is typed against a concrete dataclass,
    so a config-agnostic version of it cannot type-check under strict mode. The
    *policy* it applies is shared — see :func:`~nekron.cv.fold_seed` and
    :func:`~nekron.cv.fold_checkpoint_dir`.
    """
    if num_folds == 1 and cfg.cv.seed_mode == "fixed":
        return cfg
    return replace(
        cfg,
        train=replace(
            cfg.train,
            seed=fold_seed(cfg.train.seed, index, cfg.cv.seed_mode),
            checkpoint_dir=fold_checkpoint_dir(cfg.train.checkpoint_dir, index, num_folds),
        ),
    )
