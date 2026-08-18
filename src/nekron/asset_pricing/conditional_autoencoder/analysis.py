"""Read a finished sweep back: its fold models, and what they made of each characteristic.

:func:`load_folds` recovers every fold of a cross-validated run from MLflow — the
fitted model, the column order it was trained on, the fold's window and its
per-period errors. :func:`characteristic_importance` then asks each of those models
the same question about each input characteristic, which turns a sweep into a
*distribution* of answers rather than one: a characteristic that matters in every
fold is a different finding from one that mattered in 2020 alone, and a single
split cannot tell them apart.

Column order is the thing to be careful about
---------------------------------------------
The beta network's input width is inferred from the data, so input dimension ``p``
means "the p-th name in ``beta_columns``" and nothing else. A model reloaded
against a feature configuration that has since changed will answer confidently and
wrongly. Every loader here therefore carries the column tuples the fold was trained
with, recorded next to the model at fit time, and refuses to interpret a model
without them.

What pools across folds, and what does not
------------------------------------------
The importances below are invariant to the model's rotation indeterminacy — they
are computed from fitted returns, which are unchanged by ``beta f = (beta R)(R^-1
f)`` — so they pool across folds and can be compared fold to fold directly. Raw
loadings and the factor series themselves are *not*: separate fits land in
different bases, and "factor 3" in one fold has no reason to be "factor 3" in the
next. Anything factor-specific needs an explicit alignment step first.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from nekron.nn import resolve_device
from nekron.tracking import MlflowConfig

from .data.dataset import CrossSectionDataset
from .engine.cv import COLUMNS_FILE, PERIODS_FILE, STATUS_TAG
from .model import ConditionalAutoencoder

logger = logging.getLogger(__name__)


class MissingArtifactError(Exception):
    """Raised when a fold run does not carry something needed to interpret its model."""


ImportanceMethod = Literal["zero_out", "sensitivity"]
IMPORTANCE_METHODS: tuple[str, ...] = ("zero_out", "sensitivity")


@dataclass(frozen=True)
class FoldArtifacts:
    """One fold of a sweep, recovered from its MLflow run."""

    index: int
    run_id: str
    model: ConditionalAutoencoder
    beta_columns: tuple[str, ...]
    portfolio_columns: tuple[str, ...]
    window: dict[str, str]
    periods: pd.DataFrame | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_folds(
    parent_run_id: str, cfg: MlflowConfig, *, with_periods: bool = True
) -> tuple[FoldArtifacts, ...]:
    """Every fold of the sweep whose parent run is ``parent_run_id``, in fold order.

    MLflow tags a nested run with ``mlflow.parentRunId`` automatically, which is
    what identifies the folds of one sweep. They are ordered here rather than by
    the tracking server: run parameters are stored as strings, so a server-side
    sort by ``fold_index`` would put fold 10 between fold 1 and fold 2.
    """
    import mlflow

    mlflow.set_tracking_uri(cfg.tracking_uri)
    children = mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    if children.empty:
        raise ValueError(
            f"no fold runs found under parent run {parent_run_id!r} in experiment "
            f"{cfg.experiment_name!r}; check the run id and the tracking URI."
        )
    folds: list[FoldArtifacts] = []
    skipped: list[str] = []
    for _, row in children.iterrows():
        run_id = str(row["run_id"])
        if not _completed(row):
            skipped.append(f"{run_id} (did not complete)")
            continue
        try:
            folds.append(_load_fold(row, with_periods=with_periods))
        except MissingArtifactError as exc:
            skipped.append(f"{run_id} ({exc})")
    if skipped:
        # A sweep run with cv.on_fold_error="skip" leaves a run behind for each
        # fold that failed. Those are part of the record and should not stop the
        # folds that worked from being read.
        logger.warning(
            "%d of %d fold runs under %s were not loadable and are omitted: %s",
            len(skipped),
            len(children),
            parent_run_id,
            "; ".join(skipped),
        )
    if not folds:
        raise ValueError(
            f"none of the {len(children)} fold runs under {parent_run_id!r} completed, so "
            "there is no model to read; check the sweep's folds/failed metric and the "
            "fold runs' logs."
        )
    return tuple(sorted(folds, key=lambda fold: fold.index))


def _completed(row: pd.Series) -> bool:
    """Whether this fold run recorded that it got all the way through.

    An absent column means the sweep predates the marker, in which case the
    artifacts decide; a missing value where siblings have one means this fold
    stopped early.
    """
    key = f"tags.{STATUS_TAG}"
    # Membership, not ``.get``: a run that never set the tag has ``None`` in that
    # column, which ``.get`` cannot tell apart from the column being absent — and
    # those two mean opposite things here.
    if key not in row.index:
        return True
    value = row[key]
    return value is not None and not pd.isna(value) and str(value) == "completed"


def _load_fold(row: pd.Series, *, with_periods: bool) -> FoldArtifacts:
    import mlflow

    run_id = str(row["run_id"])
    columns = _read_json(run_id, COLUMNS_FILE)
    beta_columns = tuple(str(name) for name in columns["beta_columns"])
    if not beta_columns:
        raise ValueError(f"run {run_id} recorded no beta_columns; its model cannot be read.")
    return FoldArtifacts(
        index=int(columns.get("fold_index", row.get("params.fold_index", 0))),
        run_id=run_id,
        model=mlflow.pytorch.load_model(f"runs:/{run_id}/model"),
        beta_columns=beta_columns,
        portfolio_columns=tuple(str(name) for name in columns.get("portfolio_columns", ())),
        window={
            key.removeprefix("tags."): str(value)
            for key, value in row.items()
            if isinstance(key, str)
            and key.startswith("tags.")
            and not key.startswith("tags.mlflow")
        },
        periods=_read_parquet(run_id, PERIODS_FILE) if with_periods else None,
    )


def _read_json(run_id: str, name: str) -> dict[str, Any]:
    import mlflow

    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
    except Exception as exc:
        raise MissingArtifactError(
            f"no {name!r}, so the column order this model was trained on is unknown and it "
            "cannot be interpreted per feature; a fold that stopped early, or a run from "
            "before this artifact was written"
        ) from exc
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def _read_parquet(run_id: str, name: str) -> pd.DataFrame | None:
    import mlflow

    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
    except Exception as exc:  # noqa: BLE001 - an absent optional artifact is not an error
        logger.warning("run %s has no %s (%s)", run_id, name, exc)
        return None
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Characteristic importance
# --------------------------------------------------------------------------- #


def characteristic_importance(
    model: ConditionalAutoencoder,
    dataset: CrossSectionDataset,
    beta_columns: Sequence[str],
    *,
    portfolio_columns: Sequence[str] = (),
    method: str = "zero_out",
    device: str = "auto",
) -> pd.Series:
    """How much each input characteristic is worth to ``model`` over ``dataset``.

    ``"zero_out"`` follows the reference implementation of the paper: the
    characteristic is set to zero everywhere and the drop in total R-squared is
    recorded. Zero is the right neutral value rather than an arbitrary one — the
    features are rank-mapped cross-sectionally to ``[-1, 1]`` with the median at
    zero, so zeroing a column *is* setting every stock to the median of that
    characteristic. Larger is more important; a negative value means the model was
    fitting noise through that column on this window.

    ``"sensitivity"`` instead reports the mean absolute gradient of the loadings
    with respect to the characteristic, summed over factors. It answers a
    different question — how sharply the loadings respond, rather than how much
    the fit needs it — and costs one backward pass per factor per period instead
    of one forward pass per characteristic.

    When a characteristic also appears in ``portfolio_columns`` the ``"zero_out"``
    measure neutralizes it on *both* sides and re-solves that period's managed
    portfolios, because a characteristic the model sees twice is not held out by
    removing it once. Pass ``portfolio_columns`` to enable that; omitting it
    measures the loading side alone.
    """
    if method not in IMPORTANCE_METHODS:
        raise ValueError(f"method must be one of {list(IMPORTANCE_METHODS)}; got {method!r}.")
    resolved = resolve_device(device)
    model = model.to(resolved).eval()
    if method == "sensitivity":
        return _sensitivity(model, dataset, beta_columns, resolved)
    return _zero_out(model, dataset, beta_columns, portfolio_columns, resolved)


def _zero_out(
    model: ConditionalAutoencoder,
    dataset: CrossSectionDataset,
    beta_columns: Sequence[str],
    portfolio_columns: Sequence[str],
    device: torch.device,
) -> pd.Series:
    baseline = _total_r2(model, dataset, device)
    portfolio_at = {name: position for position, name in enumerate(portfolio_columns)}
    drops = {
        name: baseline
        - _total_r2(model, dataset, device, beta_at=position, portfolio_at=portfolio_at.get(name))
        for position, name in enumerate(beta_columns)
    }
    return pd.Series(drops, name="importance").sort_values(ascending=False)


@torch.no_grad()
def _total_r2(
    model: ConditionalAutoencoder,
    dataset: CrossSectionDataset,
    device: torch.device,
    *,
    beta_at: int | None = None,
    portfolio_at: int | None = None,
) -> float:
    """Total R-squared, optionally with one characteristic neutralized to the median."""
    sse = 0.0
    sst = 0.0
    for index in range(len(dataset)):
        section = dataset[index]
        beta_inputs = section.beta_inputs
        portfolios = section.portfolios
        if beta_at is not None:
            beta_inputs = beta_inputs.clone()
            beta_inputs[:, beta_at] = 0.0
        if portfolio_at is not None:
            neutralized = section.portfolio_inputs.clone()
            neutralized[:, portfolio_at] = 0.0
            # The managed portfolios are a least-squares fit of returns on these
            # characteristics, so removing one changes every coefficient, not just
            # its own. Re-solving is the only way to hold it out of the factors.
            portfolios = ConditionalAutoencoder.managed_portfolios(neutralized, section.returns)
        output = model(beta_inputs.to(device), portfolios.to(device))
        returns = section.returns.to(device)
        sse += float(((returns - output.fitted_returns) ** 2).sum())
        sst += float((returns**2).sum())
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def _sensitivity(
    model: ConditionalAutoencoder,
    dataset: CrossSectionDataset,
    beta_columns: Sequence[str],
    device: torch.device,
) -> pd.Series:
    """Mean absolute gradient of the loadings with respect to each characteristic."""
    # Accumulated on the host in double precision: the sum runs over every stock
    # of every period, and Apple-silicon MPS supports no 64-bit float at all, so
    # a device-side accumulator would both lose precision and fail outright there.
    totals = torch.zeros(len(beta_columns), dtype=torch.float64)
    stocks = 0
    for index in range(len(dataset)):
        section = dataset[index]
        beta_inputs = section.beta_inputs.to(device).detach().requires_grad_(True)
        loadings = model.beta_network(beta_inputs)
        for factor in range(loadings.shape[1]):
            if beta_inputs.grad is not None:
                beta_inputs.grad = None
            loadings[:, factor].sum().backward(retain_graph=factor < loadings.shape[1] - 1)
            if beta_inputs.grad is not None:
                totals += beta_inputs.grad.abs().sum(dim=0).detach().cpu().to(torch.float64)
        stocks += int(beta_inputs.shape[0])
    values = (totals / max(stocks, 1)).numpy()
    return pd.Series(values, index=list(beta_columns), name="importance").sort_values(
        ascending=False
    )


def fold_importances(
    folds: Sequence[FoldArtifacts],
    datasets: Sequence[CrossSectionDataset],
    *,
    method: str = "zero_out",
    device: str = "auto",
    use_portfolio_columns: bool = True,
) -> pd.DataFrame:
    """Long-form importances, one row per (fold, characteristic).

    The shape a distribution wants: group by ``characteristic`` for a box per
    characteristic across folds, or by ``fold_index`` to see one fold's ranking.
    ``datasets`` must be the window each fold is to be scored on — normally its own
    test window, so that every measurement is out of sample.
    """
    if len(folds) != len(datasets):
        raise ValueError(
            f"got {len(folds)} folds but {len(datasets)} datasets; each fold needs the "
            "window it is to be scored on."
        )
    rows: list[pd.DataFrame] = []
    for fold, dataset in zip(folds, datasets, strict=True):
        scores = characteristic_importance(
            fold.model,
            dataset,
            fold.beta_columns,
            portfolio_columns=fold.portfolio_columns if use_portfolio_columns else (),
            method=method,
            device=device,
        )
        rows.append(
            pd.DataFrame(
                {
                    "fold_index": fold.index,
                    "characteristic": scores.index,
                    "importance": scores.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def importance_summary(importances: pd.DataFrame) -> pd.DataFrame:
    """Per-characteristic median, spread and sign consistency across folds.

    ``folds_positive`` is the count of folds in which the characteristic helped at
    all. It is the column that separates a characteristic the model relies on from
    one whose large median rests on a single fold.
    """
    grouped = importances.groupby("characteristic")["importance"]
    summary = pd.DataFrame(
        {
            "median": grouped.median(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "min": grouped.min(),
            "max": grouped.max(),
            "folds": grouped.size(),
            "folds_positive": grouped.apply(lambda values: int((values > 0).sum())),
        }
    )
    return summary.sort_values("median", ascending=False)
