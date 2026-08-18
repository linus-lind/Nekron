"""Model-agnostic cross-validation: cut a schedule, fit a model per fold, pool once.

Two pieces, deliberately separable. :func:`plan_schedule` cuts a fold schedule over
the periods a model can actually use, which is not the panel's date index and is
the thing every model gets wrong the same way. :func:`run_sweep` then walks that
schedule, handing each fold to a callback the model package supplies and owning
everything around it — the seeding order, the nested MLflow runs, the failure
policy, the summary table, and the discipline that the headline number is pooled
over every fold's rows rather than averaged over their scores.

Nothing here constructs a model, names a metric or interprets a column, so a new
model joins by writing a :class:`~nekron.cv.sweep.FoldBody` and a
:class:`~nekron.cv.sweep.Pooler` and nothing else.

Importing the policy costs nothing
----------------------------------
:mod:`nekron.cv.config` imports no torch, no pandas and no MLflow, so a caller that
only wants to compose or validate a configuration should not pay for them. That is
only true if importing this package does not drag the driver in behind it, so the
driver and the schedule are resolved lazily (:pep:`562`): ``from nekron.cv.config
import CrossValidationConfig`` stays a few milliseconds, while ``from nekron.cv
import run_sweep`` loads what it needs, once, on first use. Both spellings work and
neither is special-cased anywhere else.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import (
    FOLD_ERROR_POLICIES,
    SEED_MODES,
    CrossValidationConfig,
    FoldErrorPolicy,
    SeedMode,
)

if TYPE_CHECKING:  # resolved lazily at runtime; imported here so the names type-check
    from .schedule import plan_schedule
    from .sweep import (
        COLUMNS_FILE,
        FOLD_INDEX,
        HISTORY_FILE,
        PERIODS_FILE,
        STATUS_TAG,
        SUMMARY_FILE,
        FoldBody,
        FoldOutcome,
        FoldRecord,
        Pooler,
        SweepError,
        SweepReport,
        SweepResult,
        aggregate_convergence,
        aggregate_spread,
        fold_checkpoint_dir,
        fold_seed,
        run_sweep,
        sweep_tags,
        unplaced_window,
        warn_on_overlap,
    )

_LAZY: dict[str, str] = {
    "plan_schedule": ".schedule",
    **dict.fromkeys(
        (
            "COLUMNS_FILE",
            "FOLD_INDEX",
            "HISTORY_FILE",
            "PERIODS_FILE",
            "STATUS_TAG",
            "SUMMARY_FILE",
            "FoldBody",
            "FoldOutcome",
            "FoldRecord",
            "Pooler",
            "SweepError",
            "SweepReport",
            "SweepResult",
            "aggregate_convergence",
            "aggregate_spread",
            "fold_checkpoint_dir",
            "fold_seed",
            "run_sweep",
            "sweep_tags",
            "unplaced_window",
            "warn_on_overlap",
        ),
        ".sweep",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve a driver name on first use, caching it in the module namespace."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # policy — imported eagerly, and free of torch, pandas and MLflow
    "CrossValidationConfig",
    "SEED_MODES",
    "FOLD_ERROR_POLICIES",
    "SeedMode",
    "FoldErrorPolicy",
    # schedule
    "plan_schedule",
    # the seam
    "FoldBody",
    "FoldOutcome",
    "Pooler",
    "SweepReport",
    # driver
    "run_sweep",
    "SweepResult",
    "FoldRecord",
    "SweepError",
    # helpers
    "fold_seed",
    "fold_checkpoint_dir",
    "aggregate_spread",
    "aggregate_convergence",
    "HISTORY_FILE",
    "sweep_tags",
    "unplaced_window",
    "warn_on_overlap",
    # artifact names
    "SUMMARY_FILE",
    "PERIODS_FILE",
    "COLUMNS_FILE",
    "STATUS_TAG",
    "FOLD_INDEX",
]
