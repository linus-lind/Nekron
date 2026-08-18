"""Cross-validation policy: how a run behaves once the schedule has more than one fold.

Deliberately free of torch, pandas and MLflow, so a caller that only wants to
compose or validate a configuration does not pay for a multi-second import. The
sweep driver that consumes this lives in :mod:`nekron.cv.sweep`.

The *schedule* — rolling or expanding, the window sizes, the purge — is a property
of the data and lives in :class:`~nekron.data_adapter.config.SplitConfig`. What is
here is only the training policy that a multi-fold run needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SEED_MODES: tuple[str, ...] = ("fixed", "per_fold")
FOLD_ERROR_POLICIES: tuple[str, ...] = ("raise", "skip")

SeedMode = Literal["fixed", "per_fold"]
FoldErrorPolicy = Literal["raise", "skip"]


@dataclass
class CrossValidationConfig:
    """How a run behaves once the fold schedule has more than one fold.

    Parameters
    ----------
    seed_mode:
        ``"fixed"`` initializes every fold from the same seed, so the differences
        between folds are differences in the *data* — which is what a distribution
        over folds is normally claiming to show. ``"per_fold"`` offsets the seed by
        the fold index, which mixes initialization noise into that distribution;
        use it when you want the spread to include model variance, or as a cheap
        stand-in for an ensemble.
    on_fold_error:
        What a failing fold does to the run. ``"skip"`` records the failure, marks
        that fold's MLflow run FAILED and carries on — a fold whose window is too
        thin to fit should not cost the six good folds that follow it. The
        failures are counted in the summary and in ``folds/failed``, and a run in
        which *every* fold failed still raises. ``"raise"`` stops at the first one.
    """

    seed_mode: str = "fixed"
    on_fold_error: str = "skip"

    def __post_init__(self) -> None:
        if self.seed_mode not in SEED_MODES:
            raise ValueError(
                f"cv.seed_mode must be one of {list(SEED_MODES)}; got {self.seed_mode!r}."
            )
        if self.on_fold_error not in FOLD_ERROR_POLICIES:
            raise ValueError(
                f"cv.on_fold_error must be one of {list(FOLD_ERROR_POLICIES)}; "
                f"got {self.on_fold_error!r}."
            )
