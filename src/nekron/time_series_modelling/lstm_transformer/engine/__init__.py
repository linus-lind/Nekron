"""Training engine for the LSTM-Transformer residual model.

MLflow tracking lives in :mod:`nekron.tracking` and is re-exported here.
"""

from __future__ import annotations

from nekron.nn import FitResult
from nekron.tracking import MlflowTracker

from .cv import (
    CrossValidationError,
    CrossValidationResult,
    build_schedule,
    run_folds,
)
from .trainer import CHECKPOINT_FILE, Trainer, TrainerError

__all__ = [
    "Trainer",
    "TrainerError",
    "FitResult",
    "CHECKPOINT_FILE",
    "MlflowTracker",
    "CrossValidationError",
    "CrossValidationResult",
    "build_schedule",
    "run_folds",
]
