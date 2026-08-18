"""Training engine for the conditional autoencoder.

MLflow tracking lives in :mod:`nekron.tracking` and is re-exported here.
"""

from __future__ import annotations

from nekron.nn import FitResult
from nekron.tracking import MlflowTracker

from .cv import (
    DATA_KEY_PARAM,
    CrossValidationError,
    CrossValidationResult,
    FoldResult,
    build_schedule,
    pooled_segment,
    pooled_validation,
    run_folds,
)
from .trainer import CHECKPOINT_FILE, InferenceResult, Trainer

__all__ = [
    "Trainer",
    "FitResult",
    "CHECKPOINT_FILE",
    "InferenceResult",
    "MlflowTracker",
    "CrossValidationError",
    "CrossValidationResult",
    "FoldResult",
    "build_schedule",
    "run_folds",
    "pooled_segment",
    "pooled_validation",
    "DATA_KEY_PARAM",
]
