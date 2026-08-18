"""Reusable neural-network building blocks and training utilities shared by models."""

from __future__ import annotations

from .config import OptimConfig, TrainConfig
from .diagnostics import (
    ActivationProbe,
    DiagnosticsConfig,
    OptimizerProbe,
    dead_fraction,
    flat_mask,
    saturated_fraction,
    weight_summary,
)
from .lstm import LSTMEncoder
from .mlp import MLP
from .pooling import POOLING_KINDS, TemporalPooling
from .proximal import ProximalL1
from .training import (
    EarlyStopping,
    FitResult,
    Optimization,
    build_optimization,
    save_checkpoint,
    should_log_epoch,
)
from .transformer import (
    NORM_POSITIONS,
    POSITIONAL_ENCODINGS,
    RotaryEncoderLayer,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEncoder,
)
from .utils import (
    build_activation,
    count_parameters,
    resolve_device,
    same_device,
    set_seed,
)

__all__ = [
    # blocks
    "MLP",
    "LSTMEncoder",
    "TransformerEncoder",
    "RotaryEncoderLayer",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "TemporalPooling",
    "POSITIONAL_ENCODINGS",
    "NORM_POSITIONS",
    "POOLING_KINDS",
    "EarlyStopping",
    # training loop
    "FitResult",
    "Optimization",
    "build_optimization",
    "should_log_epoch",
    "save_checkpoint",
    "OptimConfig",
    "TrainConfig",
    # diagnostics
    "DiagnosticsConfig",
    "OptimizerProbe",
    "ActivationProbe",
    "flat_mask",
    "saturated_fraction",
    "dead_fraction",
    "weight_summary",
    # penalties
    "ProximalL1",
    # utils
    "build_activation",
    "count_parameters",
    "resolve_device",
    "same_device",
    "set_seed",
]
