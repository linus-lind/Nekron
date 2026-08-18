"""Residual extraction and windowing for the LSTM-Transformer model."""

from __future__ import annotations

from .residuals import (
    PretrainedCae,
    ResidualError,
    ResidualPanel,
    build_residual_panel,
    load_pretrained_cae,
)
from .windows import (
    ResidualWindows,
    WindowBatch,
    WindowError,
    WindowPanel,
    WindowSplits,
    build_window_panel,
    build_window_splits,
    window_schedule,
)

__all__ = [
    "PretrainedCae",
    "ResidualPanel",
    "ResidualError",
    "load_pretrained_cae",
    "build_residual_panel",
    "ResidualWindows",
    "WindowBatch",
    "WindowSplits",
    "WindowPanel",
    "WindowError",
    "build_window_splits",
    "build_window_panel",
    "window_schedule",
]
