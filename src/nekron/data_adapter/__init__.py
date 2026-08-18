"""Generic data adapter: config-driven ingestion -> per-panel stages -> alignment -> features -> split.

Two entry points turn the stage configurations (panel ingestion, preprocessing,
alignment, feature creation) plus a :class:`SplitConfig` into model-ready panels,
so every model shares one implementation instead of re-wiring the stages itself.
:func:`build_datasets` returns a single train/val/test split; :func:`build_folds`
returns the whole fold schedule, which is one fold under the default ``single``
scheme and one per position of the sweep under ``walk_forward``.
"""

from __future__ import annotations

from .adapter import (
    AdapterError,
    DataSplits,
    FoldPlan,
    assemble,
    assemble_panel,
    build_datasets,
    build_folds,
    dataset_fingerprint,
    plan_panel,
    split_panel,
)
from .config import (
    DataAdapterConfig,
    FeatureStages,
    PreprocessingStages,
    SplitConfig,
    SplitScheme,
    WalkForwardConfig,
    WindowMode,
    register_configs,
    to_config,
)
from .splitting import FoldBounds, FoldWindow, Segment, generate_folds, plan_folds

__all__ = [
    "AdapterError",
    "DataAdapterConfig",
    "PreprocessingStages",
    "FeatureStages",
    "SplitConfig",
    "SplitScheme",
    "WalkForwardConfig",
    "WindowMode",
    "register_configs",
    "to_config",
    "DataSplits",
    "FoldBounds",
    "FoldPlan",
    "FoldWindow",
    "Segment",
    "assemble",
    "assemble_panel",
    "dataset_fingerprint",
    "generate_folds",
    "plan_folds",
    "plan_panel",
    "split_panel",
    "build_datasets",
    "build_folds",
]
