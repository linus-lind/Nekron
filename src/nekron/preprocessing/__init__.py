"""Panel preprocessing: composable ``(date, entity)`` cleaning steps.

Each :class:`PanelTransform` is a small, configuration-driven cleaning step; a
:class:`Pipeline` runs an ordered sequence of them. Steps are declared by
``{type, params}`` in a Hydra config, resolved through the
:mod:`~nekron.preprocessing.registry` and materialized with
:func:`build_pipeline`.
"""

from __future__ import annotations

from .adjustments import Adjustment, CorporateAdjustment, CumulativeFactor
from .base import PanelTransform, PreprocessingError
from .calendar import TradingCalendarFilter
from .config import (
    PreprocessingConfig,
    TransformSpec,
    build_pipeline,
    register_configs,
    to_config,
)
from .consistency import ConsistencyFilter, ConsistencyRule
from .deduplicate import DuplicateMerger
from .imputation import ConstantImputer, ForwardFillImputer
from .nan_filter import NaNEntityFilter
from .pipeline import Pipeline
from .registry import build_transform, register_transform, registered_transforms

__all__ = [
    # core
    "PanelTransform",
    "PreprocessingError",
    "Pipeline",
    # transforms
    "TradingCalendarFilter",
    "Adjustment",
    "CorporateAdjustment",
    "CumulativeFactor",
    "DuplicateMerger",
    "ConsistencyRule",
    "ConsistencyFilter",
    "ConstantImputer",
    "ForwardFillImputer",
    "NaNEntityFilter",
    # config / registry
    "PreprocessingConfig",
    "TransformSpec",
    "build_pipeline",
    "register_configs",
    "to_config",
    "build_transform",
    "register_transform",
    "registered_transforms",
]
