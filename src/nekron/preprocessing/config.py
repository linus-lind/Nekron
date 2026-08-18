"""Typed Hydra configuration for panel preprocessing.

A run composes ``configs/preprocessing/*.yaml``, validates it against the schema
below, and materializes a typed :class:`PreprocessingConfig` via :func:`to_config`.
The pipeline is expressed as an ordered list of ``{type, params}`` steps resolved
through the :mod:`~nekron.preprocessing.registry`, so the full cleaning pipeline is
declared — and versioned — in configuration. Build the runnable pipeline with
:func:`build_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .pipeline import Pipeline
from .registry import build_transform


@dataclass
class TransformSpec:
    """One preprocessing step: its registered ``type`` and constructor ``params``."""

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Root configuration for a preprocessing run.

    Parameters
    ----------
    steps:
        Ordered transform specs applied in sequence to the ``(date, entity)`` panel.
    """

    steps: list[TransformSpec] = field(default_factory=list)


CONFIG_SCHEMA_NAME = "base_preprocessing"


def register_configs() -> None:
    """Register the preprocessing schema with Hydra's ConfigStore (call before @hydra.main)."""
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=PreprocessingConfig)


def to_config(cfg: DictConfig) -> PreprocessingConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed ``PreprocessingConfig``."""
    return cast(PreprocessingConfig, OmegaConf.to_object(cfg))


def build_pipeline(cfg: PreprocessingConfig) -> Pipeline:
    """Construct a :class:`Pipeline` from the configuration."""
    return Pipeline(steps=tuple(build_transform(spec.type, spec.params) for spec in cfg.steps))
