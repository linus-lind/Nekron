"""Typed Hydra configuration for feature creation.

A run composes ``configs/features/*.yaml``, validates it against the schema below,
and materializes a typed :class:`FeatureConfig` via :func:`to_config`. The list of
featurizers is expressed as ``{type, params}`` entries resolved through the
:mod:`~nekron.features.registry`, so the full feature set is declared — and
versioned — in configuration. Build the runnable pipeline with
:func:`build_pipeline`.

Featurizers run top to bottom, so some of them exist only to feed a later one. Such
a scaffolding column is marked ``intermediate`` on the spec that produces it: it is
computed, handed to the steps that need it, and then dropped from the feature
panel. That keeps the declaration next to the column it describes, and lets the
build validate it — an intermediate no later featurizer consumes is a configuration
error, not silently wasted work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from nekron.constants import DATE_LEVEL, ENTITY_LEVEL

from .base import FeatureError, Featurizer, declared_keys
from .pipeline import FeaturePipeline
from .registry import build_featurizer
from .selection import ColumnSelection


@dataclass
class ColumnSelectionConfig:
    """Which columns to keep on one side of the output panel.

    Parameters
    ----------
    include:
        Column names to keep. ``null`` imposes no restriction and keeps every
        available column; an explicit list keeps exactly those, in order; ``[]``
        keeps none.
    exclude:
        Column names to drop from the included set; exclusion wins over inclusion.
    strict:
        Whether every name listed must exist among the available columns (the
        default), so a stale or misspelled name fails the run.
    """

    include: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    strict: bool = True


def _keep_no_columns() -> ColumnSelectionConfig:
    """Default input selection: retain none of the original panel columns."""
    return ColumnSelectionConfig(include=[])


@dataclass
class FeaturizerSpec:
    """One featurizer entry in the ordered pipeline.

    Parameters
    ----------
    type:
        Registered featurizer name, resolved through the
        :mod:`~nekron.features.registry`.
    params:
        Constructor parameters for that featurizer.
    intermediate:
        Outputs of *this* featurizer that exist only to feed a later one and must
        not reach the feature panel — a cumulative level behind its first
        difference, say. Each name must be one of the featurizer's own outputs and
        must be read by a later featurizer. To drop an output nothing downstream
        consumes, list it in ``keep_features.exclude`` instead.
    """

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)
    intermediate: list[str] = field(default_factory=list)


@dataclass
class FeatureConfig:
    """Root configuration for a feature-creation run.

    Parameters
    ----------
    featurizers:
        Ordered featurizer specs; later ones may depend on earlier outputs.
    entity_level, date_level:
        Names (or positions) of the entity and date levels in the panel index.
    keep_inputs:
        Which original panel columns to retain alongside the features; the default
        (``include: []``) keeps none, ``include: null`` keeps them all. Retained
        columns pass through untouched — not cast, not warm-up-trimmed — and every
        column of the result is a feature to whatever consumes the panel.
    keep_features:
        Which produced features to output; the default (``include: null``) keeps
        every feature that is not marked ``intermediate`` on its spec.
    output_dtype:
        Optional dtype to cast the feature columns to (``null`` keeps float64).
    drop_warmup:
        Whether to discard the leading burn-in dates (dates where some retained
        feature is undefined across the whole cross-section because it lacks enough
        history).
    """

    featurizers: list[FeaturizerSpec] = field(default_factory=list)
    entity_level: str = ENTITY_LEVEL
    date_level: str = DATE_LEVEL
    keep_inputs: ColumnSelectionConfig = field(default_factory=_keep_no_columns)
    keep_features: ColumnSelectionConfig = field(default_factory=ColumnSelectionConfig)
    output_dtype: str | None = None
    drop_warmup: bool = True


CONFIG_SCHEMA_NAME = "base_features"


def register_configs() -> None:
    """Register the feature schema with Hydra's ConfigStore (call before @hydra.main)."""
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=FeatureConfig)


def to_config(cfg: DictConfig) -> FeatureConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed ``FeatureConfig``."""
    return cast(FeatureConfig, OmegaConf.to_object(cfg))


def to_selection(cfg: ColumnSelectionConfig) -> ColumnSelection:
    """Build the backend-agnostic :class:`ColumnSelection` from its config."""
    return ColumnSelection(
        include=tuple(cfg.include) if cfg.include is not None else None,
        exclude=tuple(cfg.exclude),
        strict=cfg.strict,
    )


def build_pipeline(cfg: FeatureConfig) -> FeaturePipeline:
    """Construct a :class:`FeaturePipeline` from the configuration."""
    featurizers = tuple(build_featurizer(spec.type, spec.params) for spec in cfg.featurizers)
    keep_features = to_selection(cfg.keep_features)
    intermediates = _intermediate_outputs(cfg.featurizers, featurizers)
    if intermediates:
        requested = [name for name in intermediates if name in (keep_features.include or ())]
        if requested:
            raise FeatureError(
                f"{requested} are marked intermediate but also listed in keep_features.include; "
                "a column cannot be both scaffolding and an output."
            )
        exclude = tuple(dict.fromkeys((*keep_features.exclude, *intermediates)))
        keep_features = replace(keep_features, exclude=exclude)
    return FeaturePipeline(
        featurizers=featurizers,
        entity_level=cfg.entity_level,
        date_level=cfg.date_level,
        keep_inputs=to_selection(cfg.keep_inputs),
        keep_features=keep_features,
        output_dtype=cfg.output_dtype,
        drop_warmup=cfg.drop_warmup,
    )


def _intermediate_outputs(
    specs: list[FeaturizerSpec], featurizers: tuple[Featurizer, ...]
) -> tuple[str, ...]:
    """Resolve the specs' ``intermediate`` markings into the columns to drop.

    Each marked name must be one of its own featurizer's declared outputs and must
    be read by a later featurizer: an intermediate nothing consumes is computed and
    thrown away, which is a configuration mistake rather than a valid request.
    """
    marked: list[str] = []
    for position, (spec, featurizer) in enumerate(zip(specs, featurizers, strict=True)):
        names = list(dict.fromkeys(spec.intermediate))
        if not names:
            continue
        unknown = [name for name in names if name not in featurizer.outputs]
        if unknown:
            raise FeatureError(
                f"featurizer {spec.type!r} marks {unknown} as intermediate, but its outputs are "
                f"{list(featurizer.outputs)}."
            )
        consumed = {
            name
            for later in featurizers[position + 1 :]
            for name in (*later.inputs, *declared_keys(later))
        }
        unused = [name for name in names if name not in consumed]
        if unused:
            raise FeatureError(
                f"featurizer {spec.type!r} marks {unused} as intermediate, but no later "
                "featurizer consumes them; drop the featurizer, or exclude the columns with "
                "keep_features.exclude."
            )
        marked.extend(names)
    return tuple(dict.fromkeys(marked))
