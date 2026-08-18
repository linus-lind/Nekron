"""Shared abstractions for panel feature creation.

A featurizer consumes a working :class:`pandas.DataFrame` carrying a two-level
``(date, entity)`` :class:`pandas.MultiIndex` and returns one or more new feature
columns as arrays aligned to the panel's rows. :class:`Featurizer` is the common
contract that :class:`~nekron.features.pipeline.FeaturePipeline` composes.

Panel-order contract
--------------------
Grouped time-series operations (rolling windows, exponential smoothing, shifts)
require every entity's rows to be contiguous and ascending in date. The pipeline
reorders the panel into entity-major ``(entity, date)`` form once and builds a
:class:`PanelContext` describing that layout; featurizers receive the reordered
panel and the context, compute against it, and the pipeline restores the caller's
row order at the end. Featurizers therefore never sort or group the entity level
themselves — they read the precomputed codes from the context.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


class FeatureError(Exception):
    """Base class for feature-creation errors."""


def resolve_level(index: pd.Index, level: int | str) -> int:
    """Resolve an index-level spec (position or name) to a level position."""
    names = list(index.names)
    if isinstance(level, str):
        if level not in names:
            raise KeyError(f"index has no level named {level!r}; names are {names}.")
        return names.index(level)
    nlevels = index.nlevels
    if not -nlevels <= level < nlevels:
        raise IndexError(f"level {level} is out of range for a {nlevels}-level index.")
    return level % nlevels


def require_columns(panel: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ``KeyError`` if any of ``columns`` is absent from the panel."""
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns: {missing}.")


def column_values(panel: pd.DataFrame, column: str) -> FloatArray:
    """Return a column as a contiguous ``float64`` array, raising if it is absent."""
    require_columns(panel, (column,))
    return np.asarray(panel[column].to_numpy(), dtype=np.float64)


def key_array_or_column(panel: pd.DataFrame, key: str) -> npt.NDArray[np.generic]:
    """Return the values of ``key`` resolved as an index level or a column."""
    if key in list(panel.index.names):
        return np.asarray(panel.index.get_level_values(key))
    if key in panel.columns:
        return np.asarray(panel[key].to_numpy())
    raise KeyError(f"key {key!r} is neither an index level nor a column.")


def check_named(count: int, output_names: tuple[str, ...], what: str) -> None:
    """Validate that ``output_names`` has one unique name per produced column.

    Raises ``ValueError`` if the number of names differs from ``count`` (the number
    of parameter values that each generate a column) or if any name repeats.
    """
    if len(output_names) != count:
        raise ValueError(
            f"expected {count} output name(s) to match the {what}; got {len(output_names)}."
        )
    if len(set(output_names)) != len(output_names):
        raise ValueError(f"output names must be unique; got {output_names}.")


@dataclass(frozen=True)
class PanelContext:
    """Precomputed layout of an entity-major ``(date, entity)`` panel.

    The context is built once per pipeline run and shared by every featurizer, so
    the entity/date groupings and within-entity positions are computed a single
    time rather than re-derived for each feature.

    Attributes
    ----------
    index:
        The MultiIndex of the entity-major working panel.
    entity_codes:
        Integer group code per row identifying its entity. Equal entities share a
        code and their rows are contiguous.
    date_codes:
        Integer group code per row identifying its date (equal dates share a
        code); used for cross-sectional, per-date operations.
    within_entity:
        Zero-based position of each row within its entity's date-ordered series.
    entity_size:
        Number of rows in each row's entity.
    n_rows:
        Total number of rows.
    """

    index: pd.MultiIndex
    entity_codes: IntArray
    date_codes: IntArray
    within_entity: IntArray
    entity_size: IntArray
    n_rows: int

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, entity_level: int, date_level: int) -> PanelContext:
        """Build a context from an already entity-major-ordered panel.

        Assumes each entity's rows are contiguous and ascending in date, which the
        pipeline guarantees before calling this.
        """
        index = panel.index
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            raise FeatureError("feature panels require a two-level (date, entity) MultiIndex.")
        entity_values = index.get_level_values(entity_level)
        date_values = index.get_level_values(date_level)
        entity_codes = pd.factorize(entity_values, sort=False)[0].astype(np.intp)
        date_codes = pd.factorize(date_values, sort=False)[0].astype(np.intp)
        within_entity, entity_size = _within_group(entity_codes)
        return cls(
            index=index,
            entity_codes=entity_codes,
            date_codes=date_codes,
            within_entity=within_entity,
            entity_size=entity_size,
            n_rows=len(entity_codes),
        )


def _within_group(codes: IntArray) -> tuple[IntArray, IntArray]:
    """Zero-based within-group position and per-row group size for contiguous codes."""
    n = codes.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    sizes = ends - starts
    within = np.arange(n, dtype=np.intp) - np.repeat(starts, sizes)
    return within.astype(np.intp), np.repeat(sizes, sizes).astype(np.intp)


@runtime_checkable
class Featurizer(Protocol):
    """A composable step that derives feature columns from a ``(date, entity)`` panel."""

    @property
    def inputs(self) -> tuple[str, ...]:
        """Columns the featurizer reads (raw panel columns or upstream features)."""
        ...

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of the feature columns the featurizer produces."""
        ...

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        """Return a mapping from each output name to its values, aligned to ``panel``."""
        ...


@runtime_checkable
class KeyedFeaturizer(Protocol):
    """A featurizer that additionally groups by a key, which may be an index level.

    ``inputs`` lists the panel *columns* a step reads, so it cannot carry a key
    resolved by :func:`key_array_or_column` — an industry code, say, that may live
    on the index rather than in a column. Such keys are declared here instead, so
    the pipeline still sees them as dependencies: a key produced by an earlier
    featurizer is a real edge in the dependency graph, while a key that is an index
    level is simply absent from it.
    """

    @property
    def key_inputs(self) -> tuple[str, ...]:
        """Index-level-or-column keys the featurizer groups by."""
        ...


def declared_keys(step: object) -> tuple[str, ...]:
    """Return a featurizer's declared grouping keys, or ``()`` if it declares none."""
    return step.key_inputs if isinstance(step, KeyedFeaturizer) else ()
