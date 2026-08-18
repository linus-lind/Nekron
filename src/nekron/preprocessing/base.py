"""Shared abstractions for panel preprocessing transforms.

Every transform consumes and returns a :class:`pandas.DataFrame` carrying a
two-level ``(date, entity)`` :class:`pandas.MultiIndex`. :class:`PanelTransform`
is the common contract that :class:`~nekron.preprocessing.pipeline.Pipeline`
composes.

Memory contract
---------------
Column-rewriting transforms edit columns in place and return the same frame
object. Row-dropping transforms return a new frame holding only the retained
rows. Neither copies the full frame when it has no work to do, so ordering
row-reducing steps early keeps the working set small.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from nekron.frames import resolve_level
from nekron.panel import PanelGrain

__all__ = [
    "ALL_GRAINS",
    "GrainAware",
    "PanelTransform",
    "PreprocessingError",
    "entity_time_ordered",
    "key_array",
    "require_columns",
    "require_named_levels",
    "resolve_level",
    "transform_grains",
]

ALL_GRAINS: tuple[PanelGrain, ...] = tuple(PanelGrain)


class PreprocessingError(Exception):
    """Base class for preprocessing configuration errors."""


@runtime_checkable
class PanelTransform(Protocol):
    """A single, composable preprocessing step over a ``(date, entity)`` panel."""

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed panel."""
        ...


@runtime_checkable
class GrainAware(Protocol):
    """A transform that declares which panel grains it is defined over."""

    @property
    def grains(self) -> tuple[PanelGrain, ...]:
        """The grains this transform produces a meaningful result for."""
        ...


def transform_grains(step: object) -> tuple[PanelGrain, ...]:
    """Return the grains ``step`` is defined over, defaulting to ``(date, entity)``.

    The default is the strict one because the failure mode is asymmetric. Most
    transforms here are only meaningful over a real panel, and three of them do
    not raise on the wrong grain — they return a plausible wrong answer. Over an
    entity-keyed static frame, where every entity is a single row, a forward fill
    fills nothing, a cumulative factor is all ones, and a NaN-fraction filter
    degenerates into "drop any row containing a NaN". Reference data is exactly
    that shape, so a transform has to opt *in* to the looser grains.
    """
    return step.grains if isinstance(step, GrainAware) else (PanelGrain.PANEL,)


def require_columns(panel: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ``KeyError`` if any of ``columns`` is absent from the panel."""
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns: {missing}.")


def key_array(panel: pd.DataFrame, key: str) -> npt.NDArray[Any]:
    """Return the values of ``key``, resolved as an index level or a column."""
    if key in list(panel.index.names):
        return np.asarray(panel.index.get_level_values(key))
    if key in panel.columns:
        return panel[key].to_numpy()
    raise KeyError(f"grouping key {key!r} is neither an index level nor a column.")


def require_named_levels(index: pd.Index) -> None:
    """Raise ``ValueError`` if any index level is unnamed."""
    if any(name is None for name in index.names):
        raise ValueError(
            f"panel index levels must be named to be used as keys; got names {list(index.names)}."
        )


def entity_time_ordered(index: pd.Index) -> bool:
    """True if each entity's rows are date-ascending, avoiding a needless sort.

    A two-level index that is monotonic in either level order keeps every entity's
    dates ascending, so within-entity sequential operations are already well
    defined.
    """
    if index.is_monotonic_increasing:
        return True
    return (
        isinstance(index, pd.MultiIndex)
        and index.nlevels == 2
        and index.swaplevel(0, 1).is_monotonic_increasing
    )
