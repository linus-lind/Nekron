"""Imputation of missing panel values: forward-fill or a constant fill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from nekron.constants import ENTITY_LEVEL
from nekron.panel import PanelGrain

from .base import ALL_GRAINS, entity_time_ordered, require_columns, resolve_level


@dataclass(frozen=True)
class ConstantImputer:
    """Fill missing values in the given columns with a constant.

    Unlike forward-fill this needs no time ordering or grouping: every missing
    entry becomes ``value``. Suited to columns whose absence means "none" rather
    than "unchanged" (for example a no-trade day's volume is zero).

    Parameters
    ----------
    columns:
        Columns to fill.
    value:
        Value written into missing entries.
    """

    grains: ClassVar[tuple[PanelGrain, ...]] = ALL_GRAINS
    """A constant fill needs neither ordering nor grouping, so it is meaningful for
    every grain."""

    columns: tuple[str, ...]
    value: float

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        require_columns(panel, self.columns)
        for column in self.columns:
            panel[column] = panel[column].fillna(self.value)
        return panel


@dataclass(frozen=True)
class ForwardFillImputer:
    """Forward-fill missing values within each entity along the date axis.

    Each entity's rows must already be in ascending date order, which holds for
    any index sorted by ``(date, entity)`` or by ``(entity, date)``. If neither
    order holds the panel is sorted first and a new frame is returned.

    Parameters
    ----------
    columns:
        Columns to fill. When empty, every column is filled.
    limit:
        Maximum number of consecutive NaNs filled per gap. ``None`` imposes no
        limit.
    entity_level:
        Position or name of the entity level to group by.
    """

    columns: tuple[str, ...] = field(default_factory=tuple)
    limit: int | None = None
    entity_level: int | str = ENTITY_LEVEL

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        targets = list(self.columns) if self.columns else list(panel.columns)
        require_columns(panel, targets)
        if not entity_time_ordered(panel.index):
            panel = panel.sort_index()
        pos = resolve_level(panel.index, self.entity_level)
        filled = panel.groupby(level=pos, sort=False, group_keys=False)[targets].ffill(
            limit=self.limit
        )
        # Assign the Series, never ``.to_numpy()``. Round-tripping through a
        # NumPy array is what *destroys* a dtype rather than preserving it, and it
        # does so data-dependently: a nullable Int64 comes back as int64 when every
        # gap was filled but as float64 when a leading NA survived, a boolean comes
        # back as object, and a category comes back as plain strings. A fixture
        # whose gaps all fill therefore never catches it.
        for column in targets:
            panel[column] = filled[column]
        return panel
