"""Corporate-action adjustment: compounding a per-period factor and applying it.

:class:`CumulativeFactor` turns a per-period split/price factor (one except on a
distribution's ex-date) into the cumulative factor needed to make a series
continuous; :class:`CorporateAdjustment` applies a factor column to price- and
share-like columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from nekron.constants import ENTITY_LEVEL
from nekron.panel import PanelGrain

from .base import ALL_GRAINS, entity_time_ordered, require_columns, resolve_level

_OPERATIONS = ("divide", "multiply")


@dataclass(frozen=True)
class Adjustment:
    """One adjustment: apply ``factor_column`` to each of ``columns``.

    Parameters
    ----------
    columns:
        Columns to adjust.
    factor_column:
        Column holding the per-row adjustment factor.
    operation:
        ``"divide"`` divides each column by the factor (price-style adjustment);
        ``"multiply"`` multiplies by it (share-style adjustment).

    A ``"divide"`` factor column must be non-zero and finite; division is applied
    verbatim, so a zero factor yields an infinite value rather than being masked.
    """

    columns: tuple[str, ...]
    factor_column: str
    operation: str

    def __post_init__(self) -> None:
        if self.operation not in _OPERATIONS:
            raise ValueError(f"operation must be one of {_OPERATIONS}; got {self.operation!r}.")
        if not self.columns:
            raise ValueError("adjustment must name at least one column.")


@dataclass(frozen=True)
class CorporateAdjustment:
    """Apply a sequence of factor adjustments to the panel, in place."""

    grains: ClassVar[tuple[PanelGrain, ...]] = ALL_GRAINS
    """Pure column arithmetic: no index level is read, so this is meaningful for
    every grain."""

    adjustments: tuple[Adjustment, ...]

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        for adj in self.adjustments:
            require_columns(panel, (adj.factor_column, *adj.columns))
            factor = panel[adj.factor_column].to_numpy()
            for column in adj.columns:
                values = panel[column].to_numpy()
                panel[column] = values / factor if adj.operation == "divide" else values * factor
        return panel


@dataclass(frozen=True)
class CumulativeFactor:
    """Compound a per-period adjustment factor into a cumulative factor per entity.

    Daily files that provide a per-period split/price factor — one on ordinary days,
    the corporate-action ratio on a distribution's ex-date — require it to be
    compounded before it can adjust a series. The cumulative factor at row ``t`` is
    the product of every *later* in-entity per-period factor, so dividing prices (or
    multiplying shares and volume) by it back-adjusts the series to the most recent
    basis and makes it continuous across splits; the ex-date's own factor adjusts
    only the dates before it. Non-finite or non-positive factor values are treated
    as one (no adjustment).

    Parameters
    ----------
    factor_column:
        Per-period factor to compound.
    output_column:
        Column the cumulative factor is written to.
    entity_level:
        Position or name of the entity level to group by.
    """

    factor_column: str
    output_column: str
    entity_level: int | str = ENTITY_LEVEL

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        require_columns(panel, (self.factor_column,))
        if not entity_time_ordered(panel.index):
            panel = panel.sort_index()
        pos = resolve_level(panel.index, self.entity_level)
        factor = panel[self.factor_column].to_numpy()
        safe = np.where(np.isfinite(factor) & (factor > 0.0), factor, 1.0)
        grouped = pd.Series(safe, index=panel.index).groupby(level=pos, sort=False)
        # cumulative_t = (product of the whole entity) / (product up to and including
        # t) = product over rows strictly after t, so the ex-date factor lands only
        # on earlier rows. All factors are positive, so the division is well defined.
        entity_total = grouped.transform("prod").to_numpy()
        inclusive = grouped.cumprod().to_numpy()
        panel[self.output_column] = entity_total / inclusive
        return panel
