"""Calendar and seasonality features derived from the date index level.

These read only the panel's date level, so they are trivially point-in-time. Each
requested attribute produces one numeric column (weekday indices, month/quarter
numbers, and 0/1 boundary flags such as month-end).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nekron.constants import DATE_LEVEL

from .base import FloatArray, PanelContext, check_named, resolve_level

_ATTRIBUTES: dict[str, Callable[[pd.DatetimeIndex], np.ndarray]] = {
    "day_of_week": lambda d: d.dayofweek.to_numpy(),
    "day_of_month": lambda d: d.day.to_numpy(),
    "day_of_year": lambda d: d.dayofyear.to_numpy(),
    "week_of_year": lambda d: d.isocalendar().week.to_numpy(),
    "month": lambda d: d.month.to_numpy(),
    "quarter": lambda d: d.quarter.to_numpy(),
    "year": lambda d: d.year.to_numpy(),
    "is_month_start": lambda d: d.is_month_start.astype(np.float64),
    "is_month_end": lambda d: d.is_month_end.astype(np.float64),
    "is_quarter_start": lambda d: d.is_quarter_start.astype(np.float64),
    "is_quarter_end": lambda d: d.is_quarter_end.astype(np.float64),
    "is_year_start": lambda d: d.is_year_start.astype(np.float64),
    "is_year_end": lambda d: d.is_year_end.astype(np.float64),
}


@dataclass(frozen=True)
class CalendarFeatures:
    """Derive calendar attributes from the panel's date level.

    Parameters
    ----------
    attributes:
        Names of the calendar attributes to compute; one output per attribute.
        Valid names are the keys of the module's attribute table (weekday, month,
        quarter, week/day-of-year, and month/quarter/year boundary flags).
    output_names:
        Names of the produced columns, aligned to ``attributes``.
    date_level:
        Position or name of the date level in the panel index.
    """

    attributes: tuple[str, ...]
    output_names: tuple[str, ...]
    date_level: int | str = DATE_LEVEL

    def __post_init__(self) -> None:
        check_named(len(self.attributes), self.output_names, "attributes")
        unknown = [a for a in self.attributes if a not in _ATTRIBUTES]
        if unknown:
            raise ValueError(
                f"unknown calendar attributes {unknown}; valid: {sorted(_ATTRIBUTES)}."
            )

    @property
    def inputs(self) -> tuple[str, ...]:
        return ()

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        level = resolve_level(ctx.index, self.date_level)
        dates = pd.DatetimeIndex(ctx.index.get_level_values(level))
        return {
            name: np.asarray(_ATTRIBUTES[attribute](dates), dtype=np.float64)
            for attribute, name in zip(self.attributes, self.output_names, strict=True)
        }
