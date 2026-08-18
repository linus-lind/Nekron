"""Generic trailing rolling aggregation of a column within each entity.

A single configurable reducer covers the many features that are "an N-day
statistic of some column" — average volume, average turnover, rolling skewness or
kurtosis of returns, rolling median, rolling quantile — without a bespoke class
for each. Reducers that need extra parameters read them from the fields that apply
(``ddof`` for ``std`` / ``var``; ``quantile`` for ``quantile``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_rolling

_HOWS = ("mean", "std", "var", "sum", "min", "max", "median", "skew", "kurt", "quantile")


@dataclass(frozen=True)
class RollingAggregation:
    """Apply a rolling reducer to a column over one or more windows.

    Parameters
    ----------
    input_column:
        Column to aggregate.
    windows, output_names:
        Trailing windows and their output column names.
    how:
        Reducer: one of ``mean``, ``std``, ``var``, ``sum``, ``min``, ``max``,
        ``median``, ``skew``, ``kurt``, ``quantile``.
    min_periods:
        Minimum non-``NaN`` observations for a non-``NaN`` result.
    ddof:
        Delta degrees of freedom (used by ``std`` / ``var``).
    quantile:
        Quantile level in ``[0, 1]`` (required when ``how == "quantile"``).
    """

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    how: str
    min_periods: int
    ddof: int
    quantile: float | None = None

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if self.how not in _HOWS:
            raise ValueError(f"how must be one of {_HOWS}; got {self.how!r}.")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")
        if self.how == "quantile" and self.quantile is None:
            raise ValueError("quantile reducer requires a quantile level.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        return {
            name: grouped_rolling(
                values,
                ctx,
                window=window,
                min_periods=self.min_periods,
                how=self.how,
                ddof=self.ddof,
                quantile=self.quantile,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }
