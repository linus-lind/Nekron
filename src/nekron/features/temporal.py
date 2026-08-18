"""Generic temporal shifting and differencing within each entity.

Shifting is the mechanism for two distinct needs: creating lagged features (a past
value used as a predictor) and creating shifted targets (a future value used as a
forecast objective). A single signed ``periods`` expresses both — positive pulls a
past value forward, negative pulls a future value back — so the look-ahead
direction is explicit in configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_diff, grouped_shift


@dataclass(frozen=True)
class TemporalShift:
    """Shift a column within each entity by one or more signed offsets.

    A positive ``period`` yields the value ``period`` rows in the past (a lag, used
    as a feature); a negative ``period`` yields the value ``-period`` rows in the
    future (a lead, used to build a forecast target). Rows without an in-entity
    neighbor at the offset are ``NaN``.
    """

    input_column: str
    periods: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.periods), self.output_names, "periods")
        if any(p == 0 for p in self.periods):
            raise ValueError("periods must be non-zero (0 would copy the column).")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        return {
            name: grouped_shift(values, ctx, period)
            for period, name in zip(self.periods, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class Difference:
    """Difference a column within each entity, ``x_t - x_{t-period}``, per offset.

    Useful for turning a non-stationary level (a cumulative volume indicator, a
    price) into a stationary change. ``period`` must be positive.
    """

    input_column: str
    periods: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.periods), self.output_names, "periods")
        if any(p <= 0 for p in self.periods):
            raise ValueError(f"periods must be positive; got {self.periods}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        return {
            name: grouped_diff(values, ctx, period)
            for period, name in zip(self.periods, self.output_names, strict=True)
        }
