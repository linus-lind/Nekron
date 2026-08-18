"""Per-entity, time-series normalization of a column against its own history.

Unlike the cross-sectional transforms, these standardize each value against a
trailing window of the same entity's past, which is useful for turning a level
(volume, an oscillator, a cumulative indicator) into a stationary, comparable
signal. All windows are trailing and inclusive of the current row, so every value
is known at the close of its row.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_rolling, grouped_rolling_rank
from .numeric import safe_divide


@dataclass(frozen=True)
class RollingZScore:
    """Trailing per-entity z-score ``(x_t - mean_n) / std_n`` over each window."""

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    ddof: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            mean = grouped_rolling(
                values, ctx, window=window, min_periods=self.min_periods, how="mean"
            )
            std = grouped_rolling(
                values, ctx, window=window, min_periods=self.min_periods, how="std", ddof=self.ddof
            )
            out[name] = safe_divide(values - mean, std)
        return out


@dataclass(frozen=True)
class RollingMinMax:
    """Trailing per-entity min-max scaling ``(x_t - min_n) / (max_n - min_n)``."""

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            low = grouped_rolling(
                values, ctx, window=window, min_periods=self.min_periods, how="min"
            )
            high = grouped_rolling(
                values, ctx, window=window, min_periods=self.min_periods, how="max"
            )
            out[name] = safe_divide(values - low, high - low)
        return out


@dataclass(frozen=True)
class RollingRank:
    """Trailing per-entity percentile rank of the current value within its window."""

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    method: str

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        return {
            name: grouped_rolling_rank(
                values,
                ctx,
                window=window,
                min_periods=self.min_periods,
                pct=True,
                method=self.method,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }
