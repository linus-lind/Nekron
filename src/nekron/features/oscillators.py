"""Bounded technical oscillators.

Wilder-smoothed indicators (RSI) use the recursive RMA form (``alpha = 1/period``,
``adjust=False``). Every ratio is guarded: a flat window (equal highs and lows, or
no gains and no losses) yields ``NaN`` rather than a fabricated neutral value, so
undefined observations are visible downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_ewm, grouped_rolling, grouped_rolling_mad, grouped_shift
from .numeric import safe_divide


@dataclass(frozen=True)
class RSI:
    """Relative Strength Index with Wilder's smoothing over one or more periods."""

    close_column: str
    periods: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.periods), self.output_names, "periods")
        if any(p <= 0 for p in self.periods):
            raise ValueError(f"periods must be positive; got {self.periods}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        delta = close - grouped_shift(close, ctx, 1)
        gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
        loss = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0))
        out: dict[str, FloatArray] = {}
        for period, name in zip(self.periods, self.output_names, strict=True):
            avg_gain = grouped_ewm(
                gain,
                ctx,
                span=None,
                alpha=1.0 / period,
                adjust=False,
                min_periods=self.min_periods,
                how="mean",
            )
            avg_loss = grouped_ewm(
                loss,
                ctx,
                span=None,
                alpha=1.0 / period,
                adjust=False,
                min_periods=self.min_periods,
                how="mean",
            )
            out[name] = 100.0 * safe_divide(avg_gain, avg_gain + avg_loss)
        return out


@dataclass(frozen=True)
class Stochastic:
    """Stochastic oscillator %K and %D over a lookback window.

    ``%K = SMA(smooth_k)`` of ``100 * (C - LL) / (HH - LL)`` (set ``smooth_k = 1``
    for the fast oscillator, ``> 1`` for the slow); ``%D = SMA(smooth_d)`` of %K.
    """

    high_column: str
    low_column: str
    close_column: str
    window: int
    smooth_k: int
    smooth_d: int
    k_name: str
    d_name: str
    min_periods: int

    def __post_init__(self) -> None:
        if self.window <= 0 or self.smooth_k <= 0 or self.smooth_d <= 0:
            raise ValueError("window, smooth_k, and smooth_d must be positive.")
        if self.k_name == self.d_name:
            raise ValueError("k_name and d_name must differ.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.k_name, self.d_name)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        highest = grouped_rolling(
            high, ctx, window=self.window, min_periods=self.min_periods, how="max"
        )
        lowest = grouped_rolling(
            low, ctx, window=self.window, min_periods=self.min_periods, how="min"
        )
        raw_k = 100.0 * safe_divide(close - lowest, highest - lowest)
        # The smoothing sub-windows carry their own (full) min_periods; the main
        # lookback's min_periods governs raw %K above.
        percent_k = grouped_rolling(
            raw_k, ctx, window=self.smooth_k, min_periods=self.smooth_k, how="mean"
        )
        percent_d = grouped_rolling(
            percent_k, ctx, window=self.smooth_d, min_periods=self.smooth_d, how="mean"
        )
        return {self.k_name: percent_k, self.d_name: percent_d}


@dataclass(frozen=True)
class WilliamsR:
    """Williams %R over a lookback window (range ``[-100, 0]``)."""

    high_column: str
    low_column: str
    close_column: str
    window: int
    output_name: str
    min_periods: int

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive; got {self.window}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        highest = grouped_rolling(
            high, ctx, window=self.window, min_periods=self.min_periods, how="max"
        )
        lowest = grouped_rolling(
            low, ctx, window=self.window, min_periods=self.min_periods, how="min"
        )
        value = -100.0 * safe_divide(highest - close, highest - lowest)
        return {self.output_name: value}


@dataclass(frozen=True)
class CCI:
    """Commodity Channel Index using the typical price and mean absolute deviation."""

    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    constant: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")
        if self.constant <= 0.0:
            raise ValueError(f"constant must be positive; got {self.constant}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        typical = (
            column_values(panel, self.high_column)
            + column_values(panel, self.low_column)
            + column_values(panel, self.close_column)
        ) / 3.0
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            mean = grouped_rolling(typical, ctx, window=window, min_periods=window, how="mean")
            mad = grouped_rolling_mad(typical, ctx, window=window)
            out[name] = safe_divide(typical - mean, self.constant * mad)
        return out


@dataclass(frozen=True)
class UltimateOscillator:
    """Ultimate Oscillator: weighted buying-pressure-to-true-range over three periods."""

    high_column: str
    low_column: str
    close_column: str
    periods: tuple[int, int, int]
    weights: tuple[float, float, float]
    output_name: str
    min_periods: int

    def __post_init__(self) -> None:
        if len(self.periods) != 3 or any(p <= 0 for p in self.periods):
            raise ValueError("periods must be three positive integers.")
        if len(self.weights) != 3 or any(w <= 0 for w in self.weights):
            raise ValueError("weights must be three positive numbers.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        prev_close = grouped_shift(close, ctx, 1)
        min_low = np.minimum(low, prev_close)
        buying_pressure = close - min_low
        true_range = np.maximum(high, prev_close) - min_low
        averages = []
        for period in self.periods:
            bp_sum = grouped_rolling(
                buying_pressure, ctx, window=period, min_periods=self.min_periods, how="sum"
            )
            tr_sum = grouped_rolling(
                true_range, ctx, window=period, min_periods=self.min_periods, how="sum"
            )
            averages.append(safe_divide(bp_sum, tr_sum))
        w0, w1, w2 = self.weights
        weighted = w0 * averages[0] + w1 * averages[1] + w2 * averages[2]
        value = 100.0 * weighted / (w0 + w1 + w2)
        return {self.output_name: value}


@dataclass(frozen=True)
class AwesomeOscillator:
    """Awesome Oscillator: fast minus slow SMA of the median price."""

    high_column: str
    low_column: str
    fast_window: int
    slow_window: int
    output_name: str
    min_periods: int

    def __post_init__(self) -> None:
        if not 0 < self.fast_window < self.slow_window:
            raise ValueError("require 0 < fast_window < slow_window.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        median_price = (
            column_values(panel, self.high_column) + column_values(panel, self.low_column)
        ) / 2.0
        fast = grouped_rolling(
            median_price, ctx, window=self.fast_window, min_periods=self.min_periods, how="mean"
        )
        slow = grouped_rolling(
            median_price, ctx, window=self.slow_window, min_periods=self.min_periods, how="mean"
        )
        return {self.output_name: fast - slow}
