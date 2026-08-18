"""Directional-movement and trend-strength features.

These indicators quantify the strength and direction of a trend rather than its
magnitude: the Wilder directional system (ADX / +DI / -DI), the Aroon time-since-
extreme measures, the double-smoothed True Strength Index, and the slope of a
rolling linear regression of price on time. All use only trailing data and are
known at the close of their labeled day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_ewm, grouped_rolling, grouped_shift
from .numeric import safe_divide, safe_log


@dataclass(frozen=True)
class ADX:
    """Average Directional Index with +DI and -DI (Wilder smoothing).

    Produces three columns (ADX, +DI, -DI) in that order. Directional movement
    uses the prior bar; a flat window (zero smoothed true range or zero directional
    sum) yields ``NaN`` for the affected ratio.
    """

    high_column: str
    low_column: str
    close_column: str
    period: int
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(3, self.output_names, "ADX components (adx, plus_di, minus_di)")
        if self.period <= 0:
            raise ValueError(f"period must be positive; got {self.period}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        prev_high = grouped_shift(high, ctx, 1)
        prev_low = grouped_shift(low, ctx, 1)
        prev_close = grouped_shift(close, ctx, 1)

        true_range = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        true_range = np.where(np.isnan(prev_close), high - low, true_range)
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
        invalid = ~(np.isfinite(up_move) & np.isfinite(down_move))
        plus_dm[invalid] = np.nan
        minus_dm[invalid] = np.nan

        alpha = 1.0 / self.period

        def wilder(values: FloatArray) -> FloatArray:
            return grouped_ewm(
                values,
                ctx,
                span=None,
                alpha=alpha,
                adjust=False,
                min_periods=self.min_periods,
                how="mean",
            )

        atr = wilder(true_range)
        plus_di = 100.0 * safe_divide(wilder(plus_dm), atr)
        minus_di = 100.0 * safe_divide(wilder(minus_dm), atr)
        dx = 100.0 * safe_divide(np.abs(plus_di - minus_di), plus_di + minus_di)
        adx = wilder(dx)
        adx_name, plus_name, minus_name = self.output_names
        return {adx_name: adx, plus_name: plus_di, minus_name: minus_di}


def _recency_of_extreme(
    values: FloatArray, ctx: PanelContext, window: int, kind: str
) -> FloatArray:
    """Bars since the most recent window maximum/minimum (0 = current bar).

    Full windows only; boundary-straddling or NaN-containing windows yield ``NaN``.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        view = np.lib.stride_tricks.sliding_window_view(arr, window)
        reversed_view = view[:, ::-1]
        recency = (
            np.argmax(reversed_view, axis=1) if kind == "max" else np.argmin(reversed_view, axis=1)
        )
        recency = recency.astype(np.float64)
        recency[np.isnan(view).any(axis=1)] = np.nan
        out[window - 1 :] = recency
    out[ctx.within_entity < window - 1] = np.nan
    return out


@dataclass(frozen=True)
class Aroon:
    """Aroon Up, Aroon Down, and the Aroon Oscillator over a lookback window.

    Produces three columns (up, down, oscillator) in that order. A fresh extreme at
    the current bar scores 100.
    """

    high_column: str
    low_column: str
    window: int
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(3, self.output_names, "Aroon components (up, down, oscillator)")
        if self.window <= 0:
            raise ValueError(f"window must be positive; got {self.window}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        since_high = _recency_of_extreme(
            column_values(panel, self.high_column), ctx, self.window, "max"
        )
        since_low = _recency_of_extreme(
            column_values(panel, self.low_column), ctx, self.window, "min"
        )
        aroon_up = 100.0 * (self.window - since_high) / self.window
        aroon_down = 100.0 * (self.window - since_low) / self.window
        up_name, down_name, osc_name = self.output_names
        return {up_name: aroon_up, down_name: aroon_down, osc_name: aroon_up - aroon_down}


@dataclass(frozen=True)
class TrueStrengthIndex:
    """True Strength Index: double-EMA-smoothed momentum normalized by its magnitude."""

    close_column: str
    long_span: int
    short_span: int
    output_name: str
    adjust: bool
    min_periods: int

    def __post_init__(self) -> None:
        if self.long_span <= 0 or self.short_span <= 0:
            raise ValueError("long_span and short_span must be positive.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        change = close - grouped_shift(close, ctx, 1)

        def double_ema(values: FloatArray) -> FloatArray:
            first = grouped_ewm(
                values,
                ctx,
                span=self.long_span,
                alpha=None,
                adjust=self.adjust,
                min_periods=self.min_periods,
                how="mean",
            )
            return grouped_ewm(
                first,
                ctx,
                span=self.short_span,
                alpha=None,
                adjust=self.adjust,
                min_periods=self.min_periods,
                how="mean",
            )

        smoothed = double_ema(change)
        smoothed_abs = double_ema(np.abs(change))
        return {self.output_name: 100.0 * safe_divide(smoothed, smoothed_abs)}


@dataclass(frozen=True)
class LinearTrendSlope:
    """Slope of a rolling ordinary-least-squares fit of price on time.

    Regresses the (optionally log) price on the within-window time index over each
    trailing window; the slope is the per-bar trend. With ``use_log`` the slope is
    an approximate per-bar growth rate.
    """

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    use_log: bool

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w < 2 for w in self.windows):
            raise ValueError(f"windows must be >= 2; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        raw = column_values(panel, self.input_column)
        y = safe_log(raw) if self.use_log else raw
        time = ctx.within_entity.astype(np.float64)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            sum_t = grouped_rolling(time, ctx, window=window, min_periods=window, how="sum")
            sum_tt = grouped_rolling(time * time, ctx, window=window, min_periods=window, how="sum")
            sum_y = grouped_rolling(y, ctx, window=window, min_periods=window, how="sum")
            sum_ty = grouped_rolling(time * y, ctx, window=window, min_periods=window, how="sum")
            numerator = window * sum_ty - sum_t * sum_y
            denominator = window * sum_tt - sum_t * sum_t
            out[name] = safe_divide(numerator, denominator)
        return out
