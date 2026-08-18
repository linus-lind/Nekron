"""Liquidity, spread, and price-impact features from daily OHLCV data.

Spread and impact proxies (Roll, Corwin-Schultz, Kyle's lambda) are estimated over
trailing windows and aligned so every value is known at the close of its labeled
day. Ratios guard zero or missing denominators to ``NaN``; the Amihud average and
the zero-return proportion skip invalid days rather than treating them as extreme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_rolling, grouped_shift
from .numeric import safe_divide, safe_log

_CS_CONST = 3.0 - 2.0 * math.sqrt(2.0)


def _entity_mean(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Per-entity mean broadcast to every row of the entity."""
    return np.asarray(
        pd.Series(values).groupby(ctx.entity_codes, sort=False).transform("mean").to_numpy(),
        dtype=np.float64,
    )


def _rolling_cov(
    x: FloatArray, y: FloatArray, ctx: PanelContext, window: int, min_periods: int
) -> FloatArray:
    """Population covariance of two aligned series over a trailing window.

    Both series are first centered by their per-entity mean. Covariance is
    invariant to that shift, but it keeps the two-pass moments ``O(variance)``
    rather than ``O(mean^2)``, avoiding catastrophic cancellation when a series has
    a large mean relative to its variance (e.g. dollar-volume order flow).
    """
    xc = x - _entity_mean(x, ctx)
    yc = y - _entity_mean(y, ctx)
    mean_x = grouped_rolling(xc, ctx, window=window, min_periods=min_periods, how="mean")
    mean_y = grouped_rolling(yc, ctx, window=window, min_periods=min_periods, how="mean")
    mean_xy = grouped_rolling(xc * yc, ctx, window=window, min_periods=min_periods, how="mean")
    return mean_xy - mean_x * mean_y


@dataclass(frozen=True)
class AmihudIlliquidity:
    """Amihud (2002) illiquidity: rolling mean of ``|return| / dollar-volume``.

    Days with non-positive dollar volume are excluded from each window's average.

    Parameters
    ----------
    return_column, dollar_volume_column:
        Per-period return and dollar-volume series.
    windows, output_names:
        Trailing windows and their output column names.
    min_periods:
        Minimum valid days required within a window.
    scale:
        Multiplier for the average (commonly ``1e6``, i.e. dollar volume in
        millions).
    """

    return_column: str
    dollar_volume_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    scale: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column, self.dollar_volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        abs_return = np.abs(column_values(panel, self.return_column))
        dollar_volume = column_values(panel, self.dollar_volume_column)
        daily = safe_divide(abs_return, dollar_volume)
        return {
            name: self.scale
            * grouped_rolling(daily, ctx, window=window, min_periods=self.min_periods, how="mean")
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class DollarVolume:
    """Dollar volume ``price * volume`` (or its log with ``log=True``)."""

    price_column: str
    volume_column: str
    output_name: str
    log: bool

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.price_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        value = column_values(panel, self.price_column) * column_values(panel, self.volume_column)
        if self.log:
            value = np.log1p(np.where(value >= 0.0, value, np.nan))
        return {self.output_name: value}


@dataclass(frozen=True)
class Turnover:
    """Share turnover ``volume / shares_outstanding``."""

    volume_column: str
    shares_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.volume_column, self.shares_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        value = safe_divide(
            column_values(panel, self.volume_column), column_values(panel, self.shares_column)
        )
        return {self.output_name: value}


@dataclass(frozen=True)
class MarketCap:
    """Market capitalization ``price * shares_outstanding`` (or its log)."""

    price_column: str
    shares_column: str
    output_name: str
    log: bool

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.price_column, self.shares_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        value = column_values(panel, self.price_column) * column_values(panel, self.shares_column)
        return {self.output_name: safe_log(value) if self.log else value}


@dataclass(frozen=True)
class RollSpread:
    """Roll (1984) effective spread ``2 * sqrt(-cov(dP_t, dP_{t-1}))`` per window.

    The estimate is ``NaN`` where the serial covariance of price changes is
    non-negative (the model is inapplicable). With ``normalize=True`` the spread is
    divided by the window's mean price to give a proportional spread.
    """

    price_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    normalize: bool

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 1 for w in self.windows):
            raise ValueError(f"windows must be > 1; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.price_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.price_column)
        delta = price - grouped_shift(price, ctx, 1)
        delta_lag = grouped_shift(delta, ctx, 1)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            cov = _rolling_cov(delta, delta_lag, ctx, window, self.min_periods)
            spread = np.where(cov < 0.0, 2.0 * np.sqrt(np.where(cov < 0.0, -cov, np.nan)), np.nan)
            if self.normalize:
                mean_price = grouped_rolling(
                    price, ctx, window=window, min_periods=self.min_periods, how="mean"
                )
                spread = safe_divide(spread, mean_price)
            out[name] = spread
        return out


@dataclass(frozen=True)
class CorwinSchultzSpread:
    """Corwin-Schultz (2012) high-low bid-ask spread, averaged over a window.

    Each two-day estimate is aligned to the later day, overnight-gap adjusted, and
    floored at zero before averaging over the trailing window.
    """

    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

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
        high_prev = grouped_shift(high, ctx, 1)
        low_prev = grouped_shift(low, ctx, 1)
        close_prev = grouped_shift(close, ctx, 1)

        # Overnight-gap adjustment of the later day's range (used in both beta and
        # gamma): if the prior close lies outside the current range, translate the
        # current high/low so the two sessions are contiguous.
        gap_up = close_prev < low
        gap_down = close_prev > high
        high_adj = np.where(gap_up, high - (low - close_prev), np.where(gap_down, close_prev, high))
        low_adj = np.where(gap_up, close_prev, np.where(gap_down, low + (close_prev - high), low))

        beta = (safe_log(high_prev) - safe_log(low_prev)) ** 2 + (
            safe_log(high_adj) - safe_log(low_adj)
        ) ** 2
        high_two = np.maximum(high_prev, high_adj)
        low_two = np.minimum(low_prev, low_adj)
        gamma = (safe_log(high_two) - safe_log(low_two)) ** 2

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_CONST - np.sqrt(gamma / _CS_CONST)
        spread = np.maximum(2.0 * np.tanh(alpha / 2.0), 0.0)

        return {
            name: grouped_rolling(
                spread, ctx, window=window, min_periods=self.min_periods, how="mean"
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class KyleLambda:
    """Kyle's lambda: rolling regression slope of return on signed dollar volume.

    ``lambda = cov(r, q) / var(q)`` with ``q = sign(r) * dollar_volume`` over the
    trailing window; larger values indicate larger price impact (less liquid).
    """

    return_column: str
    dollar_volume_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 1 for w in self.windows):
            raise ValueError(f"windows must be > 1; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column, self.dollar_volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        returns = column_values(panel, self.return_column)
        signed_flow = np.sign(returns) * column_values(panel, self.dollar_volume_column)
        # Center flow by its per-entity mean so the rolling variance is computed on
        # deviations, not on values near 1e10; the slope is unchanged by the shift.
        centered_flow = signed_flow - _entity_mean(signed_flow, ctx)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            cov = _rolling_cov(returns, signed_flow, ctx, window, self.min_periods)
            var_q = grouped_rolling(
                centered_flow, ctx, window=window, min_periods=self.min_periods, how="var", ddof=0
            )
            out[name] = safe_divide(cov, var_q)
        return out


@dataclass(frozen=True)
class ZeroReturnFraction:
    """Lesmond-Ogden-Trzcinka zero-return proportion over a trailing window.

    A day counts as zero when ``|return| < tolerance``; days with a missing return
    are excluded from both numerator and denominator.
    """

    return_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    tolerance: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")
        if self.tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative; got {self.tolerance}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        returns = column_values(panel, self.return_column)
        is_zero = np.where(
            np.isnan(returns), np.nan, (np.abs(returns) < self.tolerance).astype(np.float64)
        )
        return {
            name: grouped_rolling(
                is_zero, ctx, window=window, min_periods=self.min_periods, how="mean"
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }
