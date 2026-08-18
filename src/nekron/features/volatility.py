"""Volatility and risk features: close-to-close, range-based, and drawdown.

Each estimator produces a *daily* volatility that is multiplied by an explicit
``annualization_factor`` (``sqrt(trading_days)`` for annualized output, ``1.0``
for daily) so the scaling convention is always visible in configuration. Range
estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) follow the
published variance forms exactly; every log ratio is guarded so a non-positive
price yields ``NaN`` rather than a spurious zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import (
    grouped_ewm,
    grouped_rolling,
    grouped_rolling_max_drawdown,
    grouped_rolling_ulcer,
    grouped_shift,
)
from .numeric import safe_divide, safe_log

_LN2 = math.log(2.0)
_PARKINSON_C = 1.0 / (4.0 * _LN2)
_GK_C = 2.0 * _LN2 - 1.0


def _vol(variance: FloatArray, factor: float) -> FloatArray:
    """Daily volatility from a daily variance, floored at zero and scaled."""
    return np.sqrt(np.maximum(variance, 0.0)) * factor


@dataclass(frozen=True)
class RealizedVolatility:
    """Rolling standard deviation of a return column (close-to-close volatility).

    Parameters
    ----------
    return_column:
        Per-period return series (log returns are the usual input).
    windows:
        Trailing windows in rows; one output column per window.
    output_names:
        Names of the produced columns, aligned to ``windows``.
    min_periods:
        Minimum non-``NaN`` returns required within a window.
    ddof:
        Delta degrees of freedom for the standard deviation.
    annualization_factor:
        Multiplier applied to the daily volatility (``sqrt(trading_days)`` to
        annualize, ``1.0`` to leave it daily).
    """

    return_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    ddof: int
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        returns = column_values(panel, self.return_column)
        return {
            name: grouped_rolling(
                returns,
                ctx,
                window=window,
                min_periods=self.min_periods,
                how="std",
                ddof=self.ddof,
            )
            * self.annualization_factor
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class EWMAVolatility:
    """Exponentially weighted volatility ``sqrt(EWMA(r^2))`` (RiskMetrics nowcast).

    Parameters
    ----------
    return_column:
        Per-period return series.
    output_name:
        Name of the produced column.
    alpha:
        Smoothing factor in ``(0, 1]``; equals ``1 - lambda`` for a RiskMetrics
        decay ``lambda`` (e.g. ``alpha = 0.06`` for the daily ``lambda = 0.94``).
    min_periods:
        Minimum observations before a value is emitted.
    annualization_factor:
        Multiplier applied to the daily volatility.
    """

    return_column: str
    output_name: str
    alpha: float
    min_periods: int
    annualization_factor: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        squared = column_values(panel, self.return_column) ** 2
        variance = grouped_ewm(
            squared,
            ctx,
            span=None,
            alpha=self.alpha,
            adjust=False,
            min_periods=self.min_periods,
            how="mean",
        )
        return {self.output_name: _vol(variance, self.annualization_factor)}


@dataclass(frozen=True)
class ParkinsonVolatility:
    """Parkinson (1980) high-low range volatility."""

    high_column: str
    low_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_hl = safe_log(column_values(panel, self.high_column)) - safe_log(
            column_values(panel, self.low_column)
        )
        term = _PARKINSON_C * log_hl**2
        return {
            name: _vol(
                grouped_rolling(term, ctx, window=window, min_periods=self.min_periods, how="mean"),
                self.annualization_factor,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class GarmanKlassVolatility:
    """Garman-Klass (1980) OHLC volatility (practical form)."""

    open_column: str
    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.open_column, self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_hl = safe_log(column_values(panel, self.high_column)) - safe_log(
            column_values(panel, self.low_column)
        )
        log_co = safe_log(column_values(panel, self.close_column)) - safe_log(
            column_values(panel, self.open_column)
        )
        term = 0.5 * log_hl**2 - _GK_C * log_co**2
        return {
            name: _vol(
                grouped_rolling(term, ctx, window=window, min_periods=self.min_periods, how="mean"),
                self.annualization_factor,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class RogersSatchellVolatility:
    """Rogers-Satchell (1991) drift-independent OHLC volatility."""

    open_column: str
    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.open_column, self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_o = safe_log(column_values(panel, self.open_column))
        u = safe_log(column_values(panel, self.high_column)) - log_o
        d = safe_log(column_values(panel, self.low_column)) - log_o
        c = safe_log(column_values(panel, self.close_column)) - log_o
        term = u * (u - c) + d * (d - c)
        return {
            name: _vol(
                grouped_rolling(term, ctx, window=window, min_periods=self.min_periods, how="mean"),
                self.annualization_factor,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class YangZhangVolatility:
    """Yang-Zhang (2000) volatility combining overnight, open-close, and RS terms."""

    open_column: str
    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w < 2 for w in self.windows):
            raise ValueError(f"windows must be >= 2 for Yang-Zhang; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.open_column, self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_o = safe_log(column_values(panel, self.open_column))
        log_h = safe_log(column_values(panel, self.high_column))
        log_l = safe_log(column_values(panel, self.low_column))
        log_c = safe_log(column_values(panel, self.close_column))
        overnight = log_o - grouped_shift(log_c, ctx, 1)
        open_close = log_c - log_o
        u = log_h - log_o
        d = log_l - log_o
        rs_term = u * (u - open_close) + d * (d - open_close)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            var_o = grouped_rolling(
                overnight, ctx, window=window, min_periods=self.min_periods, how="var", ddof=1
            )
            var_c = grouped_rolling(
                open_close, ctx, window=window, min_periods=self.min_periods, how="var", ddof=1
            )
            var_rs = grouped_rolling(
                rs_term, ctx, window=window, min_periods=self.min_periods, how="mean"
            )
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            variance = var_o + k * var_c + (1.0 - k) * var_rs
            out[name] = _vol(variance, self.annualization_factor)
        return out


@dataclass(frozen=True)
class AverageTrueRange:
    """Average True Range (Wilder) over one or more periods, optionally normalized.

    True Range uses the prior close within each entity; the smoothing is Wilder's
    RMA (``alpha = 1/window``). When ``normalize`` is set, ATR is divided by the
    close to give the scale-free ATR%.
    """

    high_column: str
    low_column: str
    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    normalize: bool

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
        prev_close = grouped_shift(close, ctx, 1)
        true_range = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        # First bar of each entity has no prior close: true range is the day's range.
        true_range = np.where(np.isnan(prev_close), high - low, true_range)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            atr = grouped_ewm(
                true_range,
                ctx,
                span=None,
                alpha=1.0 / window,
                adjust=False,
                min_periods=self.min_periods,
                how="mean",
            )
            out[name] = safe_divide(atr, close) if self.normalize else atr
        return out


@dataclass(frozen=True)
class DownsideDeviation:
    """Rolling downside deviation of returns below a threshold (Sortino convention)."""

    return_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int
    threshold: float
    annualization_factor: float

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        shortfall = np.minimum(column_values(panel, self.return_column) - self.threshold, 0.0)
        squared = shortfall**2
        return {
            name: _vol(
                grouped_rolling(
                    squared, ctx, window=window, min_periods=self.min_periods, how="mean"
                ),
                self.annualization_factor,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class BollingerBands:
    """Bollinger %B and bandwidth of a price column over one or more windows.

    Produces two columns per window: ``pct_b_names[i]`` = ``(P - lower)/(upper -
    lower)`` and ``bandwidth_names[i]`` = ``(upper - lower)/middle``, with
    ``middle = SMA_n`` and bands at ``middle +/- num_std * rolling_std``.
    """

    close_column: str
    windows: tuple[int, ...]
    pct_b_names: tuple[str, ...]
    bandwidth_names: tuple[str, ...]
    num_std: float
    min_periods: int
    ddof: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.pct_b_names, "windows")
        check_named(len(self.windows), self.bandwidth_names, "windows")
        if set(self.pct_b_names) & set(self.bandwidth_names):
            raise ValueError("%B and bandwidth output names must be disjoint.")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.pct_b_names + self.bandwidth_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        out: dict[str, FloatArray] = {}
        for window, pct_name, bw_name in zip(
            self.windows, self.pct_b_names, self.bandwidth_names, strict=True
        ):
            middle = grouped_rolling(
                close, ctx, window=window, min_periods=self.min_periods, how="mean"
            )
            sigma = grouped_rolling(
                close, ctx, window=window, min_periods=self.min_periods, how="std", ddof=self.ddof
            )
            span = 2.0 * self.num_std * sigma
            lower = middle - self.num_std * sigma
            out[pct_name] = safe_divide(close - lower, span)
            out[bw_name] = safe_divide(span, middle)
        return out


@dataclass(frozen=True)
class UlcerIndex:
    """Ulcer Index: RMS percentage drawdown from a within-window running high.

    The running maximum is anchored at each window's start, so a value is defined
    once the entity has ``window`` observations.
    """

    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        return {
            name: grouped_rolling_ulcer(close, ctx, window=window)
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class MaxDrawdown:
    """Worst peak-to-trough decline within each trailing window, a positive fraction.

    Uses a causal running maximum inside the window, so a value is defined once the
    entity has ``window`` observations.
    """

    close_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        return {
            name: grouped_rolling_max_drawdown(close, ctx, window=window)
            for window, name in zip(self.windows, self.output_names, strict=True)
        }
