"""Volume and price-volume features.

Cumulative indicators (OBV, ADL, VPT) accumulate within each entity and are
non-stationary levels; pair them with a rolling normalization or difference for
modeling. The money-flow multiplier is defined as zero on a zero-range bar
(``H == L``), matching the standard convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_cumsum, grouped_ewm, grouped_rolling, grouped_shift
from .numeric import safe_divide


def _money_flow_multiplier(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    """``((C - L) - (H - C)) / (H - L)``, defined as ``0`` when ``H == L``."""
    rng = high - low
    multiplier = safe_divide((close - low) - (high - close), rng)
    return np.where(rng == 0.0, 0.0, multiplier)


@dataclass(frozen=True)
class OnBalanceVolume:
    """On-Balance Volume: cumulative signed volume by close-to-close direction."""

    close_column: str
    volume_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        volume = column_values(panel, self.volume_column)
        direction = np.sign(close - grouped_shift(close, ctx, 1))
        return {self.output_name: grouped_cumsum(direction * volume, ctx)}


@dataclass(frozen=True)
class AccumulationDistribution:
    """Accumulation/Distribution Line: cumulative money-flow volume."""

    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        multiplier = _money_flow_multiplier(
            column_values(panel, self.high_column),
            column_values(panel, self.low_column),
            column_values(panel, self.close_column),
        )
        money_flow_volume = multiplier * column_values(panel, self.volume_column)
        return {self.output_name: grouped_cumsum(money_flow_volume, ctx)}


@dataclass(frozen=True)
class ChaikinMoneyFlow:
    """Chaikin Money Flow: money-flow volume over volume across a window."""

    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        volume = column_values(panel, self.volume_column)
        multiplier = _money_flow_multiplier(
            column_values(panel, self.high_column),
            column_values(panel, self.low_column),
            column_values(panel, self.close_column),
        )
        money_flow_volume = multiplier * volume
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            mfv_sum = grouped_rolling(
                money_flow_volume, ctx, window=window, min_periods=self.min_periods, how="sum"
            )
            vol_sum = grouped_rolling(
                volume, ctx, window=window, min_periods=self.min_periods, how="sum"
            )
            out[name] = safe_divide(mfv_sum, vol_sum)
        return out


@dataclass(frozen=True)
class MoneyFlowIndex:
    """Money Flow Index: volume-weighted RSI of the typical price."""

    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        typical = (
            column_values(panel, self.high_column)
            + column_values(panel, self.low_column)
            + column_values(panel, self.close_column)
        ) / 3.0
        raw_money_flow = typical * column_values(panel, self.volume_column)
        change = typical - grouped_shift(typical, ctx, 1)
        positive = np.where(change > 0.0, raw_money_flow, 0.0)
        negative = np.where(change < 0.0, raw_money_flow, 0.0)
        invalid = np.isnan(change)
        positive[invalid] = np.nan
        negative[invalid] = np.nan
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            pmf = grouped_rolling(
                positive, ctx, window=window, min_periods=self.min_periods, how="sum"
            )
            nmf = grouped_rolling(
                negative, ctx, window=window, min_periods=self.min_periods, how="sum"
            )
            out[name] = 100.0 * safe_divide(pmf, pmf + nmf)
        return out


@dataclass(frozen=True)
class ForceIndex:
    """Elder's Force Index: EMA-smoothed ``(C_t - C_{t-1}) * volume``."""

    close_column: str
    volume_column: str
    span: int
    output_name: str
    adjust: bool
    min_periods: int

    def __post_init__(self) -> None:
        if self.span <= 0:
            raise ValueError(f"span must be positive; got {self.span}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        raw = (close - grouped_shift(close, ctx, 1)) * column_values(panel, self.volume_column)
        smoothed = grouped_ewm(
            raw,
            ctx,
            span=self.span,
            alpha=None,
            adjust=self.adjust,
            min_periods=self.min_periods,
            how="mean",
        )
        return {self.output_name: smoothed}


@dataclass(frozen=True)
class EaseOfMovement:
    """Ease of Movement: SMA of distance moved divided by a volume box ratio.

    A zero-range bar (``H == L``) contributes zero; a zero-volume bar yields
    ``NaN``. ``volume_scale`` is the arbitrary volume divisor (library-specific);
    it only rescales the output.
    """

    high_column: str
    low_column: str
    volume_column: str
    window: int
    output_name: str
    volume_scale: float
    min_periods: int

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive; got {self.window}.")
        if self.volume_scale <= 0.0:
            raise ValueError(f"volume_scale must be positive; got {self.volume_scale}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        volume = column_values(panel, self.volume_column)
        midpoint = (high + low) / 2.0
        distance_moved = midpoint - grouped_shift(midpoint, ctx, 1)
        emv = safe_divide(distance_moved * (high - low), volume / self.volume_scale)
        smoothed = grouped_rolling(
            emv, ctx, window=self.window, min_periods=self.min_periods, how="mean"
        )
        return {self.output_name: smoothed}


@dataclass(frozen=True)
class VolumePriceTrend:
    """Volume Price Trend: cumulative ``volume * (C_t - C_{t-1}) / C_{t-1}``."""

    close_column: str
    volume_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.close_column, self.volume_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        close = column_values(panel, self.close_column)
        prev_close = grouped_shift(close, ctx, 1)
        contribution = column_values(panel, self.volume_column) * safe_divide(
            close - prev_close, prev_close
        )
        return {self.output_name: grouped_cumsum(contribution, ctx)}
