"""Momentum and moving-average trend features.

Moving averages come in simple (``sma``) and exponential (``ema``) flavors; the
exponential smoothing convention (``adjust``) is always explicit because it
changes warm-up values. Momentum is parameterized by a near and far offset so any
"skip" variant — including the canonical 12-1 momentum (near 21, far 252) — is
expressed by day offsets rather than a hardcoded nickname.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_ewm, grouped_rolling, grouped_shift
from .numeric import safe_divide, safe_log

_MA_KINDS = ("sma", "ema")


def _check_kind(kind: str) -> None:
    if kind not in _MA_KINDS:
        raise ValueError(f"kind must be one of {_MA_KINDS}; got {kind!r}.")


def moving_average(
    values: FloatArray,
    ctx: PanelContext,
    *,
    window: int,
    kind: str,
    min_periods: int,
    adjust: bool,
) -> FloatArray:
    """Simple or exponential moving average of a per-entity series.

    ``kind="ema"`` interprets ``window`` as the EMA span (``alpha = 2/(window+1)``).
    """
    if kind == "sma":
        return grouped_rolling(values, ctx, window=window, min_periods=min_periods, how="mean")
    return grouped_ewm(
        values, ctx, span=window, alpha=None, adjust=adjust, min_periods=min_periods, how="mean"
    )


@dataclass(frozen=True)
class Momentum:
    """Price momentum ``P_{t-near} / P_{t-far} - 1`` (or log form) with an optional skip.

    Setting ``near = 0`` gives plain trailing momentum over ``far`` rows; setting
    ``near > 0`` skips the most recent ``near`` rows (e.g. ``near = 21, far = 252``
    is 12-1 momentum). A negative sign of the output is short-term reversal.

    Parameters
    ----------
    input_column:
        Price column.
    near_offsets, far_offsets:
        Parallel tuples of the window endpoints in rows, with ``far > near >= 0``.
    output_names:
        Names of the produced columns, aligned to the offset pairs.
    method:
        ``"simple"`` or ``"log"``.
    """

    input_column: str
    near_offsets: tuple[int, ...]
    far_offsets: tuple[int, ...]
    output_names: tuple[str, ...]
    method: str

    def __post_init__(self) -> None:
        if len(self.near_offsets) != len(self.far_offsets):
            raise ValueError("near_offsets and far_offsets must have equal length.")
        check_named(len(self.far_offsets), self.output_names, "offset pairs")
        if self.method not in ("simple", "log"):
            raise ValueError(f"method must be 'simple' or 'log'; got {self.method!r}.")
        for near, far in zip(self.near_offsets, self.far_offsets, strict=True):
            if near < 0 or far <= near:
                raise ValueError(f"require far > near >= 0; got near={near}, far={far}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.input_column)
        base = safe_log(price) if self.method == "log" else price
        out: dict[str, FloatArray] = {}
        for near, far, name in zip(
            self.near_offsets, self.far_offsets, self.output_names, strict=True
        ):
            recent = base if near == 0 else grouped_shift(base, ctx, near)
            past = grouped_shift(base, ctx, far)
            out[name] = (recent - past) if self.method == "log" else safe_divide(recent, past) - 1.0
        return out


@dataclass(frozen=True)
class MovingAverage:
    """Simple or exponential moving average over one or more windows."""

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    kind: str
    min_periods: int
    adjust: bool

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        _check_kind(self.kind)
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
            name: moving_average(
                values,
                ctx,
                window=window,
                kind=self.kind,
                min_periods=self.min_periods,
                adjust=self.adjust,
            )
            for window, name in zip(self.windows, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class PriceToMovingAverage:
    """Price relative to its moving average, ``P_t / MA_n(P)_t - 1``, per window."""

    input_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    kind: str
    min_periods: int
    adjust: bool

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        _check_kind(self.kind)
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
            ma = moving_average(
                values,
                ctx,
                window=window,
                kind=self.kind,
                min_periods=self.min_periods,
                adjust=self.adjust,
            )
            out[name] = safe_divide(values, ma) - 1.0
        return out


@dataclass(frozen=True)
class MovingAverageRatio:
    """Fast-to-slow moving-average ratio ``MA_f(P)_t / MA_s(P)_t - 1`` per pair."""

    input_column: str
    fast_windows: tuple[int, ...]
    slow_windows: tuple[int, ...]
    output_names: tuple[str, ...]
    kind: str
    min_periods: int
    adjust: bool

    def __post_init__(self) -> None:
        if len(self.fast_windows) != len(self.slow_windows):
            raise ValueError("fast_windows and slow_windows must have equal length.")
        check_named(len(self.fast_windows), self.output_names, "window pairs")
        _check_kind(self.kind)
        for fast, slow in zip(self.fast_windows, self.slow_windows, strict=True):
            if fast <= 0 or slow <= fast:
                raise ValueError(f"require slow > fast > 0; got fast={fast}, slow={slow}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        values = column_values(panel, self.input_column)
        out: dict[str, FloatArray] = {}
        for fast, slow, name in zip(
            self.fast_windows, self.slow_windows, self.output_names, strict=True
        ):
            ma_fast = moving_average(
                values,
                ctx,
                window=fast,
                kind=self.kind,
                min_periods=self.min_periods,
                adjust=self.adjust,
            )
            ma_slow = moving_average(
                values,
                ctx,
                window=slow,
                kind=self.kind,
                min_periods=self.min_periods,
                adjust=self.adjust,
            )
            out[name] = safe_divide(ma_fast, ma_slow) - 1.0
        return out


@dataclass(frozen=True)
class MACD:
    """Moving-average convergence/divergence line, signal, and histogram.

    Produces three columns (line, signal, histogram) in that order.

    Parameters
    ----------
    input_column:
        Price column.
    fast_span, slow_span, signal_span:
        EMA spans for the fast, slow, and signal lines (``slow > fast``).
    output_names:
        Exactly three names: line, signal, histogram.
    adjust:
        Exponential-smoothing convention for all three EMAs.
    min_periods:
        Minimum observations before an EMA emits a value.
    """

    input_column: str
    fast_span: int
    slow_span: int
    signal_span: int
    output_names: tuple[str, ...]
    adjust: bool
    min_periods: int

    def __post_init__(self) -> None:
        check_named(3, self.output_names, "MACD components (line, signal, histogram)")
        if not 0 < self.fast_span < self.slow_span:
            raise ValueError("require 0 < fast_span < slow_span.")
        if self.signal_span <= 0:
            raise ValueError("signal_span must be positive.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.input_column)
        fast = grouped_ewm(
            price,
            ctx,
            span=self.fast_span,
            alpha=None,
            adjust=self.adjust,
            min_periods=self.min_periods,
            how="mean",
        )
        slow = grouped_ewm(
            price,
            ctx,
            span=self.slow_span,
            alpha=None,
            adjust=self.adjust,
            min_periods=self.min_periods,
            how="mean",
        )
        line = fast - slow
        signal = grouped_ewm(
            line,
            ctx,
            span=self.signal_span,
            alpha=None,
            adjust=self.adjust,
            min_periods=self.min_periods,
            how="mean",
        )
        histogram = line - signal
        line_name, signal_name, hist_name = self.output_names
        return {line_name: line, signal_name: signal, hist_name: histogram}


@dataclass(frozen=True)
class DistanceFromRollingExtreme:
    """Distance of the price from a rolling extreme, ``P_t / extreme_n - 1``.

    ``kind="max"`` divides by the rolling maximum of ``extreme_column`` (giving a
    non-positive proximity-to-high, e.g. the 52-week-high feature); ``kind="min"``
    divides by the rolling minimum (a non-negative proximity-to-low).
    """

    price_column: str
    extreme_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    kind: str
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if self.kind not in ("max", "min"):
            raise ValueError(f"kind must be 'max' or 'min'; got {self.kind!r}.")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.price_column, self.extreme_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.price_column)
        reference = column_values(panel, self.extreme_column)
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            extreme = grouped_rolling(
                reference, ctx, window=window, min_periods=self.min_periods, how=self.kind
            )
            out[name] = safe_divide(price, extreme) - 1.0
        return out
