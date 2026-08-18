"""Return-based features: trailing, forward (target), and compounded returns.

Simple and log returns are kept distinct throughout. Simple returns compound
multiplicatively and are the correct inputs to mean-variance portfolio math; log
returns add across time and are the natural target for many forecasting models.
Every featurizer here takes an explicit ``method`` so the choice is never
implicit, and forward returns are a first-class, clearly-named target to keep
look-ahead alignment unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values
from .engine import grouped_pct_change, grouped_rolling, grouped_shift
from .numeric import expm1_clip, log1p_ratio, safe_divide, safe_log

_METHODS = ("simple", "log")


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}.")


@dataclass(frozen=True)
class SimpleReturns:
    """Trailing simple returns ``P_t / P_{t-h} - 1`` over one or more horizons.

    Parameters
    ----------
    input_column:
        Price (or level) column to difference.
    horizons:
        Trailing horizons in rows; one output column per horizon.
    output_names:
        Names of the produced columns, aligned to ``horizons``.
    """

    input_column: str
    horizons: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.horizons), self.output_names, "horizons")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive; got {self.horizons}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.input_column)
        return {
            name: grouped_pct_change(price, ctx, horizon)
            for horizon, name in zip(self.horizons, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class LogReturns:
    """Trailing log returns ``ln(P_t) - ln(P_{t-h})`` over one or more horizons."""

    input_column: str
    horizons: tuple[int, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.horizons), self.output_names, "horizons")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive; got {self.horizons}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_price = safe_log(column_values(panel, self.input_column))
        return {
            name: log_price - grouped_shift(log_price, ctx, horizon)
            for horizon, name in zip(self.horizons, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class ForwardReturns:
    """Forward returns ``P_{t+h} / P_t - 1`` (or log form): a forecast target.

    Produced by leading the price by each horizon, so the last ``h`` rows of every
    entity are ``NaN``. These are targets and must never be used as contemporaneous
    features.

    Parameters
    ----------
    input_column:
        Price column.
    horizons:
        Forward horizons in rows; one output column per horizon.
    output_names:
        Names of the produced columns, aligned to ``horizons``.
    method:
        ``"simple"`` for ``P_{t+h}/P_t - 1`` or ``"log"`` for ``ln(P_{t+h}/P_t)``.
    """

    input_column: str
    horizons: tuple[int, ...]
    output_names: tuple[str, ...]
    method: str

    def __post_init__(self) -> None:
        check_named(len(self.horizons), self.output_names, "horizons")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive; got {self.horizons}.")
        _check_method(self.method)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        price = column_values(panel, self.input_column)
        log_price = safe_log(price) if self.method == "log" else price
        out: dict[str, FloatArray] = {}
        for horizon, name in zip(self.horizons, self.output_names, strict=True):
            future = grouped_shift(log_price if self.method == "log" else price, ctx, -horizon)
            if self.method == "log":
                out[name] = future - log_price
            else:
                out[name] = safe_divide(future, price) - 1.0
        return out


@dataclass(frozen=True)
class CompoundReturns:
    """Compound a per-period simple-return column over trailing windows.

    Useful when the panel carries a periodic return series (e.g. a vendor's daily
    return) rather than a price. Simple compounding is done in log space and
    exponentiated once, so long windows neither overflow nor drift.

    Parameters
    ----------
    return_column:
        Per-period simple returns.
    windows:
        Trailing windows in rows; one output column per window.
    output_names:
        Names of the produced columns, aligned to ``windows``.
    method:
        ``"simple"`` for the compounded simple return or ``"log"`` for the summed
        log return over the window.
    min_periods:
        Minimum non-``NaN`` returns required within a window.
    """

    return_column: str
    windows: tuple[int, ...]
    output_names: tuple[str, ...]
    method: str
    min_periods: int

    def __post_init__(self) -> None:
        check_named(len(self.windows), self.output_names, "windows")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive; got {self.windows}.")
        _check_method(self.method)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.return_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        log_returns = log1p_ratio(column_values(panel, self.return_column))
        out: dict[str, FloatArray] = {}
        for window, name in zip(self.windows, self.output_names, strict=True):
            summed = grouped_rolling(
                log_returns, ctx, window=window, min_periods=self.min_periods, how="sum"
            )
            out[name] = summed if self.method == "log" else expm1_clip(summed)
        return out


@dataclass(frozen=True)
class IntradayReturn:
    """Close-to-open intraday return ``C_t / O_t - 1`` (or log form)."""

    open_column: str
    close_column: str
    output_name: str
    method: str

    def __post_init__(self) -> None:
        _check_method(self.method)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.open_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        open_ = column_values(panel, self.open_column)
        close = column_values(panel, self.close_column)
        if self.method == "log":
            value = safe_log(close) - safe_log(open_)
        else:
            value = safe_divide(close, open_) - 1.0
        return {self.output_name: value}


@dataclass(frozen=True)
class OvernightReturn:
    """Overnight return ``O_t / C_{t-1} - 1`` (or log form).

    The prior close is taken within each entity, so the first row of every entity
    is ``NaN``.
    """

    open_column: str
    close_column: str
    output_name: str
    method: str

    def __post_init__(self) -> None:
        _check_method(self.method)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.open_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        open_ = column_values(panel, self.open_column)
        prev_close = grouped_shift(column_values(panel, self.close_column), ctx, 1)
        if self.method == "log":
            value = safe_log(open_) - safe_log(prev_close)
        else:
            value = safe_divide(open_, prev_close) - 1.0
        return {self.output_name: value}
