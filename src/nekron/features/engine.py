"""Vectorized grouped primitives over an entity-major ``(date, entity)`` panel.

These functions are the reusable computational core shared by every featurizer.
They operate on ``float64`` arrays aligned to the rows of a :class:`PanelContext`
and return arrays in the same order, so featurizers compose them without ever
touching pandas grouping machinery directly.

Two grouping axes are supported:

* **Per-entity, time-series** ops (shift, diff, cumulative, rolling window,
  exponential smoothing) group by ``ctx.entity_codes``. They assume each entity's
  rows are contiguous and date-ascending, which the pipeline guarantees.
* **Per-date, cross-sectional** ops (rank, z-score, winsorize, …) group by
  ``ctx.date_codes`` and are alignment-safe regardless of row contiguity.

All windowed reducers honor ``min_periods``; a window that is short or contains a
``NaN`` beyond the allowed count yields ``NaN`` rather than a partial value.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pandas as pd

from .base import FloatArray, IntArray, PanelContext
from .numeric import safe_divide

_ROLLING_HOWS = frozenset(
    {"mean", "std", "var", "sum", "min", "max", "median", "skew", "kurt", "quantile"}
)
_EWM_HOWS = frozenset({"mean", "std", "var"})


def _series(values: FloatArray) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Per-entity, time-series operations
# --------------------------------------------------------------------------- #
def grouped_shift(values: FloatArray, ctx: PanelContext, periods: int) -> FloatArray:
    """Shift each entity's series by ``periods`` rows.

    Positive ``periods`` pulls past values forward (a lag); negative ``periods``
    pulls future values back (a lead, used to build forecast targets). Rows with
    no in-entity neighbor at the requested offset are ``NaN``.
    """
    shifted = _series(values).groupby(ctx.entity_codes, sort=False).shift(periods)
    return np.asarray(shifted.to_numpy(), dtype=np.float64)


def grouped_diff(values: FloatArray, ctx: PanelContext, periods: int) -> FloatArray:
    """Difference of each entity's series against its value ``periods`` rows earlier."""
    return np.asarray(values, dtype=np.float64) - grouped_shift(values, ctx, periods)


def grouped_pct_change(values: FloatArray, ctx: PanelContext, periods: int) -> FloatArray:
    """Simple percentage change over ``periods`` rows within each entity.

    Computed as ``value / past_value - 1`` with a zero or non-finite base mapped to
    ``NaN``, so no infinite return is produced.
    """
    prev = grouped_shift(values, ctx, periods)
    return safe_divide(np.asarray(values, dtype=np.float64), prev) - 1.0


def grouped_cumsum(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Cumulative sum within each entity (``NaN`` values are skipped)."""
    out = _series(values).groupby(ctx.entity_codes, sort=False).cumsum()
    return np.asarray(out.to_numpy(), dtype=np.float64)


def grouped_cummax(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Cumulative maximum within each entity."""
    out = _series(values).groupby(ctx.entity_codes, sort=False).cummax()
    return np.asarray(out.to_numpy(), dtype=np.float64)


def grouped_cummin(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Cumulative minimum within each entity."""
    out = _series(values).groupby(ctx.entity_codes, sort=False).cummin()
    return np.asarray(out.to_numpy(), dtype=np.float64)


def _dispatch_window(roller: Any, how: str, ddof: int, quantile: float | None) -> Any:
    if how == "mean":
        return roller.mean()
    if how == "sum":
        return roller.sum()
    if how == "min":
        return roller.min()
    if how == "max":
        return roller.max()
    if how == "median":
        return roller.median()
    if how == "std":
        return roller.std(ddof=ddof)
    if how == "var":
        return roller.var(ddof=ddof)
    if how == "skew":
        return roller.skew()
    if how == "kurt":
        return roller.kurt()
    if how == "quantile":
        if quantile is None:
            raise ValueError("quantile reducer requires a quantile level.")
        return roller.quantile(quantile)
    raise ValueError(f"unknown rolling reducer {how!r}; valid are {sorted(_ROLLING_HOWS)}.")


def grouped_rolling(
    values: FloatArray,
    ctx: PanelContext,
    *,
    window: int,
    min_periods: int,
    how: str,
    ddof: int = 1,
    quantile: float | None = None,
) -> FloatArray:
    """Rolling window reduction within each entity.

    Parameters
    ----------
    window:
        Number of rows in the trailing window (must be positive).
    min_periods:
        Minimum non-``NaN`` observations required for a non-``NaN`` result. It is
        capped at ``window``, since a window cannot hold more observations than its
        length (this keeps a single shared ``min_periods`` valid across the several
        differently-sized windows some indicators use internally).
    how:
        Reducer name: one of ``mean``, ``std``, ``var``, ``sum``, ``min``,
        ``max``, ``median``, ``skew``, ``kurt``, ``quantile``.
    ddof:
        Delta degrees of freedom for ``std`` / ``var``.
    quantile:
        Quantile level in ``[0, 1]`` when ``how == "quantile"``.

    One flat whole-column rolling pass computes every window, which is materially
    faster than rolling per entity. Only the leading ``window - 1`` rows of each
    entity are wrong that way, because their flat window straddles the previous
    entity: with a full window those rows have no answer and are ``NaN``, and with
    a partial one their true window is exactly the entity's own prefix, so
    re-rolling that small slice repairs them. Either way the values are identical
    to the per-entity computation. ``quantile`` alone stays on the grouped path.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    if how not in _ROLLING_HOWS:
        raise ValueError(f"unknown rolling reducer {how!r}; valid are {sorted(_ROLLING_HOWS)}.")
    min_periods = min(min_periods, window)
    series = _series(values)
    if how == "quantile":
        roller = series.groupby(ctx.entity_codes, sort=False).rolling(
            window=window, min_periods=min_periods
        )
        out = np.asarray(_dispatch_window(roller, how, ddof, quantile).to_numpy(), dtype=np.float64)
    else:
        flat = _dispatch_window(
            series.rolling(window=window, min_periods=min_periods), how, ddof, None
        )
        out = np.asarray(flat.to_numpy(), dtype=np.float64).copy()
        head = ctx.within_entity < window - 1
        if min_periods == window:
            # A full window can never be satisfied from a partial prefix.
            out[head] = np.nan
        elif head.any():
            # Only the leading rows are wrong — their flat window straddles the
            # previous entity — and for those rows the true window *is* the
            # entity's prefix, so re-rolling just that slice is exact. It is also
            # a small fraction of the panel, where rolling the whole column the
            # grouped way costs 2-4.5x for the same answer.
            positions = np.flatnonzero(head)
            prefix = series.iloc[positions]
            prefix_roller = prefix.groupby(ctx.entity_codes[positions], sort=False).rolling(
                window=window, min_periods=min_periods
            )
            repaired = _dispatch_window(prefix_roller, how, ddof, None)
            out[positions] = np.asarray(repaired.to_numpy(), dtype=np.float64)
    if how == "var":
        out = np.asarray(np.where(np.isfinite(out), np.maximum(out, 0.0), out), dtype=np.float64)
    return out


def grouped_rolling_mad(values: FloatArray, ctx: PanelContext, *, window: int) -> FloatArray:
    """Rolling mean absolute deviation about the window mean (full windows only).

    Computes ``(1/window) * sum |x_i - mean(window)|`` for each trailing window that
    lies fully within one entity; shorter or boundary-straddling windows are
    ``NaN``. Memory stays ``O(n)`` by accumulating the absolute deviations one
    window-offset at a time over a strided view rather than materializing the
    ``(n, window)`` block.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        view = np.lib.stride_tricks.sliding_window_view(arr, window)
        mean = view.mean(axis=1)
        mad = np.zeros(mean.shape[0], dtype=np.float64)
        for offset in range(window):
            mad += np.abs(view[:, offset] - mean)
        mad /= window
        out[window - 1 :] = mad
    out[ctx.within_entity < window - 1] = np.nan
    return out


def grouped_rolling_rank(
    values: FloatArray,
    ctx: PanelContext,
    *,
    window: int,
    min_periods: int,
    pct: bool,
    method: str,
) -> FloatArray:
    """Rank of each value within its own trailing per-entity window.

    With ``pct=True`` the rank is scaled to ``(0, 1]``, giving a time-series
    percentile of the current observation against its recent history.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    min_periods = min(min_periods, window)
    ranked = (
        _series(values)
        .groupby(ctx.entity_codes, sort=False)
        .rolling(window=window, min_periods=min_periods)
        .rank(method=cast(Any, method), pct=pct)
    )
    return np.asarray(ranked.to_numpy(), dtype=np.float64)


def _window_drawdowns(arr: FloatArray, window: int) -> Iterator[FloatArray]:
    """Yield each window offset's causal drawdown from the window's running maximum.

    Both drawdown statistics sweep the same ``window`` offsets over a strided view,
    differing only in how they reduce the result — a maximum for drawdown, a sum of
    squares for the ulcer index. Sharing the sweep lets it run entirely in
    preallocated buffers: the obvious spelling allocates five full-length
    temporaries per offset, which at a 252-day window on a million rows is over a
    thousand eight-megabyte allocations for a single feature.

    The yielded array is reused on every iteration, so a consumer must reduce it
    before advancing. It may also modify it in place; the next offset overwrites it
    regardless.

    Validity follows :func:`~nekron.features.numeric.safe_divide` exactly: an entry
    is ``NaN`` unless both the value and its running maximum are finite and the
    maximum is non-zero. That matters at both ends — a ``-inf`` value has a finite
    running maximum, and a ``NaN`` one poisons the maximum from then on.
    """
    view = np.lib.stride_tricks.sliding_window_view(arr, window)
    width = view.shape[0]
    running_max = np.full(width, -np.inf, dtype=np.float64)
    drawdown = np.empty(width, dtype=np.float64)
    usable = np.empty(width, dtype=bool)
    scratch = np.empty(width, dtype=bool)
    for offset in range(window):
        column = view[:, offset]
        np.maximum(running_max, column, out=running_max)
        np.isfinite(column, out=usable)
        np.isfinite(running_max, out=scratch)
        np.logical_and(usable, scratch, out=usable)
        np.not_equal(running_max, 0.0, out=scratch)
        np.logical_and(usable, scratch, out=usable)
        drawdown.fill(np.nan)
        np.divide(column, running_max, out=drawdown, where=usable)
        np.subtract(1.0, drawdown, out=drawdown)
        yield drawdown


def grouped_rolling_max_drawdown(
    values: FloatArray, ctx: PanelContext, *, window: int
) -> FloatArray:
    """Maximum drawdown within each trailing window (full windows only).

    For each window that lies fully within one entity, computes the worst
    peak-to-trough decline ``max_i (1 - x_i / running_max_i)`` using a causal
    running maximum inside the window, so only ``window`` observations are needed.
    Memory stays ``O(n)`` by sweeping the window offsets over a strided view.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        max_drawdown = np.zeros(n - window + 1, dtype=np.float64)
        for drawdown in _window_drawdowns(arr, window):
            np.maximum(max_drawdown, drawdown, out=max_drawdown)
        out[window - 1 :] = max_drawdown
    out[ctx.within_entity < window - 1] = np.nan
    return out


def grouped_rolling_ulcer(values: FloatArray, ctx: PanelContext, *, window: int) -> FloatArray:
    """Ulcer Index within each trailing window (full windows only).

    Root-mean-square of the percentage drawdown from a causal running maximum
    anchored at each window's start, ``sqrt(mean_i (100 * (1 - x_i / M_i))^2)``.
    Needs only ``window`` observations; memory stays ``O(n)``.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        sum_squared = np.zeros(n - window + 1, dtype=np.float64)
        for drawdown in _window_drawdowns(arr, window):
            # The yielded buffer is scratch space, so squaring it in place costs
            # nothing; the next offset refills it.
            np.multiply(drawdown, 100.0, out=drawdown)
            np.multiply(drawdown, drawdown, out=drawdown)
            np.add(sum_squared, drawdown, out=sum_squared)
        out[window - 1 :] = np.sqrt(sum_squared / window)
    out[ctx.within_entity < window - 1] = np.nan
    return out


def grouped_ewm(
    values: FloatArray,
    ctx: PanelContext,
    *,
    span: int | None,
    alpha: float | None,
    adjust: bool,
    min_periods: int,
    how: str = "mean",
) -> FloatArray:
    """Exponentially weighted reduction within each entity.

    Exactly one of ``span`` or ``alpha`` sets the decay. ``adjust=False`` gives the
    recursive form ``y_t = (1 - a) y_{t-1} + a x_t`` (Wilder smoothing uses
    ``alpha = 1/n`` with ``adjust=False``); ``adjust=True`` gives the finite-window
    weighting. ``how`` is one of ``mean``, ``std``, ``var``.
    """
    if (span is None) == (alpha is None):
        raise ValueError("exactly one of span / alpha must be provided.")
    if how not in _EWM_HOWS:
        raise ValueError(f"unknown ewm reducer {how!r}; valid are {sorted(_EWM_HOWS)}.")
    grouped = _series(values).groupby(ctx.entity_codes, sort=False)
    if span is not None:
        ewm = grouped.ewm(span=span, adjust=adjust, min_periods=min_periods)
    else:
        ewm = grouped.ewm(alpha=alpha, adjust=adjust, min_periods=min_periods)
    if how == "mean":
        reduced = ewm.mean()
    elif how == "std":
        reduced = ewm.std()
    else:
        reduced = ewm.var()
    return np.asarray(reduced.to_numpy(), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Per-date, cross-sectional operations
# --------------------------------------------------------------------------- #
def _grouped_std(series: pd.Series, codes: IntArray, ddof: int) -> pd.Series:
    """NaN-skipping grouped standard deviation with arbitrary ``ddof``."""
    grouped = series.groupby(codes, sort=False)
    count = grouped.transform("count")
    mean = grouped.transform("mean")
    ssq = ((series - mean) ** 2).groupby(codes, sort=False).transform("sum")
    denom = count - ddof
    var = safe_divide(ssq.to_numpy(), denom.to_numpy())
    var = np.where(np.isfinite(var), np.maximum(var, 0.0), var)
    return pd.Series(np.sqrt(var), index=series.index)


def cross_sectional_rank(
    values: FloatArray,
    ctx: PanelContext,
    *,
    method: str,
    pct: bool,
    ascending: bool,
) -> FloatArray:
    """Rank each value against its cross-section (all entities sharing its date).

    ``NaN`` values keep ``NaN`` ranks. With ``pct=True`` ranks are scaled to
    ``(0, 1]``. ``method`` follows :meth:`pandas.core.groupby.GroupBy.rank`.
    """
    ranked = (
        _series(values)
        .groupby(ctx.date_codes, sort=False)
        .rank(method=method, pct=pct, ascending=ascending)
    )
    return np.asarray(ranked.to_numpy(), dtype=np.float64)


def cross_sectional_zscore(values: FloatArray, ctx: PanelContext, *, ddof: int) -> FloatArray:
    """Standardize each value within its date's cross-section to zero mean, unit std."""
    series = _series(values)
    mean = series.groupby(ctx.date_codes, sort=False).transform("mean")
    std = _grouped_std(series, ctx.date_codes, ddof)
    return safe_divide((series - mean).to_numpy(), std.to_numpy())


def cross_sectional_demean(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Subtract the per-date cross-sectional mean from each value."""
    series = _series(values)
    mean = series.groupby(ctx.date_codes, sort=False).transform("mean")
    return np.asarray((series - mean).to_numpy(), dtype=np.float64)


def cross_sectional_minmax(values: FloatArray, ctx: PanelContext) -> FloatArray:
    """Scale each value to ``[0, 1]`` across its date's cross-section."""
    series = _series(values)
    grouped = series.groupby(ctx.date_codes, sort=False)
    low = grouped.transform("min")
    high = grouped.transform("max")
    return safe_divide((series - low).to_numpy(), (high - low).to_numpy())


def cross_sectional_winsorize(
    values: FloatArray, ctx: PanelContext, *, lower: float, upper: float
) -> FloatArray:
    """Clip each value to its date's cross-sectional ``[lower, upper]`` quantiles."""
    series = _series(values)
    grouped = series.groupby(ctx.date_codes, sort=False)
    lo = grouped.quantile(lower).reindex(ctx.date_codes).to_numpy()
    hi = grouped.quantile(upper).reindex(ctx.date_codes).to_numpy()
    return np.asarray(np.clip(series.to_numpy(), lo, hi), dtype=np.float64)


def cross_sectional_bucket(values: FloatArray, ctx: PanelContext, *, n_buckets: int) -> FloatArray:
    """Assign each value to a per-date quantile bucket in ``1 .. n_buckets``.

    Bucketing is rank-based, so ties and uneven cross-section sizes are handled
    without empty-bin errors. ``NaN`` values stay ``NaN``.
    """
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive; got {n_buckets}.")
    pct = cross_sectional_rank(values, ctx, method="average", pct=True, ascending=True)
    buckets = np.ceil(pct * n_buckets)
    return np.asarray(np.clip(buckets, 1.0, float(n_buckets)), dtype=np.float64)
