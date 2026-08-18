"""Overflow- and domain-safe numeric primitives for feature computation.

Every helper works on ``float64`` arrays and maps undefined results (division by
zero, logarithm of a non-positive number, non-finite inputs) to ``NaN`` rather
than propagating ``inf`` or raising, so a single bad observation never poisons a
whole column. NaN is the library's canonical "value undefined here" marker.
"""

from __future__ import annotations

import numpy as np

from .base import FloatArray


def safe_divide(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Elementwise ``numerator / denominator``, returning ``NaN`` where undefined.

    A result is ``NaN`` wherever the denominator is zero or either operand is not
    finite; no ``inf`` is ever produced.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=valid)
    # ``np.full(nan)`` plus ``where=`` already leaves every invalid entry NaN;
    # rewriting them costs a second full-length mask and a masked store.
    return out


def safe_log(values: FloatArray) -> FloatArray:
    """Natural logarithm, returning ``NaN`` for non-positive or non-finite inputs."""
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    np.log(arr, out=out, where=valid)
    # ``np.full(nan)`` plus ``where=`` already leaves every invalid entry NaN;
    # rewriting them costs a second full-length mask and a masked store.
    return out


def log1p_ratio(values: FloatArray) -> FloatArray:
    """``log(1 + values)``, returning ``NaN`` where ``1 + values <= 0`` or non-finite.

    Used to convert a simple return ``r`` into a log return ``log(1 + r)`` while
    guarding the ``r <= -1`` (total loss) boundary.
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > -1.0)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    np.log1p(arr, out=out, where=valid)
    # ``np.full(nan)`` plus ``where=`` already leaves every invalid entry NaN;
    # rewriting them costs a second full-length mask and a masked store.
    return out


def expm1_clip(values: FloatArray) -> FloatArray:
    """``exp(values) - 1`` with the exponent clipped to avoid overflow to ``inf``.

    Converts an accumulated log return back to a simple return. The exponent is
    bounded so extreme cumulative sums saturate to a large finite magnitude
    instead of overflowing.
    """
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, -700.0, 700.0)
    out = np.expm1(clipped)
    out[~np.isfinite(arr)] = np.nan
    return out


def clip_quantiles(values: FloatArray, lower: float, upper: float) -> FloatArray:
    """Clip ``values`` to their ``[lower, upper]`` empirical quantiles, ignoring NaN.

    Parameters
    ----------
    values:
        Data to winsorize.
    lower, upper:
        Quantile levels in ``[0, 1]`` with ``lower <= upper``.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = np.quantile(finite, [lower, upper])
    return np.clip(arr, lo, hi)


def rolling_reduce_is_variance(how: str) -> bool:
    """True if ``how`` names a variance-like reducer whose result must be floored at 0."""
    return how in {"var", "std"}
