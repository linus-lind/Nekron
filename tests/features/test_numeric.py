"""Tests for the robust numeric primitives."""

from __future__ import annotations

import numpy as np

from nekron.features.numeric import (
    clip_quantiles,
    expm1_clip,
    log1p_ratio,
    safe_divide,
    safe_log,
)


def test_safe_divide_maps_zero_and_nonfinite_to_nan() -> None:
    num = np.array([1.0, 1.0, 1.0, np.inf, np.nan])
    den = np.array([2.0, 0.0, np.nan, 1.0, 1.0])
    out = safe_divide(num, den)
    assert out[0] == 0.5
    assert np.isnan(out[1])  # divide by zero -> NaN, not inf
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert np.isnan(out[4])
    assert not np.isinf(out).any()


def test_safe_log_guards_nonpositive() -> None:
    out = safe_log(np.array([1.0, np.e, 0.0, -1.0, np.nan]))
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], 1.0)
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])


def test_log1p_ratio_guards_total_loss() -> None:
    out = log1p_ratio(np.array([0.0, 1.0, -1.0, -2.0]))
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], np.log(2.0))
    assert np.isnan(out[2])  # 1 + (-1) = 0 -> undefined
    assert np.isnan(out[3])


def test_expm1_clip_no_overflow() -> None:
    out = expm1_clip(np.array([0.0, 1000.0, -1000.0, np.nan]))
    assert np.isclose(out[0], 0.0)
    assert np.isfinite(out[1]) and out[1] > 0  # saturates, no inf
    assert np.isclose(out[2], -1.0, atol=1e-6)
    assert np.isnan(out[3])


def test_expm1_log1p_roundtrip() -> None:
    returns = np.array([0.01, -0.02, 0.005, 0.03])
    compounded = expm1_clip(np.array([np.sum(log1p_ratio(returns))]))
    expected = np.prod(1 + returns) - 1
    assert np.isclose(compounded[0], expected)


def test_clip_quantiles_ignores_nan() -> None:
    values = np.array([np.nan, 1.0, 2.0, 3.0, 100.0])
    out = clip_quantiles(values, 0.0, 0.75)
    assert np.isnan(out[0])
    assert np.nanmax(out) <= np.nanquantile(values, 0.75) + 1e-9
