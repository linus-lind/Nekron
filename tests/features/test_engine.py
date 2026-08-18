"""Tests for the grouped computational engine primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nekron.features.engine import (
    cross_sectional_bucket,
    cross_sectional_rank,
    cross_sectional_winsorize,
    cross_sectional_zscore,
    grouped_cumsum,
    grouped_diff,
    grouped_ewm,
    grouped_pct_change,
    grouped_rolling,
    grouped_rolling_mad,
    grouped_rolling_max_drawdown,
    grouped_shift,
)

from .conftest import em_context, make_panel, random_ohlcv


def test_grouped_shift_lag_and_lead_no_bleed() -> None:
    panel = make_panel(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        ["A", "B"],
        {"x": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]},
    )
    em, ctx = em_context(panel)
    x = em["x"].to_numpy()
    lag = grouped_shift(x, ctx, 1)
    lead = grouped_shift(x, ctx, -1)
    # em order is A: [1,2,3], B: [10,20,30]
    assert np.isnan(lag[0]) and lag[1] == 1.0 and lag[2] == 2.0
    assert np.isnan(lag[3]) and lag[4] == 10.0  # no bleed from A into B
    assert lead[0] == 2.0 and np.isnan(lead[2])


def test_grouped_diff_and_pct_change() -> None:
    panel = make_panel(["2020-01-01", "2020-01-02"], ["A"], {"x": [4.0, 6.0]})
    em, ctx = em_context(panel)
    x = em["x"].to_numpy()
    assert grouped_diff(x, ctx, 1)[1] == 2.0
    assert np.isclose(grouped_pct_change(x, ctx, 1)[1], 0.5)


def test_grouped_pct_change_zero_base_is_nan() -> None:
    panel = make_panel(["2020-01-01", "2020-01-02"], ["A"], {"x": [0.0, 6.0]})
    em, ctx = em_context(panel)
    out = grouped_pct_change(em["x"].to_numpy(), ctx, 1)
    assert np.isnan(out[1]) and not np.isinf(out).any()


def test_grouped_cumsum_resets_per_entity() -> None:
    panel = make_panel(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        ["A", "B"],
        {"x": [1.0, 5.0, 1.0, 5.0, 1.0, 5.0]},
    )
    em, ctx = em_context(panel)
    out = grouped_cumsum(em["x"].to_numpy(), ctx)
    # A cumulative [1,2,3], B cumulative [5,10,15]
    assert list(out) == [1.0, 2.0, 3.0, 5.0, 10.0, 15.0]


def test_grouped_rolling_fast_path_equals_grouped_path() -> None:
    panel = random_ohlcv(50, ["A", "B", "C"])
    em, ctx = em_context(panel)
    values = em["close"].to_numpy()
    for how in ("mean", "std", "sum", "min", "max"):
        fast = grouped_rolling(values, ctx, window=7, min_periods=7, how=how)
        reference = (
            pd.Series(values).groupby(ctx.entity_codes, sort=False).rolling(7, min_periods=7)
        )
        ref = (reference.std() if how == "std" else getattr(reference, how)()).to_numpy()
        assert np.allclose(fast, ref, equal_nan=True), how


def test_grouped_rolling_partial_window_min_periods() -> None:
    panel = make_panel(["2020-01-01", "2020-01-02", "2020-01-03"], ["A"], {"x": [1.0, 2.0, 3.0]})
    em, ctx = em_context(panel)
    out = grouped_rolling(em["x"].to_numpy(), ctx, window=3, min_periods=1, how="mean")
    assert np.isclose(out[0], 1.0) and np.isclose(out[1], 1.5) and np.isclose(out[2], 2.0)


def test_grouped_rolling_mad_matches_bruteforce() -> None:
    panel = random_ohlcv(30, ["A", "B"])
    em, ctx = em_context(panel)
    values = em["close"].to_numpy()
    out = grouped_rolling_mad(values, ctx, window=5)
    ref = (
        pd.Series(values)
        .groupby(ctx.entity_codes, sort=False)
        .rolling(5)
        .apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
        .to_numpy()
    )
    assert np.allclose(out, ref, equal_nan=True)


def test_grouped_rolling_max_drawdown_matches_bruteforce() -> None:
    panel = random_ohlcv(40, ["A", "B"])
    em, ctx = em_context(panel)
    values = em["close"].to_numpy()
    out = grouped_rolling_max_drawdown(values, ctx, window=10)

    def mdd(window: np.ndarray) -> float:
        running_max = np.maximum.accumulate(window)
        return float(np.max(1.0 - window / running_max))

    ref = (
        pd.Series(values)
        .groupby(ctx.entity_codes, sort=False)
        .rolling(10)
        .apply(mdd, raw=True)
        .to_numpy()
    )
    assert np.allclose(out, ref, equal_nan=True)
    assert not np.isinf(out).any()


def test_grouped_ewm_wilder_recursion() -> None:
    panel = make_panel(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"], ["A"], {"x": [1.0, 2.0, 3.0, 4.0]}
    )
    em, ctx = em_context(panel)
    out = grouped_ewm(
        em["x"].to_numpy(), ctx, span=None, alpha=0.5, adjust=False, min_periods=1, how="mean"
    )
    # y0=1; y1=.5*2+.5*1=1.5; y2=.5*3+.5*1.5=2.25; y3=.5*4+.5*2.25=3.125
    assert np.allclose(out, [1.0, 1.5, 2.25, 3.125])


def test_cross_sectional_rank_and_signed_bounds() -> None:
    panel = make_panel(["2020-01-01"], ["A", "B", "C", "D"], {"x": [10.0, 20.0, 30.0, 40.0]})
    em, ctx = em_context(panel)
    pct = cross_sectional_rank(em["x"].to_numpy(), ctx, method="average", pct=True, ascending=True)
    assert np.allclose(np.sort(pct), [0.25, 0.5, 0.75, 1.0])


def test_cross_sectional_zscore_zero_mean() -> None:
    panel = make_panel(["2020-01-01"], ["A", "B", "C"], {"x": [1.0, 2.0, 3.0]})
    em, ctx = em_context(panel)
    z = cross_sectional_zscore(em["x"].to_numpy(), ctx, ddof=1)
    assert np.isclose(np.nanmean(z), 0.0, atol=1e-12)


def test_cross_sectional_winsorize_clips() -> None:
    panel = make_panel(
        ["2020-01-01"], [str(i) for i in range(11)], {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0]}
    )
    em, ctx = em_context(panel)
    out = cross_sectional_winsorize(em["x"].to_numpy(), ctx, lower=0.0, upper=0.9)
    assert out.max() < 1000.0


def test_cross_sectional_bucket_range() -> None:
    panel = make_panel(
        ["2020-01-01"], [str(i) for i in range(10)], {"x": list(map(float, range(10)))}
    )
    em, ctx = em_context(panel)
    out = cross_sectional_bucket(em["x"].to_numpy(), ctx, n_buckets=5)
    assert set(np.unique(out)) <= {1.0, 2.0, 3.0, 4.0, 5.0}
