"""Tests for liquidity featurizers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nekron.features import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    DollarVolume,
    FeaturePipeline,
    KyleLambda,
    MarketCap,
    RollSpread,
    Turnover,
    ZeroReturnFraction,
)

from .conftest import random_ohlcv, single_entity_panel

_DATES = [f"2020-01-{i:02d}" for i in range(1, 8)]


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_amihud_skips_zero_volume_days() -> None:
    panel = single_entity_panel(
        _DATES,
        {
            "ret": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01],
            "dv": [1e6, 0.0, 2e6, 1e6, 5e5, 1e6, 1e6],
        },
    )
    out = _run(AmihudIlliquidity("ret", "dv", (7,), ("amh",), min_periods=1, scale=1e6), panel)
    vals = out["amh"].to_numpy()
    # last row averages |ret|/dv over the 6 days with dv>0 (the zero-dv day excluded)
    ret = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01])
    dv = np.array([1e6, 0.0, 2e6, 1e6, 5e5, 1e6, 1e6])
    daily = np.abs(ret) / np.where(dv > 0, dv, np.nan)
    expected = 1e6 * np.nanmean(daily)
    assert np.isclose(vals[-1], expected)
    assert not np.isinf(vals).any()


def test_turnover_and_dollar_volume_and_market_cap() -> None:
    panel = single_entity_panel(
        ["2020-01-01", "2020-01-02"],
        {"close": [10.0, 20.0], "volume": [100.0, 0.0], "shares": [1000.0, 0.0]},
    )
    turn = _run(Turnover("volume", "shares", "to"), panel)["to"].to_numpy()
    assert np.isclose(turn[0], 0.1)
    assert np.isnan(turn[1])  # zero shares -> NaN, not inf
    dv = _run(DollarVolume("close", "volume", "dv", log=False), panel)["dv"].to_numpy()
    assert np.isclose(dv[0], 1000.0)
    cap = _run(MarketCap("close", "shares", "mc", log=True), panel)["mc"].to_numpy()
    assert np.isclose(cap[0], np.log(10.0 * 1000.0))


def test_zero_return_fraction() -> None:
    panel = single_entity_panel(_DATES, {"ret": [0.0, 0.0, 0.01, -0.01, 0.0, 0.02, 0.0]})
    out = _run(ZeroReturnFraction("ret", (7,), ("zr",), min_periods=1, tolerance=1e-8), panel)
    # 4 of 7 returns are zero
    assert np.isclose(out["zr"].to_numpy()[-1], 4 / 7)


def test_roll_and_corwin_and_kyle_run_without_inf() -> None:
    panel = random_ohlcv(40, ["A", "B"]).sort_index()
    pipe = FeaturePipeline(
        featurizers=(
            RollSpread("close", (21,), ("roll",), min_periods=15, normalize=True),
            CorwinSchultzSpread("high", "low", "close", (21,), ("cs",), min_periods=15),
            KyleLambda("ret", "dollar_volume", (21,), ("kyle",), min_periods=15),
        )
    )
    out = pipe.apply(panel)
    assert not np.isinf(out.to_numpy()).any()
    # Corwin-Schultz is floored at zero
    cs = out["cs"].to_numpy()
    assert np.all(cs[~np.isnan(cs)] >= 0.0)


def test_kyle_lambda_stable_with_large_near_constant_flow() -> None:
    # signed dollar volume ~1e10 with only 0.1% variation is the catastrophic-
    # cancellation case for the uncentered two-pass variance; the featurizer must
    # still match a window-centered OLS reference and stay finite.
    rng = np.random.default_rng(9)
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    ret = np.abs(rng.normal(0, 0.02, n))  # all positive -> sign(r) constant
    dollar_volume = 1e10 * (1.0 + rng.normal(0, 0.001, n))
    panel = pd.DataFrame(
        {"ret": ret, "dv": dollar_volume},
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "entity"]),
    )
    out = _run(KyleLambda("ret", "dv", (20,), ("kyle",), min_periods=20), panel)["kyle"].to_numpy()
    flow = np.sign(ret) * dollar_volume
    ref = np.full(n, np.nan)
    for i in range(19, n):
        rw, qw = ret[i - 19 : i + 1], flow[i - 19 : i + 1]
        qc = qw - qw.mean()
        ref[i] = np.sum(qc * (rw - rw.mean())) / np.sum(qc * qc)
    valid = ~np.isnan(out)
    assert np.isfinite(out[valid]).all()
    assert np.allclose(out[valid], ref[valid], rtol=1e-4)


def test_roll_spread_nan_when_positive_autocovariance() -> None:
    # strongly trending prices -> positive serial covariance of changes -> NaN spread
    dates = [f"2020-01-{i:02d}" for i in range(1, 25)]
    panel = single_entity_panel(dates, {"close": [100.0 + 2.0 * i for i in range(len(dates))]})
    out = _run(RollSpread("close", (20,), ("roll",), min_periods=15, normalize=False), panel)
    assert np.isnan(out["roll"].to_numpy()[-1])
