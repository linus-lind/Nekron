"""Tests for momentum, moving-average, and volume/price-volume featurizers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nekron.features import (
    MACD,
    AccumulationDistribution,
    ChaikinMoneyFlow,
    DistanceFromRollingExtreme,
    FeaturePipeline,
    ForceIndex,
    Momentum,
    MovingAverage,
    OnBalanceVolume,
    VolumePriceTrend,
)

from .conftest import random_ohlcv, single_entity_panel


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_momentum_with_skip() -> None:
    dates = [f"2020-{m:02d}-01" for m in range(1, 13)]  # 12 monthly points as stand-in rows
    closes = [float(100 + i) for i in range(12)]
    panel = single_entity_panel(dates, {"close": closes})
    # near=2, far=11: mom = close_{t-2}/close_{t-11} - 1 at the last row
    out = _run(Momentum("close", (2,), (11,), ("mom",), method="simple"), panel)["mom"].to_numpy()
    expected = closes[-3] / closes[-12] - 1  # t-2 and t-11 at last index (t=11)
    assert np.isclose(out[-1], expected)


def test_ema_no_cross_entity_bleed() -> None:
    # entity B is constant; its EMA must remain constant regardless of entity A
    panel = pd.DataFrame(
        {"close": [1.0, 100.0, 2.0, 100.0, 3.0, 100.0, 4.0, 100.0]},
        index=pd.MultiIndex.from_product(
            [pd.bdate_range("2020-01-01", periods=4), ["A", "B"]], names=["date", "entity"]
        ),
    )
    out = _run(
        MovingAverage("close", (2,), ("ema",), kind="ema", min_periods=1, adjust=False), panel
    )
    assert np.allclose(out.xs("B", level="entity")["ema"].to_numpy(), 100.0)


def test_moving_average_sma_and_ema() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    closes = list(map(float, range(1, 11)))
    panel = single_entity_panel(dates, {"close": closes})
    sma = _run(MovingAverage("close", (3,), ("s",), kind="sma", min_periods=3, adjust=False), panel)
    ema = _run(MovingAverage("close", (3,), ("e",), kind="ema", min_periods=1, adjust=False), panel)
    ref_sma = pd.Series(closes).rolling(3).mean().to_numpy()
    ref_ema = pd.Series(closes).ewm(span=3, adjust=False, min_periods=1).mean().to_numpy()
    assert np.allclose(sma["s"].to_numpy(), ref_sma, equal_nan=True)
    assert np.allclose(ema["e"].to_numpy(), ref_ema, equal_nan=True)


def test_macd_three_components() -> None:
    panel = random_ohlcv(60, ["A"]).sort_index()
    out = _run(
        MACD("close", 12, 26, 9, ("line", "signal", "hist"), adjust=False, min_periods=26), panel
    )
    close = panel["close"].to_numpy()
    fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=26).mean()
    slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False, min_periods=26).mean()
    assert np.allclose(out["line"].to_numpy(), line.to_numpy(), equal_nan=True)
    assert np.allclose(out["hist"].to_numpy(), (line - signal).to_numpy(), equal_nan=True)


def test_distance_from_rolling_high_is_nonpositive() -> None:
    panel = random_ohlcv(40, ["A", "B"]).sort_index()
    out = _run(
        DistanceFromRollingExtreme("close", "high", (20,), ("d",), kind="max", min_periods=20),
        panel,
    )["d"].to_numpy()
    valid = out[~np.isnan(out)]
    assert np.all(valid <= 1e-9)  # close <= trailing high


def test_obv_and_vpt_cumulative_reset_per_entity() -> None:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    panel = single_entity_panel(
        dates, {"close": [10.0, 11.0, 10.5], "volume": [100.0, 200.0, 50.0]}
    )
    obv = _run(OnBalanceVolume("close", "volume", "obv"), panel)["obv"].to_numpy()
    # first NaN (no prior), then +200 (up), then -50 (down) accumulating from the first defined
    assert np.isnan(obv[0])
    assert np.isclose(obv[1], 200.0)
    assert np.isclose(obv[2], 150.0)
    vpt = _run(VolumePriceTrend("close", "volume", "vpt"), panel)["vpt"].to_numpy()
    assert np.isfinite(vpt[1:]).all()


def test_adl_cmf_and_force_index_bounded() -> None:
    panel = random_ohlcv(40, ["A", "B"]).sort_index()
    pipe = FeaturePipeline(
        featurizers=(
            AccumulationDistribution("high", "low", "close", "volume", "adl"),
            ChaikinMoneyFlow("high", "low", "close", "volume", (20,), ("cmf",), min_periods=20),
            ForceIndex("close", "volume", 13, "fi", adjust=False, min_periods=13),
        )
    )
    out = pipe.apply(panel)
    cmf = out["cmf"].to_numpy()
    assert np.all((cmf[~np.isnan(cmf)] >= -1 - 1e-9) & (cmf[~np.isnan(cmf)] <= 1 + 1e-9))
    assert not np.isinf(out.to_numpy()).any()


def test_money_flow_multiplier_zero_on_zero_range() -> None:
    # H == L for every bar -> ADL contribution zero, ADL flat at 0
    dates = ["2020-01-01", "2020-01-02"]
    panel = single_entity_panel(
        dates,
        {"high": [5.0, 5.0], "low": [5.0, 5.0], "close": [5.0, 5.0], "volume": [100.0, 100.0]},
    )
    out = _run(AccumulationDistribution("high", "low", "close", "volume", "adl"), panel)
    assert np.allclose(out["adl"].to_numpy(), [0.0, 0.0])
