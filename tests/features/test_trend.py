"""Tests for directional-movement and trend-strength featurizers."""

from __future__ import annotations

import numpy as np

from nekron.features import ADX, Aroon, FeaturePipeline, LinearTrendSlope, TrueStrengthIndex

from .conftest import random_ohlcv, single_entity_panel


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_adx_di_bounded() -> None:
    panel = random_ohlcv(80, ["A", "B"]).sort_index()
    out = _run(ADX("high", "low", "close", 14, ("adx", "pdi", "mdi"), min_periods=14), panel)
    for col in ("adx", "pdi", "mdi"):
        vals = out[col].to_numpy()
        valid = vals[~np.isnan(vals)]
        assert np.all((valid >= -1e-9) & (valid <= 100 + 1e-9)), col
    assert not np.isinf(out.to_numpy()).any()


def test_aroon_fresh_high_is_100() -> None:
    dates = [f"2020-{m:02d}-01" for m in range(1, 13)]
    highs = [float(i) for i in range(12)]  # strictly increasing -> new high each bar
    lows = [float(i) for i in range(12)]
    panel = single_entity_panel(dates, {"high": highs, "low": lows})
    out = _run(Aroon("high", "low", 5, ("up", "down", "osc")), panel)
    up = out.xs("A", level="entity")["up"].to_numpy()
    assert np.isclose(up[-1], 100.0)  # newest bar is the high


def test_aroon_recent_low_scores_down_100() -> None:
    dates = [f"2020-{m:02d}-01" for m in range(1, 13)]
    highs = [float(i) for i in range(12)]
    lows = [float(-i) for i in range(12)]  # strictly decreasing -> new low each bar
    panel = single_entity_panel(dates, {"high": highs, "low": lows})
    out = _run(Aroon("high", "low", 5, ("up", "down", "osc")), panel)
    down = out.xs("A", level="entity")["down"].to_numpy()
    assert np.isclose(down[-1], 100.0)


def test_linear_trend_slope_recovers_known_slope() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 13)]
    # perfectly linear price with slope 2 per bar
    panel = single_entity_panel(dates, {"close": [3.0 + 2.0 * i for i in range(12)]})
    out = _run(LinearTrendSlope("close", (5,), ("slope",), use_log=False), panel)[
        "slope"
    ].to_numpy()
    valid = out[~np.isnan(out)]
    assert np.allclose(valid, 2.0)


def test_tsi_bounded_and_finite() -> None:
    panel = random_ohlcv(80, ["A"]).sort_index()
    out = _run(TrueStrengthIndex("close", 25, 13, "tsi", adjust=False, min_periods=25), panel)
    vals = out["tsi"].to_numpy()
    valid = vals[~np.isnan(vals)]
    assert np.all((valid >= -100 - 1e-6) & (valid <= 100 + 1e-6))
    assert not np.isinf(vals).any()
