"""Tests for oscillator featurizers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nekron.features import (
    CCI,
    RSI,
    AwesomeOscillator,
    FeaturePipeline,
    Stochastic,
    UltimateOscillator,
    WilliamsR,
)

from .conftest import single_entity_panel


def _ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "entity"])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_rsi_all_up_is_100() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 21)]
    panel = single_entity_panel(dates, {"close": [100.0 + i for i in range(20)]})
    out = _run(RSI("close", (5,), ("rsi",), min_periods=5), panel)["rsi"].to_numpy()
    assert np.isclose(np.nanmax(out), 100.0)
    valid = out[~np.isnan(out)]
    assert np.all((valid >= 0.0) & (valid <= 100.0))


def test_rsi_matches_wilder_reference() -> None:
    panel = _ohlc(40, 11)
    out = _run(RSI("close", (14,), ("rsi",), min_periods=14), panel)["rsi"].to_numpy()
    close = panel["close"].to_numpy()
    delta = np.concatenate([[np.nan], np.diff(close)])
    gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
    loss = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0))
    ag = pd.Series(gain).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    al = pd.Series(loss).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    ref = 100 * ag / (ag + al)
    assert np.allclose(out, ref, equal_nan=True)


def test_stochastic_and_williams_bounds() -> None:
    panel = _ohlc(30, 12)
    stoch = _run(Stochastic("high", "low", "close", 14, 3, 3, "k", "d", min_periods=14), panel)
    wr = _run(WilliamsR("high", "low", "close", 14, "wr", min_periods=14), panel)["wr"].to_numpy()
    k = stoch["k"].to_numpy()
    assert np.all((k[~np.isnan(k)] >= -1e-9) & (k[~np.isnan(k)] <= 100 + 1e-9))
    assert np.all((wr[~np.isnan(wr)] >= -100 - 1e-9) & (wr[~np.isnan(wr)] <= 1e-9))


def test_cci_matches_formula() -> None:
    panel = _ohlc(30, 13)
    out = _run(CCI("high", "low", "close", (20,), ("cci",), constant=0.015), panel)[
        "cci"
    ].to_numpy()
    tp = (panel["high"] + panel["low"] + panel["close"]).to_numpy() / 3.0
    mean = pd.Series(tp).rolling(20).mean().to_numpy()
    mad = (
        pd.Series(tp)
        .rolling(20)
        .apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
        .to_numpy()
    )
    ref = (tp - mean) / (0.015 * mad)
    assert np.allclose(out, ref, equal_nan=True)


def test_flat_window_yields_nan_not_neutral() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 21)]
    flat = [50.0] * 20
    panel = single_entity_panel(dates, {"high": flat, "low": flat, "close": flat})
    stoch = _run(Stochastic("high", "low", "close", 14, 3, 3, "k", "d", min_periods=14), panel)
    # HH == LL everywhere -> %K undefined -> NaN (not a fabricated 50)
    assert np.isnan(stoch["k"].to_numpy()).all()


def test_ultimate_and_awesome_run() -> None:
    panel = _ohlc(40, 14)
    uo = _run(
        UltimateOscillator(
            "high", "low", "close", (7, 14, 28), (4.0, 2.0, 1.0), "uo", min_periods=28
        ),
        panel,
    )["uo"].to_numpy()
    ao = _run(AwesomeOscillator("high", "low", 5, 34, "ao", min_periods=34), panel)["ao"].to_numpy()
    valid_uo = uo[~np.isnan(uo)]
    assert np.all((valid_uo >= 0.0) & (valid_uo <= 100.0))
    assert np.isfinite(ao[~np.isnan(ao)]).all()
