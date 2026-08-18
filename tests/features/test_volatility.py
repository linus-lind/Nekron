"""Tests for volatility featurizers, checked against direct formula references."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nekron.features import (
    AverageTrueRange,
    BollingerBands,
    FeaturePipeline,
    GarmanKlassVolatility,
    MaxDrawdown,
    ParkinsonVolatility,
    RealizedVolatility,
    UlcerIndex,
    YangZhangVolatility,
)

_LN2 = math.log(2.0)


def _ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "entity"])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _run(feat, panel):
    return FeaturePipeline(featurizers=(feat,)).apply(panel)


def test_realized_volatility_matches_rolling_std() -> None:
    panel = _ohlc(30, 1)
    ret = np.log(panel["close"]).diff().rename("ret")
    p = panel.assign(ret=ret)
    out = _run(
        RealizedVolatility(
            "ret", (10,), ("rv",), min_periods=10, ddof=1, annualization_factor=math.sqrt(252)
        ),
        p,
    )["rv"]
    ref = p["ret"].groupby(level="entity").rolling(10).std(ddof=1).droplevel(0) * math.sqrt(252)
    assert np.allclose(out.to_numpy(), ref.to_numpy(), equal_nan=True)


def test_parkinson_matches_formula() -> None:
    panel = _ohlc(20, 2)
    out = _run(
        ParkinsonVolatility(
            "high", "low", (10,), ("pk",), min_periods=10, annualization_factor=1.0
        ),
        panel,
    )["pk"].to_numpy()
    log_hl = np.log(panel["high"].to_numpy() / panel["low"].to_numpy())
    term = log_hl**2 / (4 * _LN2)
    ref = np.sqrt(pd.Series(term).rolling(10).mean().to_numpy())
    assert np.allclose(out, ref, equal_nan=True)


def test_garman_klass_nonnegative_and_formula() -> None:
    panel = _ohlc(20, 3)
    out = _run(
        GarmanKlassVolatility(
            "open", "high", "low", "close", (10,), ("gk",), min_periods=10, annualization_factor=1.0
        ),
        panel,
    )["gk"].to_numpy()
    log_hl = np.log(panel["high"].to_numpy() / panel["low"].to_numpy())
    log_co = np.log(panel["close"].to_numpy() / panel["open"].to_numpy())
    term = 0.5 * log_hl**2 - (2 * _LN2 - 1) * log_co**2
    ref = np.sqrt(np.maximum(pd.Series(term).rolling(10).mean().to_numpy(), 0.0))
    assert np.allclose(out, ref, equal_nan=True)
    assert np.all(out[~np.isnan(out)] >= 0.0)


def test_yang_zhang_positive_and_reasonable() -> None:
    panel = _ohlc(40, 4)
    out = _run(
        YangZhangVolatility(
            "open", "high", "low", "close", (21,), ("yz",), min_periods=21, annualization_factor=1.0
        ),
        panel,
    )["yz"].to_numpy()
    valid = out[~np.isnan(out)]
    assert len(valid) > 0
    assert np.all(valid >= 0.0)
    assert np.all(valid < 1.0)  # daily vol far below 100%


def test_atr_wilder_matches_reference() -> None:
    panel = _ohlc(30, 5)
    out = _run(
        AverageTrueRange("high", "low", "close", (14,), ("atr",), min_periods=14, normalize=False),
        panel,
    )["atr"].to_numpy()
    high, low, close = (panel[c].to_numpy() for c in ("high", "low", "close"))
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr = np.where(np.isnan(prev_close), high - low, tr)  # first-bar true range = day range
    ref = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    assert np.allclose(out, ref, equal_nan=True)


def test_bollinger_pctb_and_bandwidth() -> None:
    panel = _ohlc(30, 6)
    out = _run(
        BollingerBands("close", (20,), ("pb",), ("bw",), num_std=2.0, min_periods=20, ddof=0), panel
    )
    close = panel["close"].to_numpy()
    mid = pd.Series(close).rolling(20).mean().to_numpy()
    sigma = pd.Series(close).rolling(20).std(ddof=0).to_numpy()
    upper, lower = mid + 2 * sigma, mid - 2 * sigma
    assert np.allclose(out["pb"].to_numpy(), (close - lower) / (upper - lower), equal_nan=True)
    assert np.allclose(out["bw"].to_numpy(), (upper - lower) / mid, equal_nan=True)


def test_ulcer_index_anchored_running_max() -> None:
    dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    close = [100.0, 90.0, 95.0, 80.0, 120.0, 110.0, 105.0, 130.0, 90.0, 100.0]
    panel = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close},
        index=pd.MultiIndex.from_product([pd.to_datetime(dates), ["A"]], names=["date", "entity"]),
    )
    out = _run(UlcerIndex("close", (4,), ("ui",)), panel)["ui"].to_numpy()
    arr = np.array(close)
    ref = np.full(len(arr), np.nan)
    for i in range(3, len(arr)):
        window = arr[i - 3 : i + 1]
        running_max = np.maximum.accumulate(window)  # anchored at window start
        drawdown = 100.0 * (window / running_max - 1.0)
        ref[i] = np.sqrt(np.mean(drawdown**2))
    assert np.allclose(out, ref, equal_nan=True)


def test_ulcer_and_maxdd_nonnegative() -> None:
    panel = _ohlc(60, 7)
    ulcer = _run(UlcerIndex("close", (20,), ("ui",)), panel)["ui"].to_numpy()
    mdd = _run(MaxDrawdown("close", (20,), ("mdd",)), panel)["mdd"].to_numpy()
    assert np.all(ulcer[~np.isnan(ulcer)] >= 0.0)
    assert np.all((mdd[~np.isnan(mdd)] >= 0.0) & (mdd[~np.isnan(mdd)] <= 1.0))
