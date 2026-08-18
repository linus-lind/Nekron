"""Tests for return-based featurizers."""

from __future__ import annotations

import numpy as np

from nekron.features import (
    CompoundReturns,
    FeaturePipeline,
    ForwardReturns,
    IntradayReturn,
    LogReturns,
    OvernightReturn,
    SimpleReturns,
)

from .conftest import single_entity_panel

_DATES = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]


def _run(featurizer, panel):
    return FeaturePipeline(featurizers=(featurizer,)).apply(panel)


def test_simple_returns_values() -> None:
    panel = single_entity_panel(_DATES, {"close": [100.0, 110.0, 99.0, 99.0]})
    out = _run(SimpleReturns("close", (1,), ("r1",)), panel)["r1"].to_numpy()
    assert np.isnan(out[0])
    assert np.isclose(out[1], 0.10)
    assert np.isclose(out[2], -0.10)
    assert np.isclose(out[3], 0.0)


def test_log_returns_equal_log_of_simple() -> None:
    panel = single_entity_panel(_DATES, {"close": [100.0, 110.0, 99.0, 105.0]})
    log_out = _run(LogReturns("close", (1,), ("lr",)), panel)["lr"].to_numpy()
    simple = _run(SimpleReturns("close", (1,), ("r",)), panel)["r"].to_numpy()
    assert np.allclose(log_out[1:], np.log1p(simple[1:]), equal_nan=True)


def test_forward_returns_are_leads_with_trailing_nan() -> None:
    panel = single_entity_panel(_DATES, {"close": [100.0, 110.0, 99.0, 105.0]})
    out = _run(ForwardReturns("close", (1,), ("fwd",), method="simple"), panel)["fwd"].to_numpy()
    # fwd_t = close_{t+1}/close_t - 1 ; last is NaN
    assert np.isclose(out[0], 110.0 / 100.0 - 1)
    assert np.isclose(out[1], 99.0 / 110.0 - 1)
    assert np.isnan(out[3])


def test_compound_returns_matches_product() -> None:
    rets = [0.01, -0.02, 0.03, 0.00]
    panel = single_entity_panel(_DATES, {"ret": rets})
    out = _run(CompoundReturns("ret", (3,), ("c3",), method="simple", min_periods=3), panel)
    vals = out["c3"].to_numpy()
    expected = (1 + rets[0]) * (1 + rets[1]) * (1 + rets[2]) - 1
    assert np.isclose(vals[2], expected)
    assert np.isnan(vals[1])  # window not full


def test_intraday_and_overnight_decompose() -> None:
    panel = single_entity_panel(
        _DATES, {"open": [100.0, 102.0, 101.0, 100.0], "close": [101.0, 103.0, 100.0, 99.0]}
    )
    intraday = _run(IntradayReturn("open", "close", "id", method="simple"), panel)["id"].to_numpy()
    overnight = _run(OvernightReturn("open", "close", "on", method="simple"), panel)[
        "on"
    ].to_numpy()
    assert np.isclose(intraday[0], 101.0 / 100.0 - 1)
    assert np.isnan(overnight[0])  # needs prior close
    assert np.isclose(overnight[1], 102.0 / 101.0 - 1)


def test_returns_robust_to_zero_price() -> None:
    panel = single_entity_panel(_DATES, {"close": [100.0, 0.0, 50.0, 50.0]})
    log_out = _run(LogReturns("close", (1,), ("lr",)), panel)["lr"].to_numpy()
    assert np.isnan(log_out[1]) and np.isnan(log_out[2])  # log of/at zero -> NaN
    assert not np.isinf(log_out).any()
