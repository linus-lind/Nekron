"""Tests for :class:`CumulativeFactor` and the corrected corporate adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from nekron.preprocessing import (
    Adjustment,
    CorporateAdjustment,
    CumulativeFactor,
    Pipeline,
)

from .conftest import make_panel

_DATES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]


def test_back_adjusts_split_to_continuous_series() -> None:
    # 2:1 split ex-date at index 2: raw price halves, raw shares double there.
    panel = make_panel(
        _DATES,
        ["A"],
        {
            "DlyFacPrc": [1.0, 1.0, 2.0, 1.0, 1.0],
            "price": [20.0, 20.0, 10.0, 10.0, 10.0],
            "shares": [100.0, 100.0, 200.0, 200.0, 200.0],
        },
    )
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)
    # cumulative factor is 2 before the ex-date, 1 from the ex-date on
    assert out["CumFacPrc"].tolist() == [2.0, 2.0, 1.0, 1.0, 1.0]
    # dividing price / multiplying shares by it makes both continuous
    assert (out["price"] / out["CumFacPrc"]).tolist() == [10.0] * 5
    assert (out["shares"] * out["CumFacPrc"]).tolist() == [200.0] * 5


def test_no_distributions_is_all_one() -> None:
    panel = make_panel(_DATES[:2], ["A"], {"DlyFacPrc": [1.0, 1.0]})
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)
    assert out["CumFacPrc"].tolist() == [1.0, 1.0]


def test_multiple_splits_compound() -> None:
    # splits at index 1 (2:1) and index 3 (3:2); whole-entity product is 3.0
    panel = make_panel(_DATES[:4], ["A"], {"DlyFacPrc": [1.0, 2.0, 1.0, 1.5]})
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)
    assert out["CumFacPrc"].tolist() == [3.0, 1.5, 1.5, 1.0]


def test_guards_nonpositive_and_nan() -> None:
    panel = make_panel(_DATES[:3], ["A"], {"DlyFacPrc": [np.nan, 0.0, 2.0]})
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)
    # NaN and 0 are treated as 1 (no adjustment); only the 2.0 compounds
    assert out["CumFacPrc"].tolist() == [2.0, 2.0, 1.0]
    assert not np.isinf(out["CumFacPrc"].to_numpy()).any()


def test_no_cross_entity_bleed() -> None:
    # date-major: A splits at d1, B never; factors laid out (d0,A),(d0,B),(d1,A),...
    panel = make_panel(_DATES[:3], ["A", "B"], {"DlyFacPrc": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]})
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)
    assert out.xs("A", level="entity")["CumFacPrc"].tolist() == [2.0, 1.0, 1.0]
    assert out.xs("B", level="entity")["CumFacPrc"].tolist() == [1.0, 1.0, 1.0]


def test_unsorted_input_sorted_and_correct() -> None:
    panel = make_panel(_DATES[:3], ["A"], {"DlyFacPrc": [1.0, 2.0, 1.0]})
    reversed_panel = panel.iloc[::-1]  # date-descending -> not entity-time ordered
    assert not reversed_panel.index.is_monotonic_increasing
    out = CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(reversed_panel)
    assert out.index.is_monotonic_increasing
    assert out.xs("A", level="entity")["CumFacPrc"].tolist() == [2.0, 1.0, 1.0]


def test_missing_factor_column_raises() -> None:
    panel = make_panel(_DATES[:1], ["A"], {"other": [1.0]})
    with pytest.raises(KeyError):
        CumulativeFactor("DlyFacPrc", "CumFacPrc").apply(panel)


def test_pipeline_cumulative_then_adjust_gives_continuous_ohlcv() -> None:
    # end-to-end: the two-step (compound factor -> apply) adjustment as configured.
    panel = make_panel(
        _DATES,
        ["A"],
        {
            "DlyFacPrc": [1.0, 1.0, 2.0, 1.0, 1.0],
            "DlyClose": [20.0, 21.0, 10.5, 10.0, 11.0],  # raw, halves at the split
            "ShrOut": [100.0, 100.0, 200.0, 200.0, 200.0],
            "DlyVol": [50.0, 60.0, 140.0, 80.0, 90.0],  # raw share volume doubles
        },
    )
    pipeline = Pipeline(
        (
            CumulativeFactor("DlyFacPrc", "CumFacPrc"),
            CorporateAdjustment(
                (
                    Adjustment(("DlyClose",), "CumFacPrc", "divide"),
                    Adjustment(("ShrOut", "DlyVol"), "CumFacPrc", "multiply"),
                )
            ),
        )
    )
    out = pipeline.apply(panel)
    close = out["DlyClose"].tolist()
    # pre-split closes back-adjusted to post-split basis (divided by 2)
    assert close == [10.0, 10.5, 10.5, 10.0, 11.0]
    # shares are continuous at 200 across the split
    assert out["ShrOut"].tolist() == [200.0] * 5
    # dollar volume (close * volume) is unchanged by the adjustment (split-invariant)
    raw_close = [20.0, 21.0, 10.5, 10.0, 11.0]
    raw_vol = [50.0, 60.0, 140.0, 80.0, 90.0]
    adj_dollar_vol = [c * v for c, v in zip(out["DlyClose"], out["DlyVol"], strict=True)]
    assert adj_dollar_vol == [rc * rv for rc, rv in zip(raw_close, raw_vol, strict=True)]
