"""Tests for :class:`ForwardFillImputer`."""

from __future__ import annotations

import numpy as np
import pytest

from nekron.preprocessing import ConstantImputer, ForwardFillImputer

from .conftest import make_panel


def test_fills_within_entity_along_date_axis() -> None:
    # date-major layout: (d0,A),(d0,B),(d1,A),(d1,B),(d2,A),(d2,B)
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {"price": [1.0, 100.0, np.nan, np.nan, np.nan, np.nan]},
    )
    result = ForwardFillImputer(columns=("price",)).apply(panel)
    # A's series 1 -> 1 -> 1 ; B's series 100 -> 100 -> 100
    a = result.xs("A", level="entity")["price"].tolist()
    b = result.xs("B", level="entity")["price"].tolist()
    assert a == [1.0, 1.0, 1.0]
    assert b == [100.0, 100.0, 100.0]


def test_no_cross_entity_bleed() -> None:
    # A leads with a value then NaN; B leads with NaN then a value.
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A", "B"],
        {"price": [1.0, np.nan, np.nan, 2.0]},  # (d0,A)=1,(d0,B)=nan,(d1,A)=nan,(d1,B)=2
    )
    result = ForwardFillImputer(columns=("price",)).apply(panel)
    a = result.xs("A", level="entity")["price"].tolist()
    b = result.xs("B", level="entity")["price"].tolist()
    assert a == [1.0, 1.0]  # A's own value carried forward
    assert np.isnan(b[0])  # B has no prior value: NaN stays, A's 1.0 did NOT bleed in
    assert b[1] == 2.0


def test_limit_caps_consecutive_fills() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A"],
        {"price": [1.0, np.nan, np.nan]},
    )
    result = ForwardFillImputer(columns=("price",), limit=1).apply(panel)
    vals = result["price"].tolist()
    assert vals[0] == 1.0
    assert vals[1] == 1.0  # first gap filled
    assert np.isnan(vals[2])  # second consecutive NaN not filled


def test_unsorted_index_sorted_and_filled_correctly() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {"price": [1.0, 100.0, np.nan, np.nan, np.nan, np.nan]},
    )
    shuffled = panel.sample(frac=1.0, random_state=7)
    assert not shuffled.index.is_monotonic_increasing

    result = ForwardFillImputer(columns=("price",)).apply(shuffled)
    assert result.index.is_monotonic_increasing
    assert result.xs("A", level="entity")["price"].tolist() == [1.0, 1.0, 1.0]
    assert result.xs("B", level="entity")["price"].tolist() == [100.0, 100.0, 100.0]


def test_empty_columns_fills_all_columns() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A"],
        {"open": [1.0, np.nan], "close": [2.0, np.nan]},
    )
    result = ForwardFillImputer().apply(panel)
    assert result["open"].tolist() == [1.0, 1.0]
    assert result["close"].tolist() == [2.0, 2.0]


def test_only_named_columns_filled() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A"],
        {"open": [1.0, np.nan], "close": [2.0, np.nan]},
    )
    result = ForwardFillImputer(columns=("open",)).apply(panel)
    assert result["open"].tolist() == [1.0, 1.0]
    assert np.isnan(result["close"].tolist()[1])  # not selected -> stays NaN


def test_entity_level_by_name() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A", "B"],
        {"price": [1.0, 100.0, np.nan, np.nan]},
    )
    result = ForwardFillImputer(columns=("price",), entity_level="entity").apply(panel)
    assert result.xs("A", level="entity")["price"].tolist() == [1.0, 1.0]
    assert result.xs("B", level="entity")["price"].tolist() == [100.0, 100.0]


def test_missing_column_raises() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"price": [1.0]})
    with pytest.raises(KeyError):
        ForwardFillImputer(columns=("nope",)).apply(panel)


def test_constant_imputer_fills_with_value() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A", "B"],
        {"vol": [np.nan, 5.0, np.nan, np.nan], "price": [1.0, np.nan, 2.0, np.nan]},
    )
    result = ConstantImputer(columns=("vol",), value=0.0).apply(panel)
    # every missing vol becomes 0, across entities and without ordering
    assert result["vol"].tolist() == [0.0, 5.0, 0.0, 0.0]
    # untouched column keeps its NaN (not selected)
    assert np.isnan(result["price"].tolist()[1])


def test_constant_imputer_nonzero_value() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"x": [np.nan]})
    result = ConstantImputer(columns=("x",), value=-1.0).apply(panel)
    assert result["x"].tolist() == [-1.0]


def test_constant_imputer_missing_column_raises() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"price": [1.0]})
    with pytest.raises(KeyError):
        ConstantImputer(columns=("nope",), value=0.0).apply(panel)
