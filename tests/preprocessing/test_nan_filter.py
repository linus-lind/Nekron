"""Tests for :class:`NaNEntityFilter`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.preprocessing import NaNEntityFilter

from .conftest import make_panel


def test_drops_group_strictly_above_threshold() -> None:
    # A: 1 NaN of 3 -> 1/3 ; B: 2 NaN of 3 -> 2/3
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {"price": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0]},
    )
    result = NaNEntityFilter(columns=("price",), threshold=0.5).apply(panel)
    entities = set(result.index.get_level_values("entity"))
    assert entities == {"A"}  # B (2/3 > 0.5) dropped, A (1/3 <= 0.5) kept


def test_boundary_equal_to_threshold_is_kept() -> None:
    # date-major interleave A,B per date. Values chosen so:
    #   A = [1.0, 2.0, nan, nan] -> 2 NaN of 4 = exactly 0.5
    #   B = [1.0, nan, nan, nan] -> 3 NaN of 4 = 0.75
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"],
        ["A", "B"],
        {"price": [1.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan]},
    )
    a_frac = panel.xs("A", level="entity")["price"].isna().mean()
    b_frac = panel.xs("B", level="entity")["price"].isna().mean()
    assert a_frac == 0.5 and b_frac == 0.75
    result = NaNEntityFilter(columns=("price",), threshold=0.5).apply(panel)
    entities = set(result.index.get_level_values("entity"))
    assert entities == {"A"}  # A exactly at threshold kept, B above dropped


def test_nothing_dropped_returns_same_object() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A", "B"],
        {"price": [1.0, 2.0, 3.0, 4.0]},  # no NaN anywhere
    )
    result = NaNEntityFilter(columns=("price",), threshold=0.5).apply(panel)
    assert result is panel


def test_subgroup_keys_split_by_column() -> None:
    # single entity A split into two periods; sparsity judged per period.
    index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]),
            ["A", "A", "A", "A"],
        ],
        names=["date", "entity"],
    )
    panel = pd.DataFrame(
        {
            "period": ["p1", "p1", "p2", "p2"],
            "price": [1.0, 2.0, np.nan, np.nan],  # p1: 0 NaN, p2: all NaN
        },
        index=index,
    )
    result = NaNEntityFilter(columns=("price",), threshold=0.5, subgroup_keys=("period",)).apply(
        panel
    )
    # p2 (fraction 1.0) dropped; p1 (fraction 0.0) kept
    assert result["period"].tolist() == ["p1", "p1"]
    assert result["price"].tolist() == [1.0, 2.0]


def test_entity_level_by_name() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {"price": [1.0, np.nan, 2.0, np.nan, np.nan, 3.0]},
    )
    result = NaNEntityFilter(columns=("price",), threshold=0.5, entity_level="entity").apply(panel)
    assert set(result.index.get_level_values("entity")) == {"A"}


def test_threshold_below_zero_raises() -> None:
    with pytest.raises(ValueError):
        NaNEntityFilter(columns=("price",), threshold=-0.1)


def test_threshold_above_one_raises() -> None:
    with pytest.raises(ValueError):
        NaNEntityFilter(columns=("price",), threshold=1.5)


def test_missing_column_raises() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"price": [1.0]})
    with pytest.raises(KeyError):
        NaNEntityFilter(columns=("nope",), threshold=0.5).apply(panel)


def test_drops_when_any_of_several_columns_over_threshold() -> None:
    # date-major layout: A rows are indices 0,2,4 ; B rows are 1,3,5.
    # A: price 0 NaN, vol 2/3 NaN -> dropped via vol. B: both 0 NaN -> kept.
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {
            "price": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "vol": [np.nan, 10.0, np.nan, 11.0, 5.0, 12.0],
        },
    )
    assert panel.xs("A", level="entity")["vol"].isna().mean() > 0.5
    result = NaNEntityFilter(columns=("price", "vol"), threshold=0.5).apply(panel)
    assert set(result.index.get_level_values("entity")) == {"B"}


def test_empty_columns_evaluates_all_columns() -> None:
    # omitting columns -> every column is evaluated; A is too sparse in 'vol'.
    panel = make_panel(
        ["2021-01-04", "2021-01-05", "2021-01-06"],
        ["A", "B"],
        {
            "price": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "vol": [np.nan, 10.0, np.nan, 11.0, 5.0, 12.0],
        },
    )
    filterer = NaNEntityFilter(threshold=0.5)
    assert filterer.columns == ()  # default: no columns -> all columns
    result = filterer.apply(panel)
    assert set(result.index.get_level_values("entity")) == {"B"}
