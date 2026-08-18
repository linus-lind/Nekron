"""Tests for :class:`DuplicateMerger`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.preprocessing import DuplicateMerger


def _panel(tuples: list[tuple[str, str]], data: dict[str, list[object]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), e) for d, e in tuples], names=["date", "entity"]
    )
    return pd.DataFrame(data, index=index)


def test_numeric_nan_skipping_mean_and_object_first_non_null() -> None:
    panel = _panel(
        [("2021-01-04", "A"), ("2021-01-04", "A"), ("2021-01-05", "B")],
        {"price": [10.0, np.nan, 5.0], "name": [None, "x", "y"]},
    )
    result = DuplicateMerger().apply(panel)

    assert len(result) == 2
    row = result.loc[(pd.Timestamp("2021-01-04"), "A")]
    # mean over {10.0} (NaN skipped) == 10.0
    assert row["price"] == 10.0
    # first non-null object value
    assert row["name"] == "x"
    # untouched unique row passes through
    assert result.loc[(pd.Timestamp("2021-01-05"), "B"), "price"] == 5.0


def test_mean_of_two_values() -> None:
    panel = _panel(
        [("2021-01-04", "A"), ("2021-01-04", "A")],
        {"price": [2.0, 4.0]},
    )
    result = DuplicateMerger().apply(panel)
    assert result.loc[(pd.Timestamp("2021-01-04"), "A"), "price"] == 3.0


def test_result_index_is_unique_and_sorted() -> None:
    panel = _panel(
        [
            ("2021-01-05", "B"),
            ("2021-01-04", "A"),
            ("2021-01-04", "A"),
            ("2021-01-06", "C"),
        ],
        {"price": [1.0, 2.0, 3.0, 4.0]},
    )
    result = DuplicateMerger().apply(panel)
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing


def test_no_duplicates_returns_same_object() -> None:
    panel = _panel(
        [("2021-01-04", "A"), ("2021-01-05", "B")],
        {"price": [1.0, 2.0]},
    )
    result = DuplicateMerger().apply(panel)
    assert result is panel


def test_column_name_keys() -> None:
    panel = _panel(
        [("2021-01-04", "A"), ("2021-01-04", "A"), ("2021-01-05", "B")],
        {"grp": ["g", "g", "h"], "v": [2.0, 4.0, 9.0]},
    )
    result = DuplicateMerger(keys=("grp",)).apply(panel)
    # the two "g" rows collapse; "h" passes through
    assert len(result) == 2
    grp_values = sorted(result["grp"].tolist())
    assert grp_values == ["g", "h"]
    g_row = result[result["grp"] == "g"].iloc[0]
    assert g_row["v"] == 3.0  # mean(2, 4)


def test_index_level_name_as_key_merges_across_entities() -> None:
    # Keying on the date level only: two rows on the same date merge even
    # though their entity differs.
    panel = _panel(
        [("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-05", "C")],
        {"v": [2.0, 6.0, 9.0]},
    )
    result = DuplicateMerger(keys=("date",)).apply(panel)
    assert len(result) == 2
    # date 2021-01-04 collapses to mean(2, 6) == 4.0; 2021-01-05 passes through
    assert sorted(result["v"].tolist()) == [4.0, 9.0]


def test_unnamed_index_levels_raise() -> None:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-01-04"), "A"), (pd.Timestamp("2021-01-04"), "A")]
    )
    panel = pd.DataFrame({"price": [1.0, 2.0]}, index=index)
    with pytest.raises(ValueError):
        DuplicateMerger().apply(panel)
