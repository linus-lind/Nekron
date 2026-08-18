"""Tests for :class:`ConsistencyRule` and :class:`ConsistencyFilter`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.preprocessing import ConsistencyFilter, ConsistencyRule


def _frame(data: dict[str, list[object]]) -> pd.DataFrame:
    n = len(next(iter(data.values())))
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime([f"2021-01-{4 + i:02d}" for i in range(n)]), ["A"] * n],
        names=["date", "entity"],
    )
    return pd.DataFrame(data, index=index)


def test_value_mode_sets_failing_rows_to_nan() -> None:
    # valid condition: vol > 0
    panel = _frame({"vol": [5.0, -3.0, np.nan, 0.0]})
    result = ConsistencyFilter((ConsistencyRule("vol", "gt", bound=0.0),)).apply(panel)
    out = result["vol"].tolist()
    assert out[0] == 5.0  # valid, untouched
    assert np.isnan(out[1])  # -3 fails -> NaN
    assert np.isnan(out[2])  # NaN operand left untouched
    assert np.isnan(out[3])  # 0 is not > 0 -> NaN


def test_value_mode_ge_boundary_kept() -> None:
    # valid condition: x >= 0 ; exactly 0 is valid and untouched, -1 fails -> NaN
    panel = _frame({"x": [0.0, -1.0]})
    result = ConsistencyFilter((ConsistencyRule("x", "ge", bound=0.0),)).apply(panel)
    out = result["x"].tolist()
    assert out[0] == 0.0
    assert np.isnan(out[1])


def test_column_mode_sets_target_to_nan() -> None:
    # valid condition: low <= high ; on failure set high to NaN
    panel = _frame({"low": [1.0, 9.0, np.nan], "high": [5.0, 4.0, 7.0]})
    rule = ConsistencyRule("low", "le", other_column="high", target="high")
    result = ConsistencyFilter((rule,)).apply(panel)
    low = result["low"].tolist()
    assert low[0] == 1.0 and low[1] == 9.0 and np.isnan(low[2])  # left column untouched
    high = result["high"].tolist()
    assert high[0] == 5.0  # 1 <= 5 valid
    assert np.isnan(high[1])  # 9 <= 4 fails -> high NaN
    assert high[2] == 7.0  # low NaN operand -> untouched


def test_column_mode_target_is_left_column() -> None:
    panel = _frame({"low": [1.0, 9.0], "high": [5.0, 4.0]})
    rule = ConsistencyRule("low", "le", other_column="high", target="low")
    result = ConsistencyFilter((rule,)).apply(panel)
    low = result["low"].tolist()
    assert low[0] == 1.0
    assert np.isnan(low[1])  # failing row: target (low) set to NaN
    assert result["high"].tolist() == [5.0, 4.0]  # other column untouched


def test_nan_operand_row_untouched_column_mode() -> None:
    # both a NaN-left and a NaN-right row must be left alone
    panel = _frame({"low": [np.nan, 3.0, 8.0], "high": [5.0, np.nan, 4.0]})
    rule = ConsistencyRule("low", "le", other_column="high", target="high")
    result = ConsistencyFilter((rule,)).apply(panel)
    high = result["high"].tolist()
    assert high[0] == 5.0  # low NaN -> untouched
    assert np.isnan(high[1])  # high already NaN -> untouched (stays NaN)
    assert np.isnan(high[2])  # 8 <= 4 fails -> high set to NaN


def test_applies_in_place_returns_same_object() -> None:
    panel = _frame({"vol": [5.0, -3.0]})
    result = ConsistencyFilter((ConsistencyRule("vol", "gt", bound=0.0),)).apply(panel)
    assert result is panel


def test_invalid_op_raises() -> None:
    with pytest.raises(ValueError):
        ConsistencyRule("vol", "eq", bound=0.0)


def test_both_bound_and_other_column_raises() -> None:
    with pytest.raises(ValueError):
        ConsistencyRule("low", "le", bound=0.0, other_column="high", target="high")


def test_neither_bound_nor_other_column_raises() -> None:
    with pytest.raises(ValueError):
        ConsistencyRule("low", "le")


def test_column_mode_missing_target_raises() -> None:
    with pytest.raises(ValueError):
        ConsistencyRule("low", "le", other_column="high")


def test_column_mode_bad_target_raises() -> None:
    with pytest.raises(ValueError):
        ConsistencyRule("low", "le", other_column="high", target="close")
