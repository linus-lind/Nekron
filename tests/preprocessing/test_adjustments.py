"""Tests for :class:`Adjustment` and :class:`CorporateAdjustment`."""

from __future__ import annotations

import numpy as np
import pytest

from nekron.preprocessing import Adjustment, CorporateAdjustment

from .conftest import make_panel


def test_divide_operation() -> None:
    panel = make_panel(
        ["2021-01-04", "2021-01-05"],
        ["A"],
        {"close": [10.0, 20.0], "split": [2.0, 4.0]},
    )
    result = CorporateAdjustment((Adjustment(("close",), "split", "divide"),)).apply(panel)
    assert result["close"].tolist() == [5.0, 5.0]
    # factor column untouched
    assert result["split"].tolist() == [2.0, 4.0]


def test_multiply_operation_multiple_columns() -> None:
    panel = make_panel(
        ["2021-01-04"],
        ["A", "B"],
        {"shares": [100.0, 50.0], "vwap": [3.0, 6.0], "factor": [2.0, 0.5]},
    )
    adj = Adjustment(("shares", "vwap"), "factor", "multiply")
    result = CorporateAdjustment((adj,)).apply(panel)
    assert result["shares"].tolist() == [200.0, 25.0]
    assert result["vwap"].tolist() == [6.0, 3.0]


def test_applies_in_place_returns_same_object() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"close": [10.0], "split": [2.0]})
    result = CorporateAdjustment((Adjustment(("close",), "split", "divide"),)).apply(panel)
    assert result is panel


def test_sequence_of_adjustments_applied_in_order() -> None:
    panel = make_panel(
        ["2021-01-04"],
        ["A"],
        {"close": [10.0], "split": [2.0], "cash": [4.0]},
    )
    adjustments = (
        Adjustment(("close",), "split", "divide"),  # 10 / 2 = 5
        Adjustment(("close",), "cash", "multiply"),  # 5 * 4 = 20
    )
    result = CorporateAdjustment(adjustments).apply(panel)
    assert result["close"].tolist() == [20.0]


def test_nan_factor_propagates() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"close": [10.0], "split": [np.nan]})
    result = CorporateAdjustment((Adjustment(("close",), "split", "divide"),)).apply(panel)
    assert np.isnan(result["close"].iloc[0])


def test_invalid_operation_raises_at_construction() -> None:
    with pytest.raises(ValueError):
        Adjustment(("close",), "split", "subtract")


def test_empty_columns_raises_at_construction() -> None:
    with pytest.raises(ValueError):
        Adjustment((), "split", "divide")


def test_missing_column_raises_on_apply() -> None:
    panel = make_panel(["2021-01-04"], ["A"], {"close": [10.0]})
    with pytest.raises(KeyError):
        CorporateAdjustment((Adjustment(("close",), "split", "divide"),)).apply(panel)
