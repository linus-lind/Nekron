"""Regression guards for fixes surfaced during review."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nekron.preprocessing import (
    Adjustment,
    CorporateAdjustment,
    ForwardFillImputer,
    NaNEntityFilter,
    Pipeline,
    TradingCalendarFilter,
)

from .conftest import make_panel


def test_imputer_preserves_per_column_dtypes() -> None:
    """Filling all columns must not collapse mixed dtypes to a single dtype."""
    panel = make_panel(
        ["2020-01-01", "2020-01-02"],
        ["A", "B"],
        {
            "price": [1.0, 3.0, np.nan, np.nan],
            "sector": ["fin", "tech", None, None],
            "volume": [10, 30, 20, 40],
        },
    )
    expected = panel.dtypes
    out = ForwardFillImputer().apply(panel.copy())
    # Compare against the input's own dtypes rather than naming them: the dtype a
    # column of strings gets differs between pandas versions, but "unchanged by
    # filling" is the invariant this guards either way.
    assert out.dtypes.equals(expected)
    assert out["price"].dtype == np.float64
    assert out["volume"].dtype == np.int64
    # A had no NaN gaps; B's (2020-01-02) rows forward-fill from (2020-01-01).
    assert out.loc[(pd.Timestamp("2020-01-02"), "B"), "sector"] == "tech"


def test_imputer_entity_major_input_not_reordered() -> None:
    """An (entity, date)-sorted panel is already fill-ready; no sort needed."""
    idx = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            ["A", "A", "B", "B"],
        ],
        names=["date", "entity"],
    )
    panel = pd.DataFrame({"price": [10.0, np.nan, 20.0, np.nan]}, index=idx)
    out = ForwardFillImputer(columns=("price",)).apply(panel.copy())
    assert out["price"].tolist() == [10.0, 10.0, 20.0, 20.0]
    assert list(out.index) == list(idx)  # order preserved, not sorted


def test_nan_filter_keeps_rows_with_nan_subgroup_key() -> None:
    """Rows outside any subgroup (NaN key) are judged on their own fraction."""
    panel = make_panel(
        ["2020-01-01", "2020-01-02"],
        ["A", "B"],
        {
            "price": [1.0, 2.0, 3.0, 4.0],  # no NaNs at all
            "period": [np.nan, np.nan, np.nan, np.nan],  # all outside membership
        },
    )
    out = NaNEntityFilter(columns=("price",), threshold=0.5, subgroup_keys=("period",)).apply(
        panel.copy()
    )
    assert len(out) == 4  # nothing dropped despite NaN subgroup keys


def test_calendar_handles_non_session_boundary_dates() -> None:
    """First/last panel dates being non-sessions must not raise out-of-bounds."""
    panel = make_panel(
        ["2020-01-04", "2020-01-06", "2020-01-11"],  # Sat, Mon (session), Sat
        ["A"],
        {"price": [1.0, 2.0, 3.0]},
    )
    out = TradingCalendarFilter(calendar="XNYS").apply(panel.copy())
    kept = out.index.get_level_values("date").unique()
    assert list(kept) == [pd.Timestamp("2020-01-06")]


def test_pipeline_write_after_filter_emits_no_setting_with_copy_warning() -> None:
    """A column-writer after a row-dropping filter must not warn on the slice."""
    panel = make_panel(
        ["2020-01-04", "2020-01-06"],  # 01-04 is a Saturday (dropped by XNYS)
        ["A", "B"],
        {"price": [10.0, 20.0, 12.0, 24.0], "factor": [2.0, 2.0, 2.0, 2.0]},
    )
    pipe = Pipeline(
        (
            TradingCalendarFilter(calendar="XNYS"),
            CorporateAdjustment(
                (Adjustment(columns=("price",), factor_column="factor", operation="divide"),)
            ),
        )
    )
    with warnings.catch_warnings():
        # SettingWithCopyWarning was removed in pandas 3.0 and
        # ChainedAssignmentError does not exist in 2.2, so trap whichever the
        # running version defines. Both are raised as warnings, not exceptions.
        for name in ("SettingWithCopyWarning", "ChainedAssignmentError"):
            category = getattr(pd.errors, name, None)
            if category is not None:
                warnings.simplefilter("error", category)
        out = pipe.apply(panel.copy())
    assert out.loc[(pd.Timestamp("2020-01-06"), "A"), "price"] == 6.0
