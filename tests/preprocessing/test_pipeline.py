"""Tests for :class:`Pipeline` composing several transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nekron.preprocessing import (
    Adjustment,
    ConsistencyFilter,
    ConsistencyRule,
    CorporateAdjustment,
    ForwardFillImputer,
    NaNEntityFilter,
    Pipeline,
    TradingCalendarFilter,
)


def test_pipeline_composes_calendar_adjust_ffill_nanfilter() -> None:
    # Dates: one holiday (2021-01-01) that must be dropped, plus real sessions.
    dates = ["2020-12-31", "2021-01-01", "2021-01-04", "2021-01-05"]
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), ["A", "B"]], names=["date", "entity"]
    )
    # date-major interleave (A,B) per date d0=12-31, d1=01-01(holiday), d2=01-04, d3=01-05.
    # A close per date = [10, nan, 99, nan]; the d1 holiday row is dropped before ffill.
    # B close is entirely NaN -> dropped by the NaN filter.
    close = [10.0, np.nan, np.nan, np.nan, 99.0, np.nan, np.nan, np.nan]
    split = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    panel = pd.DataFrame({"close": close, "split": split}, index=index)

    pipeline = Pipeline(
        (
            TradingCalendarFilter("XNYS"),  # drop 2021-01-01 holiday rows
            CorporateAdjustment((Adjustment(("close",), "split", "divide"),)),
            ForwardFillImputer(columns=("close",)),
            NaNEntityFilter(columns=("close",), threshold=0.5),  # drop entity B (all-NaN close)
        )
    )
    result = pipeline.apply(panel)

    # Holiday rows removed.
    kept_dates = {
        ts.date().isoformat() for ts in result.index.get_level_values("date").normalize().unique()
    }
    assert "2021-01-01" not in kept_dates
    assert kept_dates == {"2020-12-31", "2021-01-04", "2021-01-05"}

    # Entity B dropped entirely (100% NaN close > 0.5).
    assert set(result.index.get_level_values("entity")) == {"A"}

    # A's close after divide then ffill:
    #   2020-12-31: 10 / 2 = 5.0
    #   2021-01-04: 99 / 1 = 99.0
    #   2021-01-05: NaN -> ffill from 99.0
    a_close = result.xs("A", level="entity")["close"].tolist()
    assert a_close == [5.0, 99.0, 99.0]


def test_pipeline_empty_steps_is_identity() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04"]), ["A"]], names=["date", "entity"]
    )
    panel = pd.DataFrame({"price": [1.0]}, index=index)
    result = Pipeline(()).apply(panel)
    assert result is panel


def test_pipeline_with_consistency_step() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-04", "2021-01-05"]), ["A"]],
        names=["date", "entity"],
    )
    panel = pd.DataFrame({"vol": [-5.0, 3.0]}, index=index)
    pipeline = Pipeline((ConsistencyFilter((ConsistencyRule("vol", "ge", bound=0.0),)),))
    result = pipeline.apply(panel)
    out = result["vol"].tolist()
    assert np.isnan(out[0]) and out[1] == 3.0  # -5 fails -> NaN
