"""Tests for :class:`TradingCalendarFilter`."""

from __future__ import annotations

import pandas as pd

from nekron.preprocessing import TradingCalendarFilter

# 2020-12-31 Thu  -> session
# 2021-01-01 Fri  -> New Year holiday (not a session)
# 2021-01-02 Sat  -> weekend (not a session)
# 2021-01-04 Mon  -> session
# 2021-01-05 Tue  -> session


def _panel(dates: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([pd.to_datetime(dates), ["A"]], names=["date", "entity"])
    return pd.DataFrame({"price": range(len(index))}, index=index)


def test_drops_holiday_and_weekend_keeps_sessions() -> None:
    panel = _panel(["2020-12-31", "2021-01-01", "2021-01-02", "2021-01-04"])
    result = TradingCalendarFilter("XNYS").apply(panel)

    kept = result.index.get_level_values("date").normalize().unique()
    kept_dates = {ts.date().isoformat() for ts in kept}
    assert kept_dates == {"2020-12-31", "2021-01-04"}
    # New Year holiday and the Saturday are gone.
    assert pd.Timestamp("2021-01-01") not in result.index.get_level_values("date")
    assert pd.Timestamp("2021-01-02") not in result.index.get_level_values("date")


def test_all_sessions_is_noop_fast_path_identity() -> None:
    panel = _panel(["2020-12-31", "2021-01-04", "2021-01-05"])
    result = TradingCalendarFilter("XNYS").apply(panel)
    assert result is panel


def test_empty_panel_returned_as_is() -> None:
    index = pd.MultiIndex.from_arrays([pd.to_datetime([]), []], names=["date", "entity"])
    panel = pd.DataFrame({"price": []}, index=index)
    result = TradingCalendarFilter("XNYS").apply(panel)
    assert result is panel


def test_date_level_by_name() -> None:
    panel = _panel(["2020-12-31", "2021-01-01", "2021-01-04"])
    result = TradingCalendarFilter("XNYS", date_level="date").apply(panel)
    kept = {
        ts.date().isoformat() for ts in result.index.get_level_values("date").normalize().unique()
    }
    assert kept == {"2020-12-31", "2021-01-04"}
