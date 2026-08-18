"""Trading-calendar filtering of a ``(date, entity)`` panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import exchange_calendars as xcals
import pandas as pd

from nekron.constants import DATE_LEVEL
from nekron.panel import PanelGrain

from .base import resolve_level


@dataclass(frozen=True)
class TradingCalendarFilter:
    """Drop rows whose date is not a trading session of an exchange calendar.

    Parameters
    ----------
    calendar:
        `exchange_calendars` code (e.g. ``"XNYS"``) whose sessions define the
        set of valid dates.
    date_level:
        Position or name of the date level in the panel index.
    """

    grains: ClassVar[tuple[PanelGrain, ...]] = (PanelGrain.PANEL, PanelGrain.TIME_SERIES)
    """Needs a date level to filter on; an entity-keyed or keyless frame has none."""

    calendar: str
    date_level: int | str = DATE_LEVEL

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        if len(panel) == 0:
            return panel
        pos = resolve_level(panel.index, self.date_level)
        dates = panel.index.get_level_values(pos)
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        unique_days = pd.DatetimeIndex(dates.unique())
        normalized = unique_days.normalize()
        if normalized.tz is not None:
            normalized = normalized.tz_localize(None)

        # Bound the calendar to the data range (default bounds cover only recent
        # decades). Read ``.sessions`` rather than ``sessions_in_range`` so a
        # non-session first/last date does not raise an out-of-bounds error.
        start, end = normalized.min(), normalized.max()
        calendar = xcals.get_calendar(self.calendar, start=start, end=end)
        sessions = calendar.sessions

        keep_days = unique_days[normalized.isin(sessions)]
        if len(keep_days) == len(unique_days):
            return panel
        # ``copy(deep=False)`` detaches the boolean-mask result (which already
        # owns its data) from its parent so later in-place column writes do not
        # raise a spurious chained-assignment warning; no data is duplicated.
        return panel.loc[dates.isin(keep_days)].copy(deep=False)
