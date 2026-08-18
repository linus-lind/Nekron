"""Shared helpers for preprocessing tests."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def make_panel(
    dates: Sequence[str],
    entities: Sequence[str],
    data: dict[str, Sequence[object]],
    *,
    date_name: str = "date",
    entity_name: str = "entity",
) -> pd.DataFrame:
    """Build a ``(date, entity)`` panel with a date-major MultiIndex.

    ``pd.MultiIndex.from_product`` lays rows out date-major, so ``data`` columns
    must be ordered ``(d0,e0), (d0,e1), ..., (d1,e0), ...``.
    """
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(list(dates)), list(entities)],
        names=[date_name, entity_name],
    )
    return pd.DataFrame({k: list(v) for k, v in data.items()}, index=index)
