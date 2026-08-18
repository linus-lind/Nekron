"""Shared helpers for feature-creation tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nekron.features.base import PanelContext


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


def single_entity_panel(dates: Sequence[str], data: dict[str, Sequence[object]]) -> pd.DataFrame:
    """Build a one-entity ``(date, entity)`` panel for per-series checks."""
    return make_panel(dates, ["A"], {k: list(v) for k, v in data.items()})


def random_ohlcv(n_dates: int, entities: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """Build a random but internally consistent OHLCV ``(date, entity)`` panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, list(entities)], names=["date", "entity"])
    n = len(index)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(10_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dollar_volume": close * volume,
            "shares": rng.integers(1_000_000, 100_000_000, n).astype(float),
            "ret": rng.normal(0, 0.01, n),
        },
        index=index,
    )


def em_context(panel: pd.DataFrame) -> tuple[pd.DataFrame, PanelContext]:
    """Return an entity-major-sorted copy of ``panel`` and its PanelContext.

    Engine primitives require values aligned to the context's row order, so tests
    read their input columns from the returned entity-major frame.
    """
    entity_major = panel.swaplevel().sort_index().swaplevel()
    ctx = PanelContext.from_panel(entity_major, entity_level=1, date_level=0)
    return entity_major, ctx
