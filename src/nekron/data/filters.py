"""Configurable filters applied while ingesting a ``(date, entity)`` panel.

A filter is a small, composable step selected by a short ``type`` name and a
parameter mapping (resolved through the registry below), so the full filter set is
declared — and versioned — in configuration. Each filter declares a
:class:`FilterPhase`:

* ``ROW`` — stateless, applied to each raw chunk *during* the chunked read (before
  rows accumulate), so it reduces peak memory.
* ``CROSS_SECTION`` — applied once to the assembled ``(date, entity)``-indexed
  panel, for filters that need a whole period's cross-section (e.g. keeping the
  largest entities per date).

Adding a filter means writing a class satisfying :class:`PanelFilter` and
registering it here under a name — no calling code changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from nekron.constants import DATE_LEVEL

from .base import IngestionError


class FilterPhase(Enum):
    """When a filter runs during ingestion."""

    ROW = "row"
    CROSS_SECTION = "cross_section"


@runtime_checkable
class PanelFilter(Protocol):
    """A filter applied to a panel (a raw chunk for ``ROW``, the indexed panel otherwise)."""

    @property
    def phase(self) -> FilterPhase:
        """The ingestion phase at which the filter is applied."""
        ...

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return the filtered panel."""
        ...


def _require(panel: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise IngestionError(f"filter requires columns absent from the panel: {missing}.")


@dataclass(frozen=True)
class TopNByMarketCap:
    """Keep the ``n`` largest entities by market capitalization in each period.

    Market cap is read from ``market_cap_column`` when set, otherwise computed as
    ``|price| * shares`` from ``price_column`` and ``shares_column`` (the absolute
    price accommodates sign conventions that mark non-traded prices negative).
    Entities are ranked per date (index ``date_level``), largest first, and rows
    whose market cap is missing are dropped.
    """

    n: int
    market_cap_column: str | None = None
    price_column: str | None = None
    shares_column: str | None = None
    date_level: int | str = DATE_LEVEL

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive.")
        by_column = self.market_cap_column is not None
        by_pair = self.price_column is not None and self.shares_column is not None
        if by_column == by_pair:
            raise ValueError(
                "set exactly one of market_cap_column or (price_column and shares_column)."
            )

    @property
    def phase(self) -> FilterPhase:
        return FilterPhase.CROSS_SECTION

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        market_cap = self._market_cap(panel)
        rank = market_cap.groupby(level=self.date_level).rank(method="first", ascending=False)
        # ``fillna(False)`` before the cast because a nullable market-cap dtype
        # (Float64/Int64) makes rank() nullable too, and the comparison then yields
        # pandas' `boolean` dtype whose ``to_numpy()`` is an object array holding
        # pd.NA — which .loc refuses, with an error naming neither this filter nor
        # the column. A missing market cap is not a top-n entity anyway.
        return panel.loc[(rank <= self.n).fillna(False).to_numpy(dtype=bool)]

    def _market_cap(self, panel: pd.DataFrame) -> pd.Series[float]:
        if self.market_cap_column is not None:
            _require(panel, (self.market_cap_column,))
            return panel[self.market_cap_column].abs()
        assert self.price_column is not None and self.shares_column is not None
        _require(panel, (self.price_column, self.shares_column))
        return panel[self.price_column].abs() * panel[self.shares_column]


FilterBuilder = Callable[[Mapping[str, Any]], PanelFilter]

_REGISTRY: dict[str, FilterBuilder] = {}


def register_filter(name: str, builder: FilterBuilder) -> None:
    """Register a filter builder under a type ``name`` (overwrites in place)."""
    _REGISTRY[name] = builder


def registered_filters() -> tuple[str, ...]:
    """Return the sorted names of all registered filter types."""
    return tuple(sorted(_REGISTRY))


def build_filter(name: str, params: Mapping[str, Any]) -> PanelFilter:
    """Construct the filter registered under ``name`` from ``params``."""
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise IngestionError(
            f"unknown filter type {name!r}; registered types: {registered_filters()}."
        ) from None
    try:
        return builder(params)
    except (TypeError, ValueError, KeyError) as exc:
        raise IngestionError(f"cannot build filter {name!r}: {exc}.") from exc


def _direct(cls: Callable[..., PanelFilter]) -> FilterBuilder:
    """Builder for a filter whose params map directly onto its constructor."""

    def build(params: Mapping[str, Any]) -> PanelFilter:
        return cls(**params)

    return build


_DEFAULTS: dict[str, FilterBuilder] = {
    "top_n_market_cap": _direct(TopNByMarketCap),
}

for _name, _builder in _DEFAULTS.items():
    register_filter(_name, _builder)
