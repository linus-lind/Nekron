"""Tests for ingestion filters and the filter registry."""

from __future__ import annotations

import pandas as pd
import pytest

from nekron.constants import DATE_LEVEL
from nekron.data import (
    FilterPhase,
    IngestionError,
    TopNByMarketCap,
    build_filter,
    register_filter,
    registered_filters,
)

_D1 = pd.Timestamp("2020-01-02")
_D2 = pd.Timestamp("2020-01-03")


def _entity_major_panel() -> pd.DataFrame:
    """A panel whose index levels are ordered ``(entity, date)``.

    Date is level 1 here, not level 0, and the caps are chosen so that "largest
    per date" and "largest per entity" select different rows. Any filter that
    grouped by index *position* rather than by level name would therefore return a
    plausible — and wrong — answer on this frame instead of raising.
    """
    index = pd.MultiIndex.from_tuples(
        [("A", _D1), ("A", _D2), ("B", _D1), ("B", _D2)],
        names=["entity", DATE_LEVEL],
    )
    return pd.DataFrame({"cap": [10.0, 20.0, 5.0, 8.0]}, index=index)


def test_top_n_defaults_to_the_named_date_level() -> None:
    assert TopNByMarketCap(n=1, market_cap_column="cap").date_level == DATE_LEVEL


def test_top_n_selects_per_date_not_per_index_position() -> None:
    panel = _entity_major_panel()
    kept = TopNByMarketCap(n=1, market_cap_column="cap").apply(panel)
    assert list(kept.index) == [("A", _D1), ("A", _D2)]


def test_positional_level_zero_would_select_per_entity_instead() -> None:
    # The discriminating half of the test above: on this frame level 0 is entity,
    # so a positional default picks a different set of rows.
    panel = _entity_major_panel()
    kept = TopNByMarketCap(n=1, market_cap_column="cap", date_level=0).apply(panel)
    assert list(kept.index) == [("A", _D2), ("B", _D2)]


def test_top_n_keeps_n_rows_per_date() -> None:
    panel = _entity_major_panel()
    kept = TopNByMarketCap(n=2, market_cap_column="cap").apply(panel)
    assert len(kept) == 4
    assert kept.index.equals(panel.index)


def test_top_n_on_a_date_major_panel() -> None:
    index = pd.MultiIndex.from_tuples(
        [(_D1, "A"), (_D1, "B"), (_D2, "A"), (_D2, "B")], names=[DATE_LEVEL, "entity"]
    )
    panel = pd.DataFrame({"cap": [10.0, 5.0, 20.0, 8.0]}, index=index)
    kept = TopNByMarketCap(n=1, market_cap_column="cap").apply(panel)
    assert list(kept.index) == [(_D1, "A"), (_D2, "A")]


def test_market_cap_from_price_and_shares_uses_absolute_price() -> None:
    # A negative price marks a non-traded quote; its magnitude is still the size.
    panel = _entity_major_panel().assign(prc=[-4.0, 1.0, 1.0, 1.0], shrout=[1.0, 1.0, 3.0, 3.0])
    kept = TopNByMarketCap(n=1, price_column="prc", shares_column="shrout").apply(panel)
    assert list(kept.index) == [("A", _D1), ("B", _D2)]


def test_rows_with_a_missing_market_cap_are_dropped() -> None:
    panel = _entity_major_panel()
    panel.loc[("A", _D1), "cap"] = float("nan")
    kept = TopNByMarketCap(n=2, market_cap_column="cap").apply(panel)
    assert list(kept.index) == [("A", _D2), ("B", _D1), ("B", _D2)]


def test_top_n_is_a_cross_section_filter() -> None:
    assert TopNByMarketCap(n=1, market_cap_column="cap").phase is FilterPhase.CROSS_SECTION


def test_non_positive_n_is_rejected() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        TopNByMarketCap(n=0, market_cap_column="cap")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"market_cap_column": "cap", "price_column": "prc", "shares_column": "shrout"},
        {"price_column": "prc"},
    ],
)
def test_exactly_one_market_cap_definition_is_required(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TopNByMarketCap(n=1, **kwargs)


def test_missing_market_cap_column_raises() -> None:
    with pytest.raises(IngestionError, match="absent from the panel"):
        TopNByMarketCap(n=1, market_cap_column="nope").apply(_entity_major_panel())


def test_build_filter_constructs_the_registered_type() -> None:
    built = build_filter("top_n_market_cap", {"n": 3, "market_cap_column": "cap"})
    assert built == TopNByMarketCap(n=3, market_cap_column="cap")
    assert "top_n_market_cap" in registered_filters()


def test_build_filter_rejects_an_unknown_type() -> None:
    with pytest.raises(IngestionError, match="unknown filter type"):
        build_filter("no_such_filter", {})


def test_build_filter_wraps_constructor_errors() -> None:
    with pytest.raises(IngestionError, match="cannot build filter"):
        build_filter("top_n_market_cap", {"n": 0, "market_cap_column": "cap"})


def test_register_custom_filter() -> None:
    sentinel = TopNByMarketCap(n=7, market_cap_column="cap")
    register_filter("test_dummy_filter", lambda params: sentinel)
    assert build_filter("test_dummy_filter", {}) is sentinel
    assert "test_dummy_filter" in registered_filters()


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_top_n_handles_a_nullable_market_cap_column(dtype: str) -> None:
    """A nullable dtype makes the rank comparison nullable too.

    ``(rank <= n)`` is then pandas' ``boolean`` dtype, whose ``to_numpy()`` is an
    object array holding ``pd.NA`` — which ``.loc`` refuses outright. The filter
    must give the same answer for either dtype, treating a missing market cap as
    "not among the largest".
    """
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A", "B", "C"]],
        names=["date", "entity"],
    )
    panel = pd.DataFrame({"cap": pd.array([10, 20, None, 5, None, 30], dtype=dtype)}, index=index)

    kept = TopNByMarketCap(n=1, market_cap_column="cap").apply(panel)

    assert list(kept.index) == [
        (pd.Timestamp("2020-01-01"), "B"),
        (pd.Timestamp("2020-01-02"), "C"),
    ]
