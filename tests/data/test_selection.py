"""Tests for :class:`DateRange` and :class:`PanelSelection`."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from nekron.data import DateRange, IngestionError, PanelSchema, PanelSelection


def _chunk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DlyCalDt": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
            "PERMNO": [10001, 10002, 10003],
            "DlyRet": [0.1, 0.2, 0.3],
        }
    )


def test_from_iso_maps_empty_to_open_bound() -> None:
    rng = DateRange.from_iso("2010-01-01", None)
    assert rng.start == pd.Timestamp("2010-01-01")
    assert rng.end is None
    assert not rng.is_open()
    assert DateRange.from_iso(None, None).is_open()


def test_date_lower_bound_is_inclusive(make_schema: Callable[..., PanelSchema]) -> None:
    selection = PanelSelection(DateRange(pd.Timestamp("2020-01-03"), None), None, None)
    result = selection.filter_rows(_chunk(), make_schema())
    assert result["PERMNO"].tolist() == [10002, 10003]


def test_date_upper_bound_is_inclusive(make_schema: Callable[..., PanelSchema]) -> None:
    selection = PanelSelection(DateRange(None, pd.Timestamp("2020-01-03")), None, None)
    result = selection.filter_rows(_chunk(), make_schema())
    assert result["PERMNO"].tolist() == [10001, 10002]


def test_entity_filter(make_schema: Callable[..., PanelSchema]) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({10001, 10003}), None)
    result = selection.filter_rows(_chunk(), make_schema())
    assert result["PERMNO"].tolist() == [10001, 10003]


def test_combined_date_and_entity(make_schema: Callable[..., PanelSchema]) -> None:
    selection = PanelSelection(
        DateRange(pd.Timestamp("2020-01-03"), None), frozenset({10002}), None
    )
    result = selection.filter_rows(_chunk(), make_schema())
    assert result["PERMNO"].tolist() == [10002]


def test_no_filter_returns_same_object(make_schema: Callable[..., PanelSchema]) -> None:
    chunk = _chunk()
    selection = PanelSelection(DateRange(None, None), None, None)
    assert selection.filter_rows(chunk, make_schema()) is chunk


def test_date_range_without_a_date_column_raises(static_schema: PanelSchema) -> None:
    # Silently ignoring the bound would load the whole file and look like success.
    selection = PanelSelection(DateRange(pd.Timestamp("2020-01-03"), None), None, None)
    with pytest.raises(IngestionError, match="no date_column"):
        selection.filter_rows(_chunk(), static_schema)


def test_entity_filter_without_an_entity_column_raises(factor_schema: PanelSchema) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({10001}), None)
    with pytest.raises(IngestionError, match="no entity_column"):
        selection.filter_rows(_chunk(), factor_schema)


def test_open_selection_on_a_keyless_schema_is_a_pass_through(link_schema: PanelSchema) -> None:
    chunk = _chunk()
    selection = PanelSelection(DateRange(None, None), None, None)
    assert selection.filter_rows(chunk, link_schema) is chunk
