"""Tests for the source-format registry."""

from __future__ import annotations

import pytest

from nekron.data import (
    CsvPanelSource,
    DateRange,
    IngestionError,
    PanelSchema,
    PanelSelection,
    build_source,
    register_source,
)
from nekron.data.config import CsvSourceConfig, SourceConfig

_SCHEMA = PanelSchema(
    date_column="DlyCalDt",
    entity_column="PERMNO",
    date_format="%d/%m/%Y",
    dtypes={},
    category_columns=(),
    date_columns=(),
    date_name="date",
    entity_name="entity",
)
_SELECTION = PanelSelection(DateRange(None, None), None, None)


def test_build_csv_source() -> None:
    source = build_source(
        SourceConfig(format="csv", csv=CsvSourceConfig(path="x.csv")), _SCHEMA, _SELECTION
    )
    assert isinstance(source, CsvPanelSource)
    assert source.path == "x.csv"


def test_unknown_format_raises() -> None:
    with pytest.raises(IngestionError):
        build_source(SourceConfig(format="parquet"), _SCHEMA, _SELECTION)


def test_register_custom_source() -> None:
    sentinel = object()
    register_source("test_dummy_source", lambda source, schema, selection, row_filters: sentinel)  # type: ignore[arg-type,return-value]
    built = build_source(SourceConfig(format="test_dummy_source"), _SCHEMA, _SELECTION)
    assert built is sentinel
