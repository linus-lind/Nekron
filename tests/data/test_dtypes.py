"""Tests for dtype resolution helpers."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from nekron.data import (
    IngestionError,
    PanelSchema,
    apply_categoricals,
    build_read_dtypes,
    parse_dates,
)


def test_build_read_dtypes_reads_date_and_category_as_text(
    make_schema: Callable[..., PanelSchema],
) -> None:
    schema = make_schema()
    dtypes = build_read_dtypes(["PERMNO", "DlyCalDt", "DlyPrevDt", "DlyRet", "PrimaryExch"], schema)
    assert dtypes["DlyCalDt"] == "object"
    assert dtypes["DlyPrevDt"] == "object"
    assert dtypes["PrimaryExch"] == "object"
    assert dtypes["PERMNO"] == "int32"
    assert dtypes["DlyRet"] == "float64"


def test_build_read_dtypes_rejects_unresolved_columns(
    make_schema: Callable[..., PanelSchema],
) -> None:
    with pytest.raises(IngestionError):
        build_read_dtypes(["Unlisted"], make_schema())


def test_parse_dates_converts_and_marks_missing(
    make_schema: Callable[..., PanelSchema],
) -> None:
    frame = pd.DataFrame({"DlyCalDt": ["02/01/2020", None], "DlyPrevDt": ["31/12/2019", None]})
    parse_dates(frame, make_schema())
    assert is_datetime64_any_dtype(frame["DlyCalDt"])
    assert frame["DlyCalDt"].iloc[0] == pd.Timestamp("2020-01-02")
    assert pd.isna(frame["DlyCalDt"].iloc[1])


def test_parse_dates_rejects_wrong_format(make_schema: Callable[..., PanelSchema]) -> None:
    frame = pd.DataFrame({"DlyCalDt": ["2020-01-02"]})
    with pytest.raises(ValueError):
        parse_dates(frame, make_schema())


def test_apply_categoricals(make_schema: Callable[..., PanelSchema]) -> None:
    frame = pd.DataFrame({"PrimaryExch": ["N", "Q", "N"]})
    apply_categoricals(frame, make_schema())
    assert isinstance(frame["PrimaryExch"].dtype, pd.CategoricalDtype)


def test_build_read_dtypes_resolves_a_keyless_table(link_schema: PanelSchema) -> None:
    # A link table has no key columns, so every column it declares must resolve
    # from dtypes or date_columns alone.
    dtypes = build_read_dtypes(["GVKEY", "PERMNO", "LinkDt", "LinkEndDt"], link_schema)
    assert dtypes == {
        "GVKEY": "int32",
        "PERMNO": "int32",
        "LinkDt": "object",
        "LinkEndDt": "object",
    }


def test_build_read_dtypes_resolves_a_date_only_source(factor_schema: PanelSchema) -> None:
    dtypes = build_read_dtypes(["CalDt", "MktRf", "SMB"], factor_schema)
    assert dtypes == {"CalDt": "object", "MktRf": "float64", "SMB": "float64"}


def test_build_read_dtypes_resolves_an_entity_only_source(static_schema: PanelSchema) -> None:
    dtypes = build_read_dtypes(["PERMNO", "Sector", "SIC"], static_schema)
    assert dtypes == {"PERMNO": "int32", "Sector": "object", "SIC": "int32"}


def test_parse_dates_on_a_keyless_table_parses_the_validity_window(
    link_schema: PanelSchema,
) -> None:
    frame = pd.DataFrame(
        {"LinkDt": ["01/01/2015", "01/01/2016"], "LinkEndDt": ["31/12/2099", None]}
    )
    parse_dates(frame, link_schema)
    assert is_datetime64_any_dtype(frame["LinkDt"])
    assert frame["LinkEndDt"].iloc[0] == pd.Timestamp("2099-12-31")
    assert pd.isna(frame["LinkEndDt"].iloc[1])
