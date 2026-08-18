"""End-to-end tests for :class:`CsvPanelSource`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal

from nekron.data import (
    DATE_UNIT,
    CsvPanelSource,
    DateRange,
    IngestionError,
    MissingColumnsError,
    PanelSchema,
    PanelSelection,
)

MakeSchema = Callable[..., PanelSchema]
MakeSource = Callable[..., CsvPanelSource]


def test_basic_load_shape_index_and_dtypes(
    make_schema: MakeSchema, make_source: MakeSource, open_selection: PanelSelection
) -> None:
    panel = make_source(make_schema(), open_selection).load()

    assert list(panel.index.names) == ["date", "entity"]
    assert len(panel) == 5
    # Ingestion pins every parsed date to DATE_UNIT so the dtype does not depend on
    # the pandas version (2.2 defaults to ns, 3.0 infers from the data).
    assert str(panel.index.get_level_values("date").dtype) == f"datetime64[{DATE_UNIT}]"
    assert str(panel.index.get_level_values("entity").dtype) == "int32"
    assert str(panel["DlyRet"].dtype) == "float64"
    assert isinstance(panel["PrimaryExch"].dtype, pd.CategoricalDtype)
    assert is_datetime64_any_dtype(panel["DlyPrevDt"])
    # Missing prev-date for the first row of each entity becomes NaT.
    assert pd.isna(panel.loc[(pd.Timestamp("2020-01-02"), 10001), "DlyPrevDt"])


def test_sorted_by_date_then_entity(
    make_schema: MakeSchema, make_source: MakeSource, open_selection: PanelSelection
) -> None:
    panel = make_source(make_schema(), open_selection).load()
    assert panel.index.is_monotonic_increasing
    assert list(panel.index) == [
        (pd.Timestamp("2020-01-02"), 10001),
        (pd.Timestamp("2020-01-02"), 10002),
        (pd.Timestamp("2020-01-02"), 10003),
        (pd.Timestamp("2020-01-03"), 10001),
        (pd.Timestamp("2020-01-03"), 10002),
    ]


def test_sort_false_preserves_source_order(
    make_schema: MakeSchema, make_source: MakeSource, open_selection: PanelSelection
) -> None:
    panel = make_source(make_schema(), open_selection, sort=False).load()
    # Source is entity-major, so the panel is not (date, entity) monotonic.
    assert not panel.index.is_monotonic_increasing
    assert panel.index[0] == (pd.Timestamp("2020-01-02"), 10001)
    assert panel.index[1] == (pd.Timestamp("2020-01-03"), 10001)


def test_date_format_controls_parsing(tmp_path: Path, make_schema: MakeSchema) -> None:
    path = tmp_path / "amb.csv"
    path.write_text("PERMNO,DlyCalDt,DlyRet\n10001,02/03/2020,0.1\n", encoding="utf-8")
    schema_kwargs = {
        "category_columns": (),
        "date_columns": (),
        "dtypes": {"PERMNO": "int32", "DlyRet": "float64"},
    }

    day_first = CsvPanelSource(
        path=str(path),
        schema=make_schema(date_format="%d/%m/%Y", **schema_kwargs),
        selection=PanelSelection(DateRange(None, None), None, None),
        delimiter=",",
        encoding="utf-8",
        decimal=".",
        chunk_size=10,
        na_values=(),
        keep_default_na=True,
        sort=True,
    ).load()
    month_first = CsvPanelSource(
        path=str(path),
        schema=make_schema(date_format="%m/%d/%Y", **schema_kwargs),
        selection=PanelSelection(DateRange(None, None), None, None),
        delimiter=",",
        encoding="utf-8",
        decimal=".",
        chunk_size=10,
        na_values=(),
        keep_default_na=True,
        sort=True,
    ).load()

    assert day_first.index.get_level_values("date")[0] == pd.Timestamp("2020-03-02")
    assert month_first.index.get_level_values("date")[0] == pd.Timestamp("2020-02-03")


def test_column_selection_prunes_columns(make_schema: MakeSchema, make_source: MakeSource) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("DlyRet",))
    panel = make_source(make_schema(), selection).load()
    assert list(panel.columns) == ["DlyRet"]
    assert list(panel.index.names) == ["date", "entity"]


def test_entity_selection(make_schema: MakeSchema, make_source: MakeSource) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({10002}), None)
    panel = make_source(make_schema(), selection).load()
    assert set(panel.index.get_level_values("entity").unique()) == {10002}


def test_date_range_selection_inclusive(make_schema: MakeSchema, make_source: MakeSource) -> None:
    selection = PanelSelection(
        DateRange(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-03")), None, None
    )
    panel = make_source(make_schema(), selection).load()
    assert set(panel.index.get_level_values("date").unique()) == {pd.Timestamp("2020-01-03")}


def test_chunk_boundaries_do_not_change_result(
    make_schema: MakeSchema, make_source: MakeSource, open_selection: PanelSelection
) -> None:
    single = make_source(make_schema(), open_selection, chunk_size=1).load()
    whole = make_source(make_schema(), open_selection, chunk_size=10_000).load()
    assert_frame_equal(single, whole)


def test_empty_result_is_typed_empty_panel(
    make_schema: MakeSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({999999}), None)
    panel = make_source(make_schema(), selection).load()
    assert len(panel) == 0
    assert list(panel.index.names) == ["date", "entity"]
    assert "DlyRet" in panel.columns
    assert str(panel["DlyRet"].dtype) == "float64"
    # Types are preserved on the empty frame, matching a non-empty read.
    assert str(panel.index.get_level_values("entity").dtype) == "int32"
    assert isinstance(panel["PrimaryExch"].dtype, pd.CategoricalDtype)


def test_duplicate_date_entity_rows_are_preserved(tmp_path: Path, make_schema: MakeSchema) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text(
        "PERMNO,DlyCalDt,DlyPrevDt,DlyRet,DlyPrc,PrimaryExch\n"
        "10001,02/01/2020,,0.01,10.5,N\n"
        "10001,02/01/2020,,0.02,10.6,N\n",
        encoding="utf-8",
    )
    panel = CsvPanelSource(
        path=str(path),
        schema=make_schema(),
        selection=PanelSelection(DateRange(None, None), None, None),
        delimiter=",",
        encoding="utf-8",
        decimal=".",
        chunk_size=10,
        na_values=(),
        keep_default_na=True,
        sort=True,
    ).load()
    assert len(panel) == 2
    assert panel.index.duplicated().any()


def test_missing_requested_column_raises(make_schema: MakeSchema, make_source: MakeSource) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("DlyRet", "DoesNotExist"))
    with pytest.raises(MissingColumnsError):
        make_source(make_schema(), selection).load()


def test_missing_schema_column_raises(
    make_schema: MakeSchema, make_source: MakeSource, open_selection: PanelSelection
) -> None:
    with pytest.raises(MissingColumnsError):
        make_source(make_schema(date_column="NotAColumn"), open_selection).load()


def test_entities_are_cast_to_entity_dtype(
    make_schema: MakeSchema, make_source: MakeSource
) -> None:
    # String entity ids compare equal to the int32 PERMNO column after casting.
    selection = PanelSelection(DateRange(None, None), frozenset({"10002"}), None)
    panel = make_source(make_schema(), selection).load()
    assert set(panel.index.get_level_values("entity").unique()) == {10002}


def test_incompatible_entities_raise(make_schema: MakeSchema, make_source: MakeSource) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({"NOT_AN_INT"}), None)
    with pytest.raises(IngestionError):
        make_source(make_schema(), selection).load()


def test_requested_column_order_is_preserved(
    make_schema: MakeSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("DlyPrc", "DlyRet"))
    panel = make_source(make_schema(), selection).load()
    assert list(panel.columns) == ["DlyPrc", "DlyRet"]


def test_auxiliary_date_parsed_after_filtering(
    make_schema: MakeSchema, make_source: MakeSource
) -> None:
    # DlyPrevDt is parsed on retained rows (after entity filtering), across chunks.
    selection = PanelSelection(DateRange(None, None), frozenset({10001}), None)
    panel = make_source(make_schema(), selection, chunk_size=1).load()
    assert is_datetime64_any_dtype(panel["DlyPrevDt"])
    assert panel.loc[(pd.Timestamp("2020-01-03"), 10001), "DlyPrevDt"] == pd.Timestamp("2020-01-02")


def test_empty_file_raises(tmp_path: Path, make_schema: MakeSchema) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(IngestionError):
        CsvPanelSource(
            path=str(path),
            schema=make_schema(),
            selection=PanelSelection(DateRange(None, None), None, None),
            delimiter=",",
            encoding="utf-8",
            decimal=".",
            chunk_size=10,
            na_values=(),
            keep_default_na=True,
            sort=True,
        ).load()


def test_time_series_source_is_keyed_by_date_alone(
    factor_csv: Path,
    factor_schema: PanelSchema,
    make_source: MakeSource,
    open_selection: PanelSelection,
) -> None:
    panel = make_source(factor_schema, open_selection, path=factor_csv).load()
    assert list(panel.index.names) == ["date"]
    assert not isinstance(panel.index, pd.MultiIndex)
    assert is_datetime64_any_dtype(panel.index)
    assert panel.index.is_monotonic_increasing
    assert list(panel.columns) == ["MktRf", "SMB"]


def test_cross_section_source_is_keyed_by_entity_alone(
    static_csv: Path,
    static_schema: PanelSchema,
    make_source: MakeSource,
    open_selection: PanelSelection,
) -> None:
    panel = make_source(static_schema, open_selection, path=static_csv).load()
    assert list(panel.index.names) == ["entity"]
    assert not isinstance(panel.index, pd.MultiIndex)
    assert str(panel.index.dtype) == "int32"
    assert list(panel.index) == [10001, 10002, 10003]
    assert isinstance(panel["Sector"].dtype, pd.CategoricalDtype)


def test_table_source_keeps_a_positional_index(
    link_csv: Path,
    link_schema: PanelSchema,
    make_source: MakeSource,
    open_selection: PanelSelection,
) -> None:
    panel = make_source(link_schema, open_selection, path=link_csv).load()
    assert list(panel.index.names) == [None]
    assert list(panel.index) == [0, 1, 2]
    assert list(panel.columns) == ["GVKEY", "PERMNO", "LinkDt", "LinkEndDt"]
    # sort=True is a no-op without keys, so source order survives.
    assert list(panel["GVKEY"]) == [1001, 1002, 1003]


def test_usecols_keeps_the_panel_key_columns(
    make_schema: MakeSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("DlyRet",))
    panel = make_source(make_schema(), selection).load()
    assert list(panel.columns) == ["DlyRet"]
    assert list(panel.index.names) == ["date", "entity"]
    assert len(panel) == 5


def test_usecols_keeps_the_date_key_of_a_time_series(
    factor_csv: Path, factor_schema: PanelSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("MktRf",))
    panel = make_source(factor_schema, selection, path=factor_csv).load()
    assert list(panel.columns) == ["MktRf"]
    assert list(panel.index.names) == ["date"]
    assert len(panel) == 3


def test_usecols_keeps_the_entity_key_of_a_cross_section(
    static_csv: Path, static_schema: PanelSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), None, ("Sector",))
    panel = make_source(static_schema, selection, path=static_csv).load()
    assert list(panel.columns) == ["Sector"]
    assert list(panel.index.names) == ["entity"]
    assert list(panel.index) == [10001, 10002, 10003]


def test_column_selected_table_returns_requested_columns_positionally(
    link_csv: Path, link_schema: PanelSchema, make_source: MakeSource
) -> None:
    # A keyless source has no key columns to add back, so usecols is exactly the
    # requested set and the frame keeps its positional index.
    selection = PanelSelection(DateRange(None, None), None, ("PERMNO", "LinkEndDt"))
    panel = make_source(link_schema, selection, path=link_csv).load()
    assert list(panel.columns) == ["PERMNO", "LinkEndDt"]
    assert list(panel.index.names) == [None]
    assert list(panel.index) == [0, 1, 2]
    assert is_datetime64_any_dtype(panel["LinkEndDt"])
    assert panel["LinkEndDt"].iloc[1] == pd.Timestamp("2020-12-31")


def test_time_series_date_range_selection(
    factor_csv: Path, factor_schema: PanelSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(
        DateRange(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")), None, None
    )
    panel = make_source(factor_schema, selection, path=factor_csv).load()
    assert list(panel.index) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]


def test_cross_section_entity_selection(
    static_csv: Path, static_schema: PanelSchema, make_source: MakeSource
) -> None:
    selection = PanelSelection(DateRange(None, None), frozenset({10003}), None)
    panel = make_source(static_schema, selection, path=static_csv).load()
    assert list(panel.index) == [10003]
