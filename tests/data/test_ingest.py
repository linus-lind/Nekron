"""Tests for the :func:`load_panel` / :func:`load_panels` orchestration entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from nekron.data import IngestionError, load_panel, load_panels
from nekron.data.config import (
    CsvSourceConfig,
    FilterSpec,
    IngestionConfig,
    PanelSpec,
    SchemaConfig,
    SelectionConfig,
    SourceConfig,
)
from nekron.panel import Panel, PanelError, PanelGrain, PanelSet


def _spec(path: Path, schema: SchemaConfig, **selection: Any) -> PanelSpec:
    """A CSV-backed spec for ``path``, with an otherwise unrestricted selection."""
    return PanelSpec(
        schema=schema,
        selection=SelectionConfig(**selection),
        source=SourceConfig(format="csv", csv=CsvSourceConfig(path=str(path))),
    )


def _panel_schema() -> SchemaConfig:
    return SchemaConfig(
        date_column="DlyCalDt",
        entity_column="PERMNO",
        date_format="%d/%m/%Y",
        dtypes={"PERMNO": "int32", "DlyRet": "float64", "DlyPrc": "float64"},
        category_columns=["PrimaryExch"],
        date_columns=["DlyPrevDt"],
    )


def _factor_schema() -> SchemaConfig:
    return SchemaConfig(
        date_column="CalDt",
        entity_column=None,
        date_format="%d/%m/%Y",
        dtypes={"MktRf": "float64", "SMB": "float64"},
    )


def _static_schema() -> SchemaConfig:
    return SchemaConfig(
        date_column=None,
        entity_column="PERMNO",
        date_format="%d/%m/%Y",
        dtypes={"PERMNO": "int32", "SIC": "int32"},
        category_columns=["Sector"],
    )


def _link_schema() -> SchemaConfig:
    return SchemaConfig(
        date_column=None,
        entity_column=None,
        date_format="%d/%m/%Y",
        dtypes={"GVKEY": "int32", "PERMNO": "int32"},
        date_columns=["LinkDt", "LinkEndDt"],
    )


def test_load_panel_end_to_end(sample_csv: Path) -> None:
    spec = _spec(sample_csv, _panel_schema(), entities=[10001, 10002], columns=["DlyRet"])
    panel = load_panel(spec)

    assert isinstance(panel, Panel)
    assert panel.grain is PanelGrain.PANEL
    frame = panel.frame
    assert list(frame.index.names) == ["date", "entity"]
    assert list(frame.columns) == ["DlyRet"]
    assert set(frame.index.get_level_values("entity").unique()) == {10001, 10002}
    assert len(frame) == 4
    assert is_datetime64_any_dtype(frame.index.get_level_values("date"))
    assert str(frame.index.get_level_values("entity").dtype) == "int32"


def test_load_panel_declares_both_key_names(sample_csv: Path) -> None:
    panel = load_panel(_spec(sample_csv, _panel_schema()))
    assert panel.date_name == "date"
    assert panel.entity_name == "entity"
    assert panel.key_names == ("date", "entity")
    panel.validate()


def test_load_panel_honours_renamed_levels(sample_csv: Path) -> None:
    schema = _panel_schema()
    schema.date_name = "as_of"
    schema.entity_name = "permno"
    panel = load_panel(_spec(sample_csv, schema))
    assert panel.key_names == ("as_of", "permno")
    assert list(panel.frame.index.names) == ["as_of", "permno"]
    panel.validate()


def test_load_time_series_panel(factor_csv: Path) -> None:
    panel = load_panel(_spec(factor_csv, _factor_schema()))
    panel.validate()

    assert panel.grain is PanelGrain.TIME_SERIES
    assert panel.date_name == "date"
    assert panel.entity_name is None
    index = panel.frame.index
    assert list(index.names) == ["date"]
    assert is_datetime64_any_dtype(index)
    # The source is not in date order; a keyed source is sorted on load.
    assert index.is_monotonic_increasing
    assert list(panel.frame.columns) == ["MktRf", "SMB"]
    assert panel.frame.loc[pd.Timestamp("2020-01-02"), "MktRf"] == pytest.approx(0.010)


def test_load_cross_section_panel(static_csv: Path) -> None:
    panel = load_panel(_spec(static_csv, _static_schema()))
    panel.validate()

    assert panel.grain is PanelGrain.CROSS_SECTION
    assert panel.date_name is None
    assert panel.entity_name == "entity"
    index = panel.frame.index
    assert list(index.names) == ["entity"]
    assert str(index.dtype) == "int32"
    assert list(index) == [10001, 10002, 10003]
    assert isinstance(panel.frame["Sector"].dtype, pd.CategoricalDtype)


def test_load_table_panel_keeps_positional_index(link_csv: Path) -> None:
    panel = load_panel(_spec(link_csv, _link_schema()))
    panel.validate()

    assert panel.grain is PanelGrain.TABLE
    assert panel.key_names == ()
    index = panel.frame.index
    assert list(index.names) == [None]
    assert list(index) == [0, 1, 2]
    # Rows keep source order: a keyless table has nothing to sort by.
    assert list(panel.frame["GVKEY"]) == [1001, 1002, 1003]
    # A validity window is made of date columns, not date keys.
    assert is_datetime64_any_dtype(panel.frame["LinkDt"])
    assert is_datetime64_any_dtype(panel.frame["LinkEndDt"])
    assert panel.frame["LinkEndDt"].iloc[1] == pd.Timestamp("2020-12-31")


def test_load_panels_returns_named_set_of_mixed_grains(
    sample_csv: Path, factor_csv: Path, static_csv: Path, link_csv: Path
) -> None:
    cfg = IngestionConfig(
        panels={
            "crsp": _spec(sample_csv, _panel_schema()),
            "factors": _spec(factor_csv, _factor_schema()),
            "sectors": _spec(static_csv, _static_schema()),
            "ccm_link": _spec(link_csv, _link_schema()),
        }
    )
    panels = load_panels(cfg)

    assert isinstance(panels, PanelSet)
    assert set(panels) == {"crsp", "factors", "sectors", "ccm_link"}
    assert panels["crsp"].grain is PanelGrain.PANEL
    assert panels["factors"].grain is PanelGrain.TIME_SERIES
    assert panels["sectors"].grain is PanelGrain.CROSS_SECTION
    assert panels["ccm_link"].grain is PanelGrain.TABLE
    assert set(panels.of_grain(PanelGrain.PANEL, PanelGrain.TIME_SERIES)) == {"crsp", "factors"}
    assert len(panels["crsp"].frame) == 5


def test_load_panels_preserves_composition_order(sample_csv: Path, factor_csv: Path) -> None:
    cfg = IngestionConfig(
        panels={
            "factors": _spec(factor_csv, _factor_schema()),
            "crsp": _spec(sample_csv, _panel_schema()),
        }
    )
    assert list(load_panels(cfg)) == ["factors", "crsp"]


def test_load_panels_of_empty_config_is_empty() -> None:
    panels = load_panels(IngestionConfig())
    assert len(panels) == 0
    assert dict(panels) == {}


def test_unknown_panel_name_names_the_loaded_ones(sample_csv: Path) -> None:
    panels = load_panels(IngestionConfig(panels={"crsp": _spec(sample_csv, _panel_schema())}))
    with pytest.raises(PanelError, match="crsp"):
        panels["compustat"]


def test_cross_section_filter_runs_on_the_assembled_panel(sample_csv: Path) -> None:
    spec = _spec(sample_csv, _panel_schema(), columns=["DlyPrc"])
    spec.filters = [
        FilterSpec(type="top_n_market_cap", params={"n": 1, "market_cap_column": "DlyPrc"})
    ]
    panel = load_panel(spec)

    # Largest |price| per date is 10002 on both dates.
    assert set(panel.frame.index.get_level_values("entity")) == {10002}
    assert len(panel.frame) == 2


def test_date_range_on_a_source_without_a_date_key_raises(static_csv: Path) -> None:
    spec = _spec(static_csv, _static_schema(), start_date="2020-01-01")
    with pytest.raises(IngestionError, match="no date_column"):
        load_panel(spec)


def test_entities_on_a_source_without_an_entity_key_raises(factor_csv: Path) -> None:
    spec = _spec(factor_csv, _factor_schema(), entities=[10001])
    with pytest.raises(IngestionError, match="no entity_column"):
        load_panel(spec)
