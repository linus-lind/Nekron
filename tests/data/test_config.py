"""Tests for config-to-object conversion."""

from __future__ import annotations

import pandas as pd
import pytest
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from nekron.data import to_config, to_schema, to_selection
from nekron.data.config import (
    CONFIG_SCHEMA_NAME,
    PANEL_SCHEMA_NAME,
    CsvSourceConfig,
    IngestionConfig,
    PanelSpec,
    SchemaConfig,
    SelectionConfig,
    SourceConfig,
    register_configs,
)
from nekron.panel import PanelGrain


def test_to_schema_maps_and_freezes_collections() -> None:
    schema = to_schema(
        SchemaConfig(
            date_column="D",
            entity_column="E",
            date_format="%Y",
            dtypes={"E": "int32"},
            category_columns=["c1", "c2"],
            date_columns=["d2"],
        )
    )
    assert schema.date_column == "D"
    assert schema.category_columns == ("c1", "c2")
    assert schema.date_columns == ("d2",)
    assert schema.parsed_date_columns == ("D", "d2")


def test_to_selection_typed_values() -> None:
    selection = to_selection(
        SelectionConfig(start_date="2010-01-01", end_date=None, entities=[1, 2], columns=["x"])
    )
    assert selection.date_range.start == pd.Timestamp("2010-01-01")
    assert selection.date_range.end is None
    assert selection.entities == frozenset({1, 2})
    assert selection.columns == ("x",)


def test_to_selection_defaults_are_unrestricted() -> None:
    selection = to_selection(SelectionConfig())
    assert selection.entities is None
    assert selection.columns is None
    assert selection.date_range.is_open()


@pytest.mark.parametrize(
    ("date_column", "entity_column", "grain"),
    [
        ("D", "E", PanelGrain.PANEL),
        ("D", None, PanelGrain.TIME_SERIES),
        (None, "E", PanelGrain.CROSS_SECTION),
        (None, None, PanelGrain.TABLE),
    ],
)
def test_to_schema_carries_the_grain_through(
    date_column: str | None, entity_column: str | None, grain: PanelGrain
) -> None:
    schema = to_schema(
        SchemaConfig(date_column=date_column, entity_column=entity_column, date_format="%Y")
    )
    assert schema.grain is grain


def test_key_columns_are_mandatory_but_nullable() -> None:
    # Omitting a key column must fail loudly: leaving it out silently reshapes the
    # source into a different grain, so a keyless source has to say `null`.
    cfg = OmegaConf.structured(SchemaConfig)
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_object(cfg)

    cfg.date_column = None
    cfg.entity_column = None
    cfg.date_format = "%Y"
    resolved = OmegaConf.to_object(cfg)
    assert isinstance(resolved, SchemaConfig)
    assert to_schema(resolved).grain is PanelGrain.TABLE


def test_to_config_materializes_named_panels() -> None:
    cfg = OmegaConf.structured(
        IngestionConfig(
            panels={
                "crsp": PanelSpec(
                    schema=SchemaConfig(
                        date_column="DlyCalDt", entity_column="PERMNO", date_format="%d/%m/%Y"
                    ),
                    source=SourceConfig(csv=CsvSourceConfig(path="crsp.csv")),
                ),
                "factors": PanelSpec(
                    schema=SchemaConfig(
                        date_column="CalDt", entity_column=None, date_format="%d/%m/%Y"
                    ),
                    source=SourceConfig(csv=CsvSourceConfig(path="ff.csv")),
                ),
            }
        )
    )
    resolved = to_config(cfg)

    assert isinstance(resolved, IngestionConfig)
    assert list(resolved.panels) == ["crsp", "factors"]
    assert to_schema(resolved.panels["crsp"].schema).grain is PanelGrain.PANEL
    assert to_schema(resolved.panels["factors"].schema).grain is PanelGrain.TIME_SERIES
    assert resolved.panels["factors"].source.csv.path == "ff.csv"


def test_to_config_reports_an_unresolved_value() -> None:
    cfg = OmegaConf.structured(IngestionConfig(panels={"crsp": PanelSpec()}))
    with pytest.raises(MissingMandatoryValue):
        to_config(cfg)


def test_register_configs_stores_both_nodes() -> None:
    register_configs()
    repo = ConfigStore.instance().repo
    assert f"{CONFIG_SCHEMA_NAME}.yaml" in repo
    assert f"{PANEL_SCHEMA_NAME}.yaml" in repo
