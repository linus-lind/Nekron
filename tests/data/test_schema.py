"""Tests for how :class:`PanelSchema` derives a grain from its key columns."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nekron.data import PanelSchema
from nekron.panel import PanelGrain


def _schema(date_column: str | None, entity_column: str | None, **overrides: object) -> PanelSchema:
    params: dict[str, object] = {
        "date_column": date_column,
        "entity_column": entity_column,
        "date_format": "%d/%m/%Y",
        "dtypes": {},
        "category_columns": (),
        "date_columns": (),
        "date_name": "date",
        "entity_name": "entity",
    }
    params.update(overrides)
    return PanelSchema(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("date_column", "entity_column", "grain", "key_columns", "index_names"),
    [
        ("D", "E", PanelGrain.PANEL, ("D", "E"), ("date", "entity")),
        ("D", None, PanelGrain.TIME_SERIES, ("D",), ("date",)),
        (None, "E", PanelGrain.CROSS_SECTION, ("E",), ("entity",)),
        (None, None, PanelGrain.TABLE, (), ()),
    ],
)
def test_grain_and_keys_for_every_combination(
    date_column: str | None,
    entity_column: str | None,
    grain: PanelGrain,
    key_columns: tuple[str, ...],
    index_names: tuple[str, ...],
) -> None:
    schema = _schema(date_column, entity_column)
    assert schema.grain is grain
    assert schema.key_columns == key_columns
    assert schema.index_names == index_names
    # The two tuples are what a source zips together to build its index.
    assert len(schema.key_columns) == len(schema.index_names)


def test_index_names_follow_the_configured_level_names() -> None:
    schema = _schema("D", "E", date_name="as_of", entity_name="permno")
    assert schema.index_names == ("as_of", "permno")
    assert schema.key_columns == ("D", "E")


@pytest.mark.parametrize("field_name", ["date_column", "entity_column"])
def test_empty_string_is_not_a_no_key_sentinel(field_name: str) -> None:
    # An empty string is the shape a mistyped YAML value takes; accepting it would
    # silently reshape the source into a different grain.
    keys: dict[str, str | None] = {"date_column": "D", "entity_column": "E"}
    keys[field_name] = ""
    with pytest.raises(ValueError, match=field_name):
        _schema(keys["date_column"], keys["entity_column"])


def test_parsed_date_columns_survive_a_missing_date_key() -> None:
    # A link table has no date key but still parses its validity window.
    schema = _schema(None, None, date_columns=("LinkDt", "LinkEndDt"))
    assert schema.grain is PanelGrain.TABLE
    assert schema.parsed_date_columns == ("LinkDt", "LinkEndDt")


def test_parsed_date_columns_puts_the_index_date_first_without_duplicates() -> None:
    schema = _schema("D", "E", date_columns=("D", "other"))
    assert schema.parsed_date_columns == ("D", "other")


def test_make_schema_fixture_defaults_are_a_panel(make_schema: Callable[..., PanelSchema]) -> None:
    schema = make_schema()
    assert schema.grain is PanelGrain.PANEL
    assert schema.key_columns == ("DlyCalDt", "PERMNO")
    assert schema.index_names == ("date", "entity")
