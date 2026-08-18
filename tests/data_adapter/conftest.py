"""Fixtures building a small multi-source dataset on disk.

The sources deliberately cover all four grains, because the point of the adapter
is that one pipeline reads them together: a ``(date, entity)`` price panel, a
date-keyed factor series, an entity-keyed static frame and a keyless link table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from nekron.data.config import (
    CsvSourceConfig,
    IngestionConfig,
    PanelSpec,
    SchemaConfig,
    SelectionConfig,
    SourceConfig,
)

# Two entities over four business days.
_PRICES = """PERMNO,DlyCalDt,DlyClose,DlyVol
10001,02/01/2020,10.0,100
10001,03/01/2020,11.0,110
10001,06/01/2020,12.0,120
10001,07/01/2020,13.0,130
10002,02/01/2020,20.0,200
10002,03/01/2020,21.0,210
10002,06/01/2020,22.0,220
10002,07/01/2020,23.0,230
"""

# One row per date: a market factor, no entity key.
_FACTORS = """date,mktrf,rf
02/01/2020,0.01,0.0001
03/01/2020,-0.02,0.0001
06/01/2020,0.03,0.0001
07/01/2020,0.00,0.0001
"""

# One row per entity: static reference data, no date key.
_SECTORS = """PERMNO,sector
10001,Tech
10002,Energy
"""

# Quarterly fundamentals keyed by a foreign identifier, with a report date.
_FUNDAMENTALS = """gvkey,datadate,rdq,assets
G1,31/12/2019,03/01/2020,500.0
G1,31/03/2020,06/04/2020,550.0
G2,31/12/2019,06/01/2020,900.0
"""

# Keyless link table: gvkey -> permno over a validity window.
_LINK = """gvkey,lpermno,linkdt,linkenddt
G1,10001,01/01/2000,31/12/2099
G2,10002,01/01/2000,31/12/2099
"""


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Path]:
    """Write every source CSV and return its path by panel name."""
    return {
        "prices": _write(tmp_path, "prices.csv", _PRICES),
        "factors": _write(tmp_path, "factors.csv", _FACTORS),
        "sectors": _write(tmp_path, "sectors.csv", _SECTORS),
        "fundamentals": _write(tmp_path, "fundamentals.csv", _FUNDAMENTALS),
        "link": _write(tmp_path, "link.csv", _LINK),
    }


def _spec(path: Path, schema: SchemaConfig, columns: list[str] | None) -> PanelSpec:
    return PanelSpec(
        schema=schema,
        selection=SelectionConfig(columns=columns),
        source=SourceConfig(format="csv", csv=CsvSourceConfig(path=str(path))),
    )


@pytest.fixture
def panel_specs(sources: dict[str, Path]) -> dict[str, PanelSpec]:
    """A ``PanelSpec`` per source, one of each grain."""
    return {
        "prices": _spec(
            sources["prices"],
            SchemaConfig(
                date_column="DlyCalDt",
                entity_column="PERMNO",
                date_format="%d/%m/%Y",
                dtypes={"PERMNO": "int32", "DlyClose": "float64", "DlyVol": "float64"},
            ),
            ["DlyClose", "DlyVol"],
        ),
        "factors": _spec(
            sources["factors"],
            SchemaConfig(
                date_column="date",
                entity_column=None,
                date_format="%d/%m/%Y",
                dtypes={"mktrf": "float64", "rf": "float64"},
            ),
            ["mktrf", "rf"],
        ),
        "sectors": _spec(
            sources["sectors"],
            SchemaConfig(
                date_column=None,
                entity_column="PERMNO",
                date_format="%d/%m/%Y",
                dtypes={"PERMNO": "int32"},
                category_columns=["sector"],
            ),
            ["sector"],
        ),
        "fundamentals": _spec(
            sources["fundamentals"],
            SchemaConfig(
                date_column="datadate",
                entity_column="gvkey",
                date_format="%d/%m/%Y",
                date_columns=["rdq"],
                dtypes={"gvkey": "string", "assets": "float64"},
            ),
            ["assets", "rdq"],
        ),
        "link": _spec(
            sources["link"],
            SchemaConfig(
                date_column=None,
                entity_column=None,
                date_format="%d/%m/%Y",
                date_columns=["linkdt", "linkenddt"],
                dtypes={"gvkey": "string", "lpermno": "int32"},
            ),
            ["gvkey", "lpermno", "linkdt", "linkenddt"],
        ),
    }


@pytest.fixture
def ingestion(panel_specs: dict[str, PanelSpec]) -> Callable[..., IngestionConfig]:
    """Build an ``IngestionConfig`` over a chosen subset of the sources."""

    def _build(*names: str, **overrides: object) -> IngestionConfig:
        chosen = {name: panel_specs[name] for name in (names or tuple(panel_specs))}
        for name, spec in list(chosen.items()):
            if name in overrides:
                chosen[name] = replace(spec, **overrides[name])  # type: ignore[arg-type]
        return IngestionConfig(panels=chosen)

    return _build
