"""Shared fixtures for data-ingestion tests.

The four grains :class:`~nekron.panel.PanelGrain` distinguishes each get their own
tiny source file here, because "does this source load correctly" is a different
question for a ``(date, entity)`` panel, a date-keyed factor series, an
entity-keyed static frame and a keyless link table, and the interesting failures
live in what happens to the *index* rather than to the values.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nekron.data import CsvPanelSource, DateRange, PanelSchema, PanelSelection

# Entity-major, DD/MM/YYYY dates; 10003 has missing return/price and no prev date.
_SAMPLE_CSV = """PERMNO,DlyCalDt,DlyPrevDt,DlyRet,DlyPrc,PrimaryExch
10001,02/01/2020,,0.01,10.5,N
10001,03/01/2020,02/01/2020,-0.02,10.3,N
10002,02/01/2020,,0.00,20.0,Q
10002,03/01/2020,02/01/2020,0.05,21.0,Q
10003,02/01/2020,,,,A
"""

# One row per date: a factor series, which has no entity key at all.
_FACTOR_CSV = """CalDt,MktRf,SMB
03/01/2020,-0.004,0.001
02/01/2020,0.010,-0.002
06/01/2020,0.002,0.003
"""

# One row per entity: static reference data, which has no date key.
_STATIC_CSV = """PERMNO,Sector,SIC
10002,Tech,7372
10001,Energy,4924
10003,Energy,1311
"""

# No key at all: an identifier link table. It still carries date columns — its
# validity window — which is exactly why "has date columns" and "has a date key"
# have to stay separate ideas. End dates stay inside the pandas-2.2 nanosecond
# bound, so 2099 stands in for the usual 9999 sentinel.
_LINK_CSV = """GVKEY,PERMNO,LinkDt,LinkEndDt
1001,10001,01/01/2015,31/12/2099
1002,10002,01/01/2016,31/12/2020
1003,10003,01/01/2017,31/12/2099
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "panel.csv"
    path.write_text(_SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def factor_csv(tmp_path: Path) -> Path:
    path = tmp_path / "factors.csv"
    path.write_text(_FACTOR_CSV, encoding="utf-8")
    return path


@pytest.fixture
def static_csv(tmp_path: Path) -> Path:
    path = tmp_path / "static.csv"
    path.write_text(_STATIC_CSV, encoding="utf-8")
    return path


@pytest.fixture
def link_csv(tmp_path: Path) -> Path:
    path = tmp_path / "link.csv"
    path.write_text(_LINK_CSV, encoding="utf-8")
    return path


@pytest.fixture
def make_schema() -> Callable[..., PanelSchema]:
    def _make(**overrides: object) -> PanelSchema:
        params: dict[str, object] = {
            "date_column": "DlyCalDt",
            "entity_column": "PERMNO",
            "date_format": "%d/%m/%Y",
            "dtypes": {"PERMNO": "int32", "DlyRet": "float64", "DlyPrc": "float64"},
            "category_columns": ("PrimaryExch",),
            "date_columns": ("DlyPrevDt",),
            "date_name": "date",
            "entity_name": "entity",
        }
        params.update(overrides)
        return PanelSchema(**params)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def factor_schema() -> PanelSchema:
    """Schema for :data:`_FACTOR_CSV` — a date key and no entity key."""
    return PanelSchema(
        date_column="CalDt",
        entity_column=None,
        date_format="%d/%m/%Y",
        dtypes={"MktRf": "float64", "SMB": "float64"},
        category_columns=(),
        date_columns=(),
        date_name="date",
        entity_name="entity",
    )


@pytest.fixture
def static_schema() -> PanelSchema:
    """Schema for :data:`_STATIC_CSV` — an entity key and no date key."""
    return PanelSchema(
        date_column=None,
        entity_column="PERMNO",
        date_format="%d/%m/%Y",
        dtypes={"PERMNO": "int32", "SIC": "int32"},
        category_columns=("Sector",),
        date_columns=(),
        date_name="date",
        entity_name="entity",
    )


@pytest.fixture
def link_schema() -> PanelSchema:
    """Schema for :data:`_LINK_CSV` — no keys, but two parsed date columns."""
    return PanelSchema(
        date_column=None,
        entity_column=None,
        date_format="%d/%m/%Y",
        dtypes={"GVKEY": "int32", "PERMNO": "int32"},
        category_columns=(),
        date_columns=("LinkDt", "LinkEndDt"),
        date_name="date",
        entity_name="entity",
    )


@pytest.fixture
def open_selection() -> PanelSelection:
    return PanelSelection(date_range=DateRange(None, None), entities=None, columns=None)


@pytest.fixture
def make_source(sample_csv: Path) -> Callable[..., CsvPanelSource]:
    def _make(
        schema: PanelSchema,
        selection: PanelSelection,
        *,
        path: Path | None = None,
        **overrides: object,
    ) -> CsvPanelSource:
        params: dict[str, object] = {
            "delimiter": ",",
            "encoding": "utf-8",
            "decimal": ".",
            "chunk_size": 500_000,
            "na_values": (),
            "keep_default_na": True,
            "sort": True,
        }
        params.update(overrides)
        return CsvPanelSource(
            path=str(path if path is not None else sample_csv),
            schema=schema,
            selection=selection,
            **params,  # type: ignore[arg-type]
        )

    return _make
