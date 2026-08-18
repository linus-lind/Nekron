"""Panel data ingestion: typed, keyed frames from a set of named, swappable sources."""

from __future__ import annotations

from .base import IngestionError, MissingColumnsError, PanelSource
from .config import (
    CsvSourceConfig,
    FilterSpec,
    IngestionConfig,
    PanelSpec,
    SchemaConfig,
    SelectionConfig,
    SourceConfig,
    register_configs,
    to_config,
    to_schema,
    to_selection,
)
from .csv_source import CsvPanelSource
from .dtypes import DATE_UNIT, apply_categoricals, build_read_dtypes, parse_dates
from .filters import (
    FilterPhase,
    PanelFilter,
    TopNByMarketCap,
    build_filter,
    register_filter,
    registered_filters,
)
from .ingest import load_panel, load_panels, source_paths
from .registry import build_source, register_source
from .schema import PanelSchema
from .selection import DateRange, PanelSelection

__all__ = [
    "IngestionError",
    "MissingColumnsError",
    "PanelSource",
    "PanelSchema",
    "DateRange",
    "PanelSelection",
    "CsvPanelSource",
    "DATE_UNIT",
    "build_read_dtypes",
    "parse_dates",
    "apply_categoricals",
    "build_source",
    "register_source",
    "load_panel",
    "load_panels",
    "source_paths",
    "IngestionConfig",
    "PanelSpec",
    "SchemaConfig",
    "SelectionConfig",
    "SourceConfig",
    "CsvSourceConfig",
    "register_configs",
    "to_config",
    "to_schema",
    "to_selection",
    # filters
    "FilterSpec",
    "FilterPhase",
    "PanelFilter",
    "TopNByMarketCap",
    "register_filter",
    "build_filter",
    "registered_filters",
]
