"""Typed Hydra configuration for panel ingestion.

The dataclasses below are the config schema *and* the source of the default
values. They are registered with Hydra's :class:`ConfigStore` (see
:func:`register_configs`), so a run composes ``configs/data/*.yaml``, validates it
against this schema, and materializes a typed :class:`IngestionConfig` via
:func:`to_config`.

A run reads a *set* of named sources, not one file. :class:`PanelSpec` describes
one of them — its schema, its row/column selection, its backend and its filters —
and :class:`IngestionConfig` holds them by name. Each name is a Hydra mount point,
so a source file is a config group that gets attached under the name a run refers
to it by::

    defaults:
      - data@data_pipeline.ingestion.panels.crsp: crsp
      - data@data_pipeline.ingestion.panels.compustat: compustat
      - data@data_pipeline.ingestion.panels.ccm_link: ccm_link

Anything is then overridable from the CLI through that name::

    python -m nekron.asset_pricing.conditional_autoencoder \\
        data_pipeline.ingestion.panels.crsp.selection.start_date=2010-01-01 \\
        data_pipeline.ingestion.panels.compustat.source.csv.chunk_size=250000

Note that a *new* panel can only be added from the command line by mounting a
group (``+data@data_pipeline.ingestion.panels.extra=compustat``); appending a raw
dict literal to a typed mapping is rejected by OmegaConf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from nekron.constants import DATE_LEVEL, ENTITY_LEVEL

from .schema import PanelSchema
from .selection import DateRange, PanelSelection


@dataclass
class SchemaConfig:
    """Column-to-index mapping and dtypes (see :class:`~.schema.PanelSchema`).

    ``date_column`` and ``entity_column`` are mandatory but nullable, and that
    combination is deliberate: which of them a source declares determines its
    grain, so a source that has no entity key must say ``entity_column: null``
    rather than leave the field out. Omitting a key by accident would otherwise
    silently reshape the source into something else.
    """

    date_column: str | None = MISSING
    entity_column: str | None = MISSING
    date_format: str = MISSING
    dtypes: dict[str, str] = field(default_factory=dict)
    category_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    date_name: str = DATE_LEVEL
    entity_name: str = ENTITY_LEVEL
    out_of_bounds_dates: str = "raise"


@dataclass
class SelectionConfig:
    """Row/column selection (see :class:`~.selection.PanelSelection`).

    ``None`` fields impose no restriction: full date range, all entities, all
    columns. ``start_date`` / ``end_date`` are ISO-8601 strings.
    """

    start_date: str | None = None
    end_date: str | None = None
    entities: list[Any] | None = None
    columns: list[str] | None = None


@dataclass
class CsvSourceConfig:
    """CSV adapter settings (see :class:`~.csv_source.CsvPanelSource`)."""

    path: str = MISSING
    delimiter: str = ","
    encoding: str = "utf-8"
    decimal: str = "."
    chunk_size: int = 200_000
    na_values: list[str] = field(default_factory=list)
    keep_default_na: bool = True
    sort: bool = True


@dataclass
class SourceConfig:
    """Which source adapter to use and its per-format settings."""

    format: str = "csv"
    csv: CsvSourceConfig = field(default_factory=CsvSourceConfig)


@dataclass
class FilterSpec:
    """One ingestion filter: its registered ``type`` and constructor ``params``."""

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelSpec:
    """Everything needed to load one named source into a keyed frame."""

    schema: SchemaConfig = field(default_factory=SchemaConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    filters: list[FilterSpec] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """The set of named sources a run loads.

    Panels are iterated in the order Hydra composed them, which is the order of
    the defaults list rather than alphabetical order. Nothing downstream should
    depend on that order — the alignment stage addresses panels by name.
    """

    panels: dict[str, PanelSpec] = field(default_factory=dict)


CONFIG_SCHEMA_NAME = "base_ingestion"
PANEL_SCHEMA_NAME = "base_panel"


def register_configs() -> None:
    """Register the ingestion schemas with Hydra's ConfigStore (call before @hydra.main).

    Both nodes are registered ungrouped, which is what lets a source file open
    with ``- /base_panel@_here_`` and be validated against :class:`PanelSpec` on
    its own, independently of where it is later mounted.
    """
    store = ConfigStore.instance()
    store.store(name=PANEL_SCHEMA_NAME, node=PanelSpec)
    store.store(name=CONFIG_SCHEMA_NAME, node=IngestionConfig)


def to_config(cfg: DictConfig) -> IngestionConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed ``IngestionConfig``.

    Unresolved mandatory values (``???``) raise here, naming the panel they belong
    to.
    """
    return cast(IngestionConfig, OmegaConf.to_object(cfg))


def to_schema(cfg: SchemaConfig) -> PanelSchema:
    """Build the backend-agnostic :class:`PanelSchema` from its config."""
    return PanelSchema(
        date_column=cfg.date_column,
        entity_column=cfg.entity_column,
        date_format=cfg.date_format,
        dtypes=dict(cfg.dtypes),
        category_columns=tuple(cfg.category_columns),
        date_columns=tuple(cfg.date_columns),
        date_name=cfg.date_name,
        entity_name=cfg.entity_name,
        out_of_bounds_dates=cfg.out_of_bounds_dates,
    )


def to_selection(cfg: SelectionConfig) -> PanelSelection:
    """Build the backend-agnostic :class:`PanelSelection` from its config."""
    return PanelSelection(
        date_range=DateRange.from_iso(cfg.start_date, cfg.end_date),
        entities=frozenset(cfg.entities) if cfg.entities is not None else None,
        columns=tuple(cfg.columns) if cfg.columns is not None else None,
    )
