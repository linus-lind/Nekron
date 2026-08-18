"""Top-level panel ingestion entry points."""

from __future__ import annotations

import logging

from nekron.panel import Panel, PanelSet

from .config import IngestionConfig, PanelSpec, to_schema, to_selection
from .filters import FilterPhase, build_filter
from .registry import build_source

logger = logging.getLogger(__name__)


def load_panel(spec: PanelSpec) -> Panel:
    """Load the single source described by ``spec`` into a keyed :class:`Panel`.

    Resolves the schema and selection, applies the configured filters (row-phase
    filters run per chunk inside the source; cross-section filters run once on the
    assembled frame), and returns the typed, indexed panel. The index shape is
    whatever the schema's key columns imply, so this is equally the entry point
    for a ``(date, entity)`` panel, a date-keyed series, an entity-keyed
    cross-section and an unkeyed link table.
    """
    schema = to_schema(spec.schema)
    selection = to_selection(spec.selection)
    filters = tuple(build_filter(item.type, item.params) for item in spec.filters)
    row_filters = tuple(item for item in filters if item.phase is FilterPhase.ROW)

    source = build_source(spec.source, schema, selection, row_filters)
    frame = source.load()
    for panel_filter in filters:
        if panel_filter.phase is FilterPhase.CROSS_SECTION:
            frame = panel_filter.apply(frame)

    panel = Panel(
        frame=frame,
        date_name=schema.date_name if schema.date_column is not None else None,
        entity_name=schema.entity_name if schema.entity_column is not None else None,
    )
    panel.validate()
    return panel


def source_paths(spec: PanelSpec) -> tuple[str, ...]:
    """The filesystem paths ``spec`` reads, for cache fingerprinting.

    Builds the source but does not read it; every adapter is a cheap value object,
    so this costs nothing and avoids duplicating each backend's notion of where
    its bytes come from.
    """
    schema = to_schema(spec.schema)
    selection = to_selection(spec.selection)
    return build_source(spec.source, schema, selection, ()).source_paths


def load_panels(cfg: IngestionConfig) -> PanelSet:
    """Load every named source in ``cfg`` into a :class:`~nekron.panel.PanelSet`.

    Sources are loaded one at a time and independently: nothing here joins them,
    so peak memory is the sum of the loaded panels plus the working set of the one
    currently being read, not a multiple of it.
    """
    panels: dict[str, Panel] = {}
    for name, spec in cfg.panels.items():
        panel = load_panel(spec)
        logger.info(
            "ingested panel %r: grain=%s rows=%d columns=%d",
            name,
            panel.grain.value,
            len(panel.frame),
            panel.frame.shape[1],
        )
        panels[name] = panel
    return PanelSet(panels)
