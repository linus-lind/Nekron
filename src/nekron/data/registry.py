"""Registry mapping a source-format name to a :class:`PanelSource` factory.

Adding a backend (Parquet, SQL, …) means writing an adapter that satisfies
:class:`~.base.PanelSource` and registering a factory here under a format name;
no calling code changes.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import IngestionError, PanelSource
from .config import SourceConfig
from .csv_source import CsvPanelSource
from .filters import PanelFilter
from .schema import PanelSchema
from .selection import PanelSelection

SourceFactory = Callable[
    [SourceConfig, PanelSchema, PanelSelection, tuple[PanelFilter, ...]], PanelSource
]

_FACTORIES: dict[str, SourceFactory] = {}


def register_source(name: str, factory: SourceFactory) -> None:
    """Register ``factory`` under a source-format ``name`` (overwrites in place)."""
    _FACTORIES[name] = factory


def build_source(
    source: SourceConfig,
    schema: PanelSchema,
    selection: PanelSelection,
    row_filters: tuple[PanelFilter, ...] = (),
) -> PanelSource:
    """Construct the :class:`PanelSource` for ``source.format``."""
    try:
        factory = _FACTORIES[source.format]
    except KeyError:
        raise IngestionError(
            f"unknown source format {source.format!r}; registered formats: {sorted(_FACTORIES)}."
        ) from None
    return factory(source, schema, selection, row_filters)


def _build_csv(
    source: SourceConfig,
    schema: PanelSchema,
    selection: PanelSelection,
    row_filters: tuple[PanelFilter, ...],
) -> CsvPanelSource:
    csv = source.csv
    return CsvPanelSource(
        path=csv.path,
        schema=schema,
        selection=selection,
        delimiter=csv.delimiter,
        encoding=csv.encoding,
        decimal=csv.decimal,
        chunk_size=csv.chunk_size,
        na_values=tuple(csv.na_values),
        keep_default_na=csv.keep_default_na,
        sort=csv.sort,
        row_filters=row_filters,
    )


register_source("csv", _build_csv)
