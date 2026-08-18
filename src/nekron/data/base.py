"""Core abstractions for panel data ingestion.

Ingestion turns an external tabular source into a *raw panel*: a
:class:`pandas.DataFrame` indexed by whichever keys its schema declares — a
two-level ``(date, entity)`` :class:`pandas.MultiIndex` in the usual case, a
single date or entity level for a factor series or a static frame, and no key
levels at all for an identifier link table — with correctly typed columns, ready
to be handed to :mod:`nekron.preprocessing`.

:class:`PanelSource` is the switchable adapter contract. One adapter exists per
source format (CSV, and — in future — Parquet, SQL, …); all produce the same
panel layout, so downstream code stays agnostic to where the data came from.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


class IngestionError(Exception):
    """Base class for data-ingestion errors."""


class MissingColumnsError(IngestionError):
    """Raised when a source lacks columns required by the schema or selection."""


@runtime_checkable
class PanelSource(Protocol):
    """A source that loads a keyed frame from one storage backend."""

    @property
    def source_paths(self) -> tuple[str, ...]:
        """Filesystem paths this source reads.

        Declared on the contract rather than discovered per backend because the
        stage cache has to fingerprint the underlying files: a configuration that
        names ``data/crsp.csv`` says nothing about what that file currently
        contains. A backend that reads no files (a database query, say) returns an
        empty tuple, which simply means its artifacts are keyed on the query alone.
        """
        ...

    def load(self) -> pd.DataFrame:
        """Read, filter, and type the source into its declared index shape."""
        ...
