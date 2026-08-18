"""Declarative description of a source's panel layout and column types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nekron.panel import PanelGrain


@dataclass(frozen=True)
class PanelSchema:
    """Maps raw source columns onto a typed, keyed frame.

    Which of :attr:`date_column` and :attr:`entity_column` are set decides the
    source's :class:`~nekron.panel.PanelGrain`, and therefore what the loaded
    frame is indexed by. Setting both — the common case — gives a ``(date,
    entity)`` panel; setting one gives a date-keyed series or an entity-keyed
    cross-section; setting neither gives an unkeyed table, which is what an
    identifier link file is.

    Parameters
    ----------
    date_column:
        Source column parsed into the date index level, or ``None`` when the
        source has no date key. A link table typically sets this to ``None`` while
        still listing its validity columns in :attr:`date_columns`.
    entity_column:
        Source column used as the entity index level, or ``None`` when the source
        has no entity key.
    date_format:
        ``strptime`` format string (e.g. ``"%d/%m/%Y"``) applied to every date
        column.
    dtypes:
        Explicit per-column dtype specification for the non-date, non-categorical
        columns. Columns absent from the mapping are left to backend inference.
    category_columns:
        Low-cardinality text columns materialized as ``category`` dtype.
    date_columns:
        Columns beyond :attr:`date_column` that are parsed as dates using
        :attr:`date_format`.
    date_name:
        Name given to the date level of the resulting index.
    entity_name:
        Name given to the entity level of the resulting index.
    out_of_bounds_dates:
        What to do with a date the parser cannot represent. ``"raise"`` (the
        default) reports it; ``"null"`` reads it as missing.

        This exists because identifier link files conventionally mark a still-open
        validity window with a far-future sentinel such as ``9999-12-31``. Under
        pandas 3 that parses at microsecond resolution and is simply a very large
        date, but pandas 2.2 parses at nanosecond resolution, which tops out in
        2262, so the same file raises. ``"null"`` is the right setting for such a
        column in either case: the link resolver already reads a missing bound as
        "still valid", which is exactly what the sentinel means. Note that it makes
        *any* unparseable value in that column null, so set it deliberately.
    """

    date_column: str | None
    entity_column: str | None
    date_format: str
    dtypes: Mapping[str, str]
    category_columns: tuple[str, ...]
    date_columns: tuple[str, ...]
    date_name: str
    entity_name: str
    out_of_bounds_dates: str = "raise"

    def __post_init__(self) -> None:
        if self.out_of_bounds_dates not in ("raise", "null"):
            raise ValueError(
                f"out_of_bounds_dates must be 'raise' or 'null'; got {self.out_of_bounds_dates!r}."
            )
        for field_name in ("date_column", "entity_column"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(
                    f"{field_name} must be a column name or None; an empty string is not a "
                    "valid sentinel for 'no such key'."
                )

    @property
    def grain(self) -> PanelGrain:
        """The index shape this schema produces."""
        return PanelGrain.of(
            date=self.date_column is not None, entity=self.entity_column is not None
        )

    @property
    def key_columns(self) -> tuple[str, ...]:
        """The source columns that become index levels, in index order."""
        return tuple(
            column for column in (self.date_column, self.entity_column) if column is not None
        )

    @property
    def index_names(self) -> tuple[str, ...]:
        """The names given to the index levels, aligned with :attr:`key_columns`."""
        names = []
        if self.date_column is not None:
            names.append(self.date_name)
        if self.entity_column is not None:
            names.append(self.entity_name)
        return tuple(names)

    @property
    def parsed_date_columns(self) -> tuple[str, ...]:
        """Every column parsed as a date, index date first, without duplicates.

        A source with no date key can still carry date columns — the validity
        window of a link table is the motivating case — so this is not empty just
        because :attr:`date_column` is ``None``.
        """
        index_date = () if self.date_column is None else (self.date_column,)
        return tuple(dict.fromkeys((*index_date, *self.date_columns)))
