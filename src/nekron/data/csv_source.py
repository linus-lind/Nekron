"""CSV-backed :class:`PanelSource` with bounded-memory chunked reading."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import pandas as pd
from pandas.errors import EmptyDataError

from .base import IngestionError, MissingColumnsError
from .dtypes import apply_categoricals, build_read_dtypes, parse_dates
from .filters import PanelFilter
from .schema import PanelSchema
from .selection import PanelSelection


@dataclass(frozen=True)
class CsvPanelSource:
    """Load a ``(date, entity)`` panel from a delimited text file.

    The file is read in row chunks; within each chunk the index date is parsed
    and the rows are reduced to the selection before being retained, and the
    remaining date columns are parsed once on the retained rows. Requested
    columns are read via ``usecols`` and typed from the schema.

    Parameters
    ----------
    path:
        Filesystem path to the delimited file.
    schema:
        Column-to-panel mapping and dtypes.
    selection:
        Date-range, entity, and column subset to load.
    delimiter:
        Field separator.
    encoding:
        Text encoding of the file.
    decimal:
        Character recognized as the decimal point.
    chunk_size:
        Number of rows read per chunk. This bounds the *parser's* transient
        working set, not peak memory: every retained row is held until the read
        finishes and is then copied once more by the concatenation, so peak scales
        with the rows retained, not with this value. Measured on a two-million-row
        file, a tenfold reduction moved peak by about a megabyte. The levers that
        do move it are :attr:`PanelSelection.columns`, the date range, and
        row-phase filters — narrowing the column selection on that same file
        halved peak.
    na_values:
        Additional strings parsed as missing values.
    keep_default_na:
        Whether the default set of NA strings is honored in addition to
        :attr:`na_values`.
    sort:
        When true, the returned panel is sorted by ``(date, entity)``.
    row_filters:
        Row-phase filters applied to each chunk before its rows are retained.
    """

    path: str
    schema: PanelSchema
    selection: PanelSelection
    delimiter: str
    encoding: str
    decimal: str
    chunk_size: int
    na_values: tuple[str, ...]
    keep_default_na: bool
    sort: bool
    row_filters: tuple[PanelFilter, ...] = ()

    @property
    def source_paths(self) -> tuple[str, ...]:
        """The single delimited file this source reads."""
        return (self.path,)

    def load(self) -> pd.DataFrame:
        header = self._read_header()
        self._require_schema_columns(header)
        usecols = self._resolve_usecols(header)
        read_columns = usecols if usecols is not None else header
        read_dtypes = build_read_dtypes(read_columns, self.schema)

        index_date = () if self.schema.date_column is None else (self.schema.date_column,)
        kept: list[pd.DataFrame] = []
        reader = pd.read_csv(
            self.path,
            sep=self.delimiter,
            encoding=self.encoding,
            decimal=self.decimal,
            usecols=usecols,
            dtype=read_dtypes,
            na_values=list(self.na_values) or None,
            keep_default_na=self.keep_default_na,
            chunksize=self.chunk_size,
        )
        rows_read = 0
        try:
            for chunk in reader:
                parse_dates(chunk, self.schema, index_date)
                selected = self.selection.filter_rows(chunk, self.schema)
                for row_filter in self.row_filters:
                    if selected.empty:
                        break
                    selected = row_filter.apply(selected)
                if not selected.empty:
                    kept.append(selected)
                rows_read += len(chunk)
        except (ValueError, OSError) as exc:
            raise IngestionError(
                f"cannot read source {self.path!r} at about row {rows_read:,}: {exc}"
            ) from exc

        if kept:
            frame = pd.concat(kept)
            parse_dates(frame, self.schema, self.schema.date_columns)
        else:
            frame = self._empty_frame(usecols, read_dtypes)
        return self._finalize(frame)

    def _read_header(self) -> list[str]:
        try:
            head = pd.read_csv(self.path, sep=self.delimiter, encoding=self.encoding, nrows=0)
        except EmptyDataError as exc:
            raise IngestionError(f"source {self.path!r} is empty; no header row found.") from exc
        return [str(column) for column in head.columns]

    def _empty_frame(
        self, usecols: list[str] | None, read_dtypes: dict[Hashable, str]
    ) -> pd.DataFrame:
        """A zero-row frame with the same columns and dtypes as a full read."""
        frame = pd.read_csv(
            self.path,
            sep=self.delimiter,
            encoding=self.encoding,
            decimal=self.decimal,
            usecols=usecols,
            dtype=read_dtypes,
            na_values=list(self.na_values) or None,
            keep_default_na=self.keep_default_na,
            nrows=0,
        )
        parse_dates(frame, self.schema)
        return frame

    def _require_schema_columns(self, header: list[str]) -> None:
        enumerated = {
            *self.schema.key_columns,
            *self.schema.date_columns,
            *self.schema.category_columns,
            *self.schema.dtypes.keys(),
        }
        missing = sorted(column for column in enumerated if column not in header)
        if missing:
            raise MissingColumnsError(f"source {self.path!r} is missing schema columns: {missing}.")

    def _resolve_usecols(self, header: list[str]) -> list[str] | None:
        if self.selection.columns is None:
            return None
        ordered = dict.fromkeys((*self.selection.columns, *self.schema.key_columns))
        missing = [column for column in ordered if column not in header]
        if missing:
            raise MissingColumnsError(
                f"source {self.path!r} is missing requested columns: {missing}."
            )
        return list(ordered)

    def _finalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        apply_categoricals(frame, self.schema)
        keys = self.schema.key_columns
        if keys:
            frame = frame.set_index(list(keys))
            # Renaming the levels in place rather than via ``rename_axis``: the
            # index was just built here and nothing else holds it, and on pandas
            # 2.2 ``rename_axis`` is an eager deep copy of the whole frame.
            frame.index.names = list(self.schema.index_names)
        else:
            # A keyless table carries whichever positional labels survived chunking
            # and row filtering; renumbering keeps position and label the same thing,
            # which is what every consumer of an unkeyed frame assumes.
            frame = frame.reset_index(drop=True)
        if self.selection.columns is not None:
            data_columns = [
                column for column in dict.fromkeys(self.selection.columns) if column not in keys
            ]
            frame = frame[data_columns]
        # Already-sorted is the common case for a keyed extract, and on pandas 2.2
        # ``sort_index`` copies even when it has nothing to do.
        if self.sort and keys and not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
        return frame
