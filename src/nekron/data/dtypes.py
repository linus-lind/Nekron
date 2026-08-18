"""Backend-agnostic dtype resolution for panel columns."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal

import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from .base import IngestionError
from .schema import PanelSchema

DATE_UNIT: Literal["s", "ms", "us", "ns"] = "us"
"""Resolution every parsed date is normalized to.

Pinning the unit is what makes ingestion deterministic. Left alone, pandas 2.2
parses to nanoseconds while pandas 3.0 infers the unit from the data — so the same
file yields different dtypes on different interpreters, and within a single
chunked read an all-blank chunk can even resolve to a different unit than the
chunk beside it. Microseconds rather than nanoseconds because nanoseconds overflow
in 2262, and identifier link files routinely carry a far-future sentinel to mean
"still valid".
"""


def build_read_dtypes(read_columns: Sequence[str], schema: PanelSchema) -> dict[Hashable, str]:
    """Map every column to the dtype used when *reading* raw values.

    Date and categorical columns are read as text and converted afterward
    (:func:`parse_dates`, :func:`apply_categoricals`); every other column is read
    as its explicit schema dtype. Raises :class:`~.base.IngestionError` if any
    column resolves to no dtype, so reads never fall back to per-chunk inference.

    Keys are typed :class:`~collections.abc.Hashable` for direct use as the
    ``dtype`` argument of :func:`pandas.read_csv`.
    """
    text_columns = set(schema.parsed_date_columns) | set(schema.category_columns)
    dtypes: dict[Hashable, str] = {}
    unresolved: list[str] = []
    for column in read_columns:
        if column in text_columns:
            dtypes[column] = "object"
        elif column in schema.dtypes:
            dtypes[column] = schema.dtypes[column]
        else:
            unresolved.append(column)
    if unresolved:
        raise IngestionError(
            f"no dtype resolved for columns {unresolved}; declare each in the schema's "
            "dtypes, category_columns, or date_columns."
        )
    return dtypes


def parse_dates(
    frame: pd.DataFrame, schema: PanelSchema, columns: Sequence[str] | None = None
) -> None:
    """Convert date columns to ``datetime64`` in place.

    ``columns`` selects which date columns to parse; ``None`` parses every date
    column. Missing entries become ``NaT``; a non-empty value that does not match
    :attr:`PanelSchema.date_format` raises, unless the schema opts into reading
    unrepresentable dates as missing (see
    :attr:`PanelSchema.out_of_bounds_dates`).
    """
    targets = schema.parsed_date_columns if columns is None else columns
    for column in targets:
        if column in frame.columns:
            frame[column] = _to_datetime(frame[column], column, schema)


def _to_datetime(values: pd.Series, column: str, schema: PanelSchema) -> pd.Series:
    """Parse one date column, turning a resolution overflow into an actionable error."""
    try:
        return pd.to_datetime(values, format=schema.date_format).dt.as_unit(DATE_UNIT)
    except OutOfBoundsDatetime as exc:
        if schema.out_of_bounds_dates == "null":
            coerced = pd.to_datetime(values, format=schema.date_format, errors="coerce")
            return coerced.dt.as_unit(DATE_UNIT)
        raise IngestionError(
            f"column {column!r} holds a date this pandas build cannot represent: {exc}. "
            "A far-future 'still valid' sentinel is the usual cause; set the schema's "
            "out_of_bounds_dates to 'null' to read such values as missing, which is how "
            "an open-ended validity window is meant to be spelled."
        ) from exc


def apply_categoricals(frame: pd.DataFrame, schema: PanelSchema) -> None:
    """Convert every present categorical column to ``category`` dtype in place."""
    for column in schema.category_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
