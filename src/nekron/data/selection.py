"""Row and column selection applied while loading a panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import IngestionError
from .schema import PanelSchema


@dataclass(frozen=True)
class DateRange:
    """Inclusive ``[start, end]`` date bound; ``None`` leaves a side unbounded."""

    start: pd.Timestamp | None
    end: pd.Timestamp | None

    @classmethod
    def from_iso(cls, start: str | None, end: str | None) -> DateRange:
        """Build from ISO-8601 strings, mapping ``None``/empty to an open bound."""
        return cls(
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )

    def is_open(self) -> bool:
        """True when neither bound constrains the data."""
        return self.start is None and self.end is None


@dataclass(frozen=True)
class PanelSelection:
    """Subset of the source to load: a date range, entities, and columns.

    A ``None`` field imposes no restriction — respectively the full date range,
    all entities, or all columns.

    Parameters
    ----------
    date_range:
        Inclusive bound applied to the panel's date column.
    entities:
        Entity identifiers to keep; ``None`` keeps every entity. Values must
        match the entity column's dtype to compare equal.
    columns:
        Data columns to load; the date and entity columns are always read.
        ``None`` loads all columns.
    """

    date_range: DateRange
    entities: frozenset[Any] | None
    columns: tuple[str, ...] | None
    _entity_cache: dict[str, pd.Index] = field(default_factory=dict, compare=False, repr=False)

    def _targets(self, dtype: Any, column: str) -> pd.Index:
        """The requested entities coerced to the column's dtype, computed once.

        Both inputs are fixed for the whole read, so rebuilding this per chunk is
        pure repetition. The coercion is also checked rather than trusted: casting
        is not validation, and a value that is not actually an identifier —
        ``"10001x"`` against an integer column, or a float against an int — would
        otherwise be silently turned into one that is, quietly selecting the wrong
        entity or none at all.
        """
        cached = self._entity_cache.get(str(dtype))
        if cached is not None:
            return cached
        requested = pd.Index(sorted(self.entities or (), key=repr))
        try:
            targets: pd.Index = requested.astype(dtype)
        except (TypeError, ValueError) as exc:
            raise IngestionError(
                f"entities are incompatible with the {column!r} dtype {dtype}: {exc}"
            ) from exc
        # Casting is not validation. Writing an id as text (``"10002"`` for an
        # int32 column) is a legitimate spelling and must keep working, but a value
        # that is not an id at all — ``10.5`` truncating to ``10`` — would silently
        # select a different entity. Comparing numerically catches the second
        # without rejecting the first, and does so identically on either pandas.
        if pd.api.types.is_integer_dtype(dtype):
            try:
                numeric = pd.to_numeric(pd.Series(list(requested)))
            except (TypeError, ValueError) as exc:
                raise IngestionError(
                    f"entities are incompatible with the {column!r} dtype {dtype}: {exc}"
                ) from exc
            changed = np.asarray(numeric.to_numpy() != targets.to_numpy())
            if changed.any():
                lost = [value for value, moved in zip(requested, changed, strict=True) if moved]
                raise IngestionError(
                    f"entities {lost[:5]} are not valid {column!r} values for dtype {dtype}; "
                    "they change value when converted, which would select the wrong rows."
                )
        self._entity_cache[str(dtype)] = targets
        return targets

    def filter_rows(self, chunk: pd.DataFrame, schema: PanelSchema) -> pd.DataFrame:
        """Return the rows of ``chunk`` within the date range and entity set.

        Passes ``chunk`` through unchanged (no copy) when no filter applies.
        """
        mask: pd.Series[bool] | None = None
        date_range = self.date_range
        if not date_range.is_open():
            if schema.date_column is None:
                raise IngestionError(
                    "a date range was requested but the schema declares no date_column; "
                    "this source has no date key to filter on."
                )
            dates = chunk[schema.date_column]
            if date_range.start is not None:
                mask = dates >= date_range.start
            if date_range.end is not None:
                end_mask = dates <= date_range.end
                mask = end_mask if mask is None else mask & end_mask
        if self.entities is not None:
            if schema.entity_column is None:
                raise IngestionError(
                    "entities were requested but the schema declares no entity_column; "
                    "this source has no entity key to filter on."
                )
            entity_values = chunk[schema.entity_column]
            targets = self._targets(entity_values.dtype, schema.entity_column)
            entity_mask = entity_values.isin(targets)
            mask = entity_mask if mask is None else mask & entity_mask
        if mask is None:
            return chunk
        return chunk.loc[mask]
