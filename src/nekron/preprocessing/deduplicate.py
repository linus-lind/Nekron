"""Merging of duplicated panel rows via NaN-resistant aggregation."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from nekron.panel import PanelGrain

from .base import key_array, require_named_levels


@dataclass(frozen=True)
class DuplicateMerger:
    """Collapse rows sharing the same key into one aggregated row.

    Numeric columns are merged by their NaN-skipping mean; other columns take
    the first non-null value. Rows with a unique key pass through unchanged.

    Parameters
    ----------
    keys:
        Index-level names and/or column names identifying a duplicate. When
        empty, the full panel index is used.
    """

    grains: ClassVar[tuple[PanelGrain, ...]] = (
        PanelGrain.PANEL,
        PanelGrain.TIME_SERIES,
        PanelGrain.CROSS_SECTION,
    )
    """Collapses rows sharing an index key, so any keyed grain works; a keyless table
    has no key to collapse on."""

    keys: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        require_named_levels(panel.index)
        index_names: list[Hashable] = list(panel.index.names)
        key_names: list[Hashable]
        if self.keys:
            dup_mask = pd.MultiIndex.from_arrays(
                [key_array(panel, k) for k in self.keys]
            ).duplicated(keep=False)
            key_names = list(self.keys)
        else:
            dup_mask = panel.index.duplicated(keep=False)
            key_names = index_names

        if not dup_mask.any():
            return panel

        merged = self._merge(panel.loc[dup_mask], key_names, index_names)
        combined = pd.concat([panel.loc[~dup_mask], merged])
        return combined.sort_index()

    @staticmethod
    def _merge(
        duplicates: pd.DataFrame, key_names: list[Hashable], index_names: list[Hashable]
    ) -> pd.DataFrame:
        reset = duplicates.reset_index()
        agg: dict[Hashable, str] = {}
        for column in reset.columns:
            if column in key_names:
                continue
            dtype = reset[column].dtype
            agg[column] = (
                "mean" if is_numeric_dtype(dtype) and not is_bool_dtype(dtype) else "first"
            )
        grouped = reset.groupby(key_names, sort=False, as_index=False).agg(agg)
        return grouped.set_index(index_names)
