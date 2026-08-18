"""Removal of entities (or entity subgroups) that are too sparse in any column."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nekron.constants import ENTITY_LEVEL

from .base import key_array, require_columns, resolve_level


@dataclass(frozen=True)
class NaNEntityFilter:
    """Drop groups whose NaN fraction exceeds a threshold in any evaluated column.

    A group is one entity, optionally further split by ``subgroup_keys`` (for
    example a membership-period column, so sparsity is judged only within each
    period). A group is removed when the NaN fraction of at least one evaluated
    column is above the threshold; all rows of that group are dropped.

    Parameters
    ----------
    threshold:
        Maximum allowed NaN fraction in ``[0, 1]``; a group strictly above it in
        any evaluated column is dropped.
    columns:
        Columns whose per-group NaN fraction is evaluated. When empty, every column
        is evaluated.
    entity_level:
        Position or name of the entity level.
    subgroup_keys:
        Additional index-level or column names appended to the grouping.
    """

    threshold: float
    columns: tuple[str, ...] = field(default_factory=tuple)
    entity_level: int | str = ENTITY_LEVEL
    subgroup_keys: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be a fraction in [0, 1]; got {self.threshold}.")

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        targets = list(self.columns) if self.columns else list(panel.columns)
        require_columns(panel, targets)
        entity = np.asarray(
            panel.index.get_level_values(resolve_level(panel.index, self.entity_level))
        )
        group_arrays = [entity, *(key_array(panel, k) for k in self.subgroup_keys)]
        # ``dropna=False`` keeps rows whose subgroup key is NaN (e.g. dates outside
        # any membership period) as their own group instead of silently discarding
        # them. Fractions are computed for every evaluated column at once, then a
        # group is over-threshold if any column is.
        fractions = (
            panel[targets].isna().groupby(group_arrays, sort=False, dropna=False).transform("mean")
        )
        over_threshold = (fractions > self.threshold).any(axis=1).to_numpy()
        if not over_threshold.any():
            return panel
        return panel.loc[~over_threshold].copy(deep=False)
