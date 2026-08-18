"""Ordering-consistency checks between columns or against fixed bounds."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from nekron.panel import PanelGrain

from .base import ALL_GRAINS, require_columns

_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
}


@dataclass(frozen=True)
class ConsistencyRule:
    """A single ``column <op> right`` validity check.

    ``right`` is a fixed value (set ``bound``) or another column (set
    ``other_column``). Rows failing the check are corrected: value comparisons set
    the offending ``column`` value to NaN; column comparisons set the designated
    ``target`` column to NaN. Rows with a missing operand are left untouched.

    Parameters
    ----------
    column:
        Left-hand column of the comparison.
    op:
        One of ``"gt"``, ``"ge"``, ``"lt"``, ``"le"`` (the valid condition).
    bound:
        Right-hand fixed value (value comparison).
    other_column:
        Right-hand column (column comparison).
    target:
        For a column comparison, which of ``column`` / ``other_column`` is set
        to NaN when the check fails.
    """

    column: str
    op: str
    bound: float | None = None
    other_column: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"op must be one of {tuple(_OPS)}; got {self.op!r}.")
        if (self.bound is None) == (self.other_column is None):
            raise ValueError("exactly one of bound / other_column must be set.")
        if self.other_column is not None:
            if self.target is None:
                raise ValueError("target is required when comparing two columns.")
            if self.target not in (self.column, self.other_column):
                raise ValueError("target must be either column or other_column.")


@dataclass(frozen=True)
class ConsistencyFilter:
    """Apply a sequence of consistency rules to the panel, in place."""

    grains: ClassVar[tuple[PanelGrain, ...]] = ALL_GRAINS
    """Pure column comparison: no index level is read, so this is meaningful for
    every grain."""

    rules: tuple[ConsistencyRule, ...]

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        for rule in self.rules:
            self._apply_rule(panel, rule)
        return panel

    @staticmethod
    def _apply_rule(panel: pd.DataFrame, rule: ConsistencyRule) -> None:
        op = _OPS[rule.op]
        if rule.other_column is None:
            require_columns(panel, (rule.column,))
            left = panel[rule.column]
            invalid = ~op(left, rule.bound) & left.notna()
            if invalid.any():
                panel.loc[invalid, rule.column] = np.nan
        else:
            require_columns(panel, (rule.column, rule.other_column))
            left = panel[rule.column]
            right = panel[rule.other_column]
            invalid = ~op(left, right) & left.notna() & right.notna()
            if invalid.any():
                assert rule.target is not None  # guaranteed by ConsistencyRule
                panel.loc[invalid, rule.target] = np.nan
