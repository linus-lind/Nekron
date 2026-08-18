"""Single-bar price-action features.

Lightweight transforms of one day's prices: the log price level, the normalized
high-low range, and where the close sits within the day's range. All are known at
the close of the same day.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import FloatArray, PanelContext, column_values
from .numeric import safe_divide, safe_log


@dataclass(frozen=True)
class LogPrice:
    """Natural log of a price column (``NaN`` for non-positive prices)."""

    input_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_column,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {self.output_name: safe_log(column_values(panel, self.input_column))}


@dataclass(frozen=True)
class HighLowRange:
    """Normalized daily range ``(H - L) / C``."""

    high_column: str
    low_column: str
    close_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        return {self.output_name: safe_divide(high - low, close)}


@dataclass(frozen=True)
class CandlePosition:
    """Position of the close within the day's range, ``(C - L) / (H - L)`` in ``[0, 1]``.

    A zero-range bar (``H == L``) yields ``NaN``.
    """

    high_column: str
    low_column: str
    close_column: str
    output_name: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.high_column, self.low_column, self.close_column)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output_name,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        high = column_values(panel, self.high_column)
        low = column_values(panel, self.low_column)
        close = column_values(panel, self.close_column)
        return {self.output_name: safe_divide(close - low, high - low)}
