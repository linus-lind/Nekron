"""Cross-sectional (per-date) transforms applied across the entity universe.

These operators standardize a raw signal against its cross-section on each date,
which is what makes features comparable across time and ready for cross-sectional
models. All are point-in-time: they use only same-date values, so they add no
look-ahead provided the input signal is itself known at the close of that date.
Each transform accepts parallel ``input_columns`` and ``output_names`` and emits
one column per input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FloatArray, PanelContext, check_named, column_values, key_array_or_column
from .engine import (
    cross_sectional_bucket,
    cross_sectional_demean,
    cross_sectional_minmax,
    cross_sectional_rank,
    cross_sectional_winsorize,
    cross_sectional_zscore,
)

_SCALINGS = ("raw", "pct", "signed")


@dataclass(frozen=True)
class CrossSectionalRank:
    """Per-date cross-sectional rank of each input column.

    ``scaling`` selects the output form: ``"raw"`` ordinal ranks, ``"pct"`` ranks
    in ``(0, 1]``, or ``"signed"`` ranks affinely mapped to ``[-1, 1]`` (the
    Gu-Kelly-Xiu normalization, ``2*(rank-1)/(N-1) - 1``, with a one-name
    cross-section mapped to ``0``).
    """

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    method: str
    ascending: bool
    scaling: str

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")
        if self.scaling not in _SCALINGS:
            raise ValueError(f"scaling must be one of {_SCALINGS}; got {self.scaling!r}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        out: dict[str, FloatArray] = {}
        for column, name in zip(self.input_columns, self.output_names, strict=True):
            values = column_values(panel, column)
            if self.scaling == "signed":
                out[name] = self._signed(values, ctx)
            else:
                out[name] = cross_sectional_rank(
                    values,
                    ctx,
                    method=self.method,
                    pct=self.scaling == "pct",
                    ascending=self.ascending,
                )
        return out

    def _signed(self, values: FloatArray, ctx: PanelContext) -> FloatArray:
        rank = cross_sectional_rank(
            values, ctx, method=self.method, pct=False, ascending=self.ascending
        )
        count = pd.Series(values).groupby(ctx.date_codes, sort=False).transform("count").to_numpy()
        with np.errstate(invalid="ignore"):
            signed = 2.0 * (rank - 1.0) / (count - 1.0) - 1.0
        return np.where(count > 1, signed, np.where(np.isnan(rank), np.nan, 0.0))


@dataclass(frozen=True)
class CrossSectionalZScore:
    """Standardize each input column to zero mean and unit std within each date."""

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    ddof: int

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {
            name: cross_sectional_zscore(column_values(panel, column), ctx, ddof=self.ddof)
            for column, name in zip(self.input_columns, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class CrossSectionalDemean:
    """Subtract the per-date cross-sectional mean from each input column."""

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {
            name: cross_sectional_demean(column_values(panel, column), ctx)
            for column, name in zip(self.input_columns, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class CrossSectionalMinMax:
    """Scale each input column to ``[0, 1]`` within each date's cross-section."""

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {
            name: cross_sectional_minmax(column_values(panel, column), ctx)
            for column, name in zip(self.input_columns, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class CrossSectionalWinsorize:
    """Clip each input column to its per-date ``[lower, upper]`` quantiles."""

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    lower: float
    upper: float

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("require 0 <= lower <= upper <= 1.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {
            name: cross_sectional_winsorize(
                column_values(panel, column), ctx, lower=self.lower, upper=self.upper
            )
            for column, name in zip(self.input_columns, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class CrossSectionalBucket:
    """Assign each input column to per-date quantile buckets ``1 .. n_buckets``."""

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    n_buckets: int

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")
        if self.n_buckets <= 0:
            raise ValueError(f"n_buckets must be positive; got {self.n_buckets}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        return {
            name: cross_sectional_bucket(
                column_values(panel, column), ctx, n_buckets=self.n_buckets
            )
            for column, name in zip(self.input_columns, self.output_names, strict=True)
        }


@dataclass(frozen=True)
class GroupNeutralize:
    """Neutralize each input column within per-date groups (e.g. industry).

    Subtracts the group mean (``mode="demean"``) or standardizes within the group
    (``mode="zscore"``), where a group is the set of names sharing both the date
    and the value of ``group_column`` (an index level or panel column).
    """

    input_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    group_column: str
    mode: str
    ddof: int

    def __post_init__(self) -> None:
        check_named(len(self.input_columns), self.output_names, "input columns")
        if self.mode not in ("demean", "zscore"):
            raise ValueError(f"mode must be 'demean' or 'zscore'; got {self.mode!r}.")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_columns

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.output_names

    @property
    def key_inputs(self) -> tuple[str, ...]:
        return (self.group_column,)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        group_values = key_array_or_column(panel, self.group_column)
        keys = [ctx.date_codes, group_values]
        out: dict[str, FloatArray] = {}
        for column, name in zip(self.input_columns, self.output_names, strict=True):
            series = pd.Series(column_values(panel, column))
            grouped = series.groupby(keys, sort=False)
            mean = grouped.transform("mean")
            centered = (series - mean).to_numpy()
            if self.mode == "demean":
                out[name] = centered
            else:
                count = grouped.transform("count").to_numpy()
                ssq = ((series - mean) ** 2).groupby(keys, sort=False).transform("sum").to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    std = np.sqrt(np.maximum(ssq / (count - self.ddof), 0.0))
                out[name] = np.where(std > 0.0, centered / std, np.nan)
        return out
