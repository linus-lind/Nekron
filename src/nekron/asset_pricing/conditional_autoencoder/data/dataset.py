"""Turn the merged feature panel into per-period cross-sections.

A training example is one period's cross-section: the beta-input matrix ``Z_beta``
of shape ``[N, P_beta]`` (the beta-network features), the portfolio-characteristic
matrix ``Z_port`` of shape ``[N, P_port]`` (used to form the managed portfolios) and
the contemporaneous return vector ``r`` of shape ``[N]``. This module selects the
beta columns, the portfolio columns and the return column, and builds one
:class:`CrossSection` per period. When standardization is enabled the beta and
portfolio columns are rank-mapped cross-sectionally to ``[-1, 1]`` per period
(missing values -> the cross-sectional median, 0) and the return column is
cross-sectionally z-scored per period (zero mean, unit standard deviation). Stocks
with a missing return, and periods with too few stocks, are dropped.

Built once, sliced many times
-----------------------------
There are two ways in. :func:`build_panel_splits` takes the adapter's
:class:`~nekron.data_adapter.DataSplits` and builds the three splits separately —
the single-split path, unchanged. :func:`build_cross_section_panel` builds every
period of the whole panel once and hands back a :class:`CrossSectionPanel` that a
fold slices with :meth:`CrossSectionPanel.splits`.

The second exists because cross-validation folds overlap. Every transform here is
strictly *within* a period — the rank is taken per date, the z-score is taken per
date, and the managed portfolios are an ordinary least-squares solve on one date's
cross-section — so a period's tensors do not depend on which other periods are
present, and slicing a shared sequence gives bit-identical results to rebuilding
each fold's periods from its own frame. Rebuilding them is what costs: a
walk-forward sweep re-ranks and re-solves the same dates once per fold that
contains them, and holds a private copy of the tensors for each.

Folds are cut over the periods that *survive*
---------------------------------------------
:attr:`CrossSectionPanel.dates` is the periods that produced a cross-section, which
is not the panel's date index: a date whose cross-section is smaller than
``min_cross_section`` is dropped here. A caller cutting folds must cut them over
that sequence, or every boundary after the first dropped date is off by the number
dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nekron.data_adapter import DataSplits
from nekron.data_adapter.splitting import FoldBounds

from ..config import DataConfig
from ..model import ConditionalAutoencoder


@dataclass(frozen=True)
class CrossSection:
    """One period's cross-section of beta inputs, portfolio inputs and returns.

    ``date_index`` is this period's position in the sequence it was built from —
    the whole panel for :func:`build_cross_section_panel`, one split for
    :func:`build_panel_splits`. A section is shared by every fold that covers it,
    so the position cannot be per-fold; use :attr:`date` to join a result back
    onto anything.
    """

    date_index: int
    date: pd.Timestamp
    beta_inputs: torch.Tensor  # [N, P_beta]
    portfolio_inputs: torch.Tensor  # [N, P_port]
    returns: torch.Tensor  # [N]
    portfolios: torch.Tensor  # [P_port] - managed portfolios, solved once here
    entities: tuple[str, ...]

    @property
    def num_stocks(self) -> int:
        return int(self.returns.shape[0])


class CrossSectionDataset(Dataset[CrossSection]):
    """A window onto a date-ordered sequence of cross-sections.

    ``span`` selects which of ``sections`` this dataset exposes; ``None`` exposes
    all of them, which is the whole of the single-split behaviour. Holding a span
    rather than its own tuple is what lets overlapping folds share one set of
    tensors: two folds covering the same period index the same object, and no
    period is standardized or least-squares-solved twice.
    """

    def __init__(self, sections: tuple[CrossSection, ...], span: range | None = None) -> None:
        self._sections = sections
        self._span = range(len(sections)) if span is None else span
        if self._span.step != 1:
            raise ValueError(
                f"a cross-section span must be contiguous; got step {self._span.step}."
            )
        # Checked on the bounds rather than the contents, so an empty span pointing
        # outside the sequence is caught too: that is a mis-cut schedule, which
        # would otherwise pass silently as "this fold has no periods".
        if (
            self._span.start < 0
            or self._span.stop < self._span.start
            or self._span.stop > len(sections)
        ):
            raise IndexError(
                f"span {self._span} does not fit a sequence of {len(sections)} cross-sections."
            )

    def __len__(self) -> int:
        return len(self._span)

    def __getitem__(self, index: int) -> CrossSection:
        return self._sections[self._span[index]]

    @property
    def span(self) -> range:
        """The positions of the underlying sequence this dataset exposes."""
        return self._span


@dataclass(frozen=True)
class PanelSplits:
    """Train/val/test cross-section datasets plus the resolved column sets."""

    train: CrossSectionDataset
    val: CrossSectionDataset
    test: CrossSectionDataset
    beta_columns: tuple[str, ...]
    portfolio_columns: tuple[str, ...]

    @property
    def num_beta_columns(self) -> int:
        return len(self.beta_columns)

    @property
    def num_portfolios(self) -> int:
        return len(self.portfolio_columns)


@dataclass(frozen=True)
class CrossSectionPanel:
    """Every usable period of the merged panel, built once, in date order.

    :attr:`dates` are the periods that produced a cross-section — see the module
    docstring on why that is not the panel's own date index, and why a fold
    schedule has to be cut over this sequence rather than that one.
    """

    sections: tuple[CrossSection, ...]
    dates: pd.Index
    beta_columns: tuple[str, ...]
    portfolio_columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def num_beta_columns(self) -> int:
        return len(self.beta_columns)

    @property
    def num_portfolios(self) -> int:
        return len(self.portfolio_columns)

    def view(self, span: range) -> CrossSectionDataset:
        """The periods at ``span``, as a dataset sharing this panel's tensors."""
        return CrossSectionDataset(self.sections, span)

    def splits(self, bounds: FoldBounds) -> PanelSplits:
        """One fold's three datasets. Constant time — nothing is copied."""
        return PanelSplits(
            train=self.view(bounds.train),
            val=self.view(bounds.val),
            test=self.view(bounds.test),
            beta_columns=self.beta_columns,
            portfolio_columns=self.portfolio_columns,
        )


def _resolve_columns(
    available: Sequence[str], requested: Sequence[str], return_column: str, role: str
) -> tuple[str, ...]:
    """Validate an explicit column list against the panel (excluding the return)."""
    missing = [c for c in requested if c not in available]
    if missing:
        raise KeyError(f"{role} not found in the feature panel: {missing}")
    if return_column in requested:
        raise ValueError(f"return_column must not be listed among {role}.")
    return tuple(requested)


def _cross_sectional_rank(frame: pd.DataFrame, date_level: int | str) -> pd.DataFrame:
    """Rank each column within every period and map ranks to ``[-1, 1]``.

    The smallest value in each cross-section maps to ``-1`` and the largest to
    ``1`` via ``2*(rank-1)/(count-1) - 1``; a lone value maps to ``0``.
    """
    grouped = frame.groupby(level=date_level)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count").to_numpy()
    # One writable copy of the ranks, then the whole mapping in place. Spelled as
    # a chain of frame expressions it allocates a fresh full-size copy at every
    # step — six of them over a panel that is already the largest thing in memory.
    # The copy is explicit because pandas 3 hands back a read-only view.
    values = ranks.to_numpy(dtype=np.float64, copy=True)
    np.subtract(values, 1.0, out=values)
    np.multiply(values, 2.0, out=values)
    np.divide(values, np.maximum(counts - 1.0, 1.0), out=values)
    np.subtract(values, 1.0, out=values)
    values[counts == 1.0] = 0.0
    return pd.DataFrame(values, index=ranks.index, columns=ranks.columns, copy=False)


def _cross_sectional_zscore(series: pd.Series, date_level: int | str) -> pd.Series:
    """Standardize each value to zero mean and unit standard deviation per period.

    The mean and (sample) standard deviation are taken over each period's finite
    values. A period whose values have zero or undefined standard deviation maps to
    ``0``; missing values stay missing so they are dropped downstream.
    """
    grouped = series.groupby(level=date_level)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    standardized = (series - mean) / std
    # Only values that are actually present collapse to zero. Writing 0.0 across
    # the whole period instead — which is what a plain ``where`` on a NaN standard
    # deviation does — turns a period with no returns into a full cross-section of
    # fabricated zeros, and the finiteness check downstream then admits it as a
    # training example.
    degenerate = (std.isna() | (std <= 0.0)) & series.notna()
    return standardized.mask(degenerate, 0.0)


def _grouped(frame: pd.DataFrame, cfg: DataConfig) -> dict[object, pd.DataFrame]:
    if cfg.standardize:
        frame = _cross_sectional_rank(frame, cfg.date_level)
    frame = frame.fillna(0.0)
    return dict(iter(frame.groupby(level=cfg.date_level, sort=True)))


def _cross_sections(
    panel: pd.DataFrame,
    cfg: DataConfig,
    beta_columns: Sequence[str],
    portfolio_columns: Sequence[str],
) -> CrossSectionDataset:
    """Build the per-period cross-sections of one split's feature panel."""
    if len(panel) == 0:
        return CrossSectionDataset(())
    # Rank the union once. The portfolio columns are normally a subset of the beta
    # columns, and ranking the two sets separately re-ranks every shared column a
    # second time — over a whole merged panel that is the most expensive step in
    # this module, doubled. The rank is taken per column within a date, so a column
    # selected out of the ranked union is identical to one ranked on its own.
    shared = list(dict.fromkeys([*beta_columns, *portfolio_columns]))
    by_date = _grouped(panel.loc[:, shared].astype("float64"), cfg)
    beta_names = list(beta_columns)
    portfolio_names = list(portfolio_columns)
    returns = panel.loc[:, cfg.return_column].astype("float64")
    if cfg.standardize:
        returns = _cross_sectional_zscore(returns, cfg.date_level)
    ret_by_date = dict(iter(returns.groupby(level=cfg.date_level)))

    dates = tuple(sorted(panel.index.get_level_values(cfg.date_level).unique()))
    sections: list[CrossSection] = []
    for date in dates:
        ret_group = ret_by_date[date]
        finite = np.isfinite(ret_group.to_numpy())
        if int(finite.sum()) < cfg.min_cross_section:
            continue
        period = by_date[date].reindex(ret_group.index)
        beta_group = period.loc[:, beta_names]
        port_group = period.loc[:, portfolio_names]
        entities = ret_group.index.get_level_values(cfg.entity_level)
        portfolio_inputs = torch.tensor(port_group.to_numpy()[finite], dtype=torch.float32)
        returns_tensor = torch.tensor(ret_group.to_numpy()[finite], dtype=torch.float32)
        sections.append(
            CrossSection(
                # Position among the sections actually produced, so it indexes the
                # sequence a fold slices. A period dropped for being too thin
                # leaves no gap.
                date_index=len(sections),
                date=pd.Timestamp(date),
                beta_inputs=torch.tensor(beta_group.to_numpy()[finite], dtype=torch.float32),
                portfolio_inputs=portfolio_inputs,
                returns=returns_tensor,
                # Solved here, once: the managed portfolios depend only on this
                # period's characteristics and returns, never on the model.
                portfolios=ConditionalAutoencoder.managed_portfolios(
                    portfolio_inputs, returns_tensor
                ),
                entities=tuple(str(e) for e in entities[finite]),
            )
        )
    return CrossSectionDataset(tuple(sections))


def resolve_column_sets(
    available: Iterable[Any], cfg: DataConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The beta and portfolio column sets, in the order the networks read them.

    ``available`` is any iterable of column labels — a panel's ``columns`` index or
    a split's ``feature_columns`` tuple — and is stringified on the way in.

    The beta network uses every feature column except the return and anything named
    in ``exclude_columns``; ``portfolio_columns`` select the (typically smaller) set
    used to form the managed portfolios, defaulting to all beta columns when left
    empty.

    Because the beta set is defined by exclusion, every column the feature stage
    emits reaches the model unless it is named. That is only safe while the target
    is the panel's only forward-looking column — list any other one in
    ``exclude_columns``.

    The *order* is part of the answer, not an incidental detail: input dimension
    ``p`` of the beta network means "the p-th name returned here" and nothing else,
    so anything that interprets the model per feature has to be handed this tuple
    alongside the weights.
    """
    names = [str(c) for c in available]
    if cfg.return_column not in names:
        raise KeyError(f"return_column {cfg.return_column!r} not found in the feature panel.")
    unknown = sorted(set(cfg.exclude_columns) - set(names))
    if unknown:
        raise KeyError(f"exclude_columns not found in the feature panel: {unknown}")
    excluded = {cfg.return_column, *cfg.exclude_columns}
    beta_columns = tuple(c for c in names if c not in excluded)
    if not beta_columns:
        raise ValueError("no beta columns remain after excluding the target and exclude_columns.")
    portfolio_columns = (
        _resolve_columns(names, cfg.portfolio_columns, cfg.return_column, "portfolio_columns")
        if cfg.portfolio_columns
        else beta_columns
    )
    return beta_columns, portfolio_columns


def build_cross_section_panel(panel: pd.DataFrame, cfg: DataConfig) -> CrossSectionPanel:
    """Build every usable period of ``panel`` once, for a fold schedule to slice.

    The counterpart to :func:`build_panel_splits` for a run with more than one
    fold. Because each period is standardized and solved independently, the
    sections this returns are the same objects either route would produce — see
    the module docstring.
    """
    beta_columns, portfolio_columns = resolve_column_sets(panel.columns, cfg)
    sections = _cross_sections(panel, cfg, beta_columns, portfolio_columns)
    if len(sections) == 0:
        raise ValueError(
            "the panel has no complete cross-sections; check the beta/portfolio/return "
            "columns and min_cross_section against the panel's cross-section sizes."
        )
    return CrossSectionPanel(
        sections=tuple(sections[i] for i in range(len(sections))),
        dates=pd.Index([sections[i].date for i in range(len(sections))]),
        beta_columns=beta_columns,
        portfolio_columns=portfolio_columns,
    )


def build_panel_splits(splits: DataSplits, cfg: DataConfig) -> PanelSplits:
    """Build train/val/test cross-section datasets from the adapter's ``DataSplits``.

    The single-split path. See :func:`resolve_column_sets` for how the two column
    sets are chosen.
    """
    beta_columns, portfolio_columns = resolve_column_sets(splits.feature_columns, cfg)
    train = _cross_sections(splits.train, cfg, beta_columns, portfolio_columns)
    if len(train) == 0:
        raise ValueError(
            "the train split has no complete cross-sections; check the beta/portfolio/return "
            "columns, min_cross_section, and the split dates against feature burn-in."
        )
    return PanelSplits(
        train=train,
        val=_cross_sections(splits.val, cfg, beta_columns, portfolio_columns),
        test=_cross_sections(splits.test, cfg, beta_columns, portfolio_columns),
        beta_columns=beta_columns,
        portfolio_columns=portfolio_columns,
    )
