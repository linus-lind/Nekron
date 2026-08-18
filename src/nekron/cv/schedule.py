"""Cutting a fold schedule over the periods a model can actually use.

Every model discards some of the panel's dates before it can fit anything — the
conditional autoencoder drops a period whose cross-section is too small to price,
a sequence model drops one that ends no complete window — so the sequence a fold
indexes is shorter than the panel's own date index. :func:`plan_schedule` is the
one place that knows what to do about it, and both halves matter:

* folds are cut over the **surviving** sequence, because cutting them over the
  panel's dates would shift every boundary after the first discarded date;
* the split is **rebased against the panel first**, so a fraction-based single
  split lands on the same calendar date whichever sequence it is then applied to.
  Without that, ``scheme="single"`` silently disagrees with
  :func:`~nekron.data_adapter.build_datasets` about where the split is, and two
  models over the same panel disagree with each other.

Both are easy to half-implement — pinning the cut dates but not the burn-in, or
the reverse — and the result is a boundary that is off by the number of discarded
dates, which no test of the model itself would notice.
"""

from __future__ import annotations

import logging

import pandas as pd

from nekron.data_adapter.config import SplitConfig
from nekron.data_adapter.splitting import FoldBounds, plan_folds, rebase_split

logger = logging.getLogger(__name__)


def plan_schedule(
    period_dates: pd.Index,
    panel_dates: pd.Index,
    split: SplitConfig,
    *,
    what: str = "periods",
) -> tuple[FoldBounds, ...]:
    """The fold schedule ``split`` describes over the periods a model can use.

    Parameters
    ----------
    period_dates:
        The dates that survived into usable periods, ascending and unique. Fold
        positions index *this* sequence.
    panel_dates:
        Every date the panel carried, ascending and unique. Used only to rebase
        the split, never to cut it.
    split:
        The schedule. ``"single"`` yields one fold; ``"walk_forward"`` as many as
        the sample admits.
    what:
        What a discarded date failed to produce, for the log line — "periods with
        a priceable cross-section", "complete windows", and so on.
    """
    dropped = len(panel_dates) - len(period_dates)
    if dropped:
        logger.info(
            "%d of %d dates produced no %s; folds are cut over the %d that remain.",
            dropped,
            len(panel_dates),
            what,
            len(period_dates),
        )
    return plan_folds(period_dates, rebase_split(split, panel_dates, period_dates))
