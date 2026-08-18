"""Fold schedules: how a panel's ordered dates are cut into train/validation/test.

Two schemes share one representation. ``single`` cuts the dates once at two
timestamps — the historical behaviour, and still the default. ``walk_forward``
sweeps a window across the sample and refits at every position, so a model is
scored on several disjoint out-of-sample windows instead of one. A single split is
the degenerate one-fold schedule, which is why both produce the same
:class:`FoldBounds` and no consumer downstream has to know which scheme ran.

Folds are positions, not timestamps
-----------------------------------
A :class:`FoldBounds` holds three half-open ``range`` objects into a sequence of
dates the caller supplies, and the caller decides which sequence that is. The
distinction is the one thing about this module that is easy to get wrong: a model
that discards periods — one whose cross-section is too small to price, say — must
cut its folds over *the periods that survive*, or every boundary after the first
discarded date is off by the number discarded. Position arithmetic over the
surviving sequence is also conservative with respect to purging, since a gap of
``n`` surviving periods spans at least ``n`` periods of the original calendar.

Nothing here reads data. A schedule is a function of a date count and a
configuration, which is what makes the edge cases — the last partial fold, a
window longer than the sample, a purge that swallows a segment — testable without
building a panel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .base import AdapterError
from .config import SplitConfig, WalkForwardConfig


@dataclass(frozen=True)
class Segment:
    """One segment of a fold, as the dates it spans and the number of them.

    :attr:`start` and :attr:`end` are ``None`` for an empty segment. That is not a
    hypothetical: the single-split scheme admits an empty validation or test
    segment whenever a cut date sits at the end of the sample, and a caller
    reporting boundaries should say so rather than invent one.
    """

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    size: int


@dataclass(frozen=True)
class FoldWindow:
    """One fold's three segments, resolved against real dates, for logging.

    :class:`FoldBounds` is what the code slices with; this is what a run records —
    the form a boundary has to take before it can become an MLflow tag, a column
    of a results table or a line on a console.
    """

    index: int
    train: Segment
    val: Segment
    test: Segment

    def to_dict(self) -> dict[str, str]:
        """Flat, string-valued bounds, named for direct use as params or tags.

        Every value is a string because that is what both MLflow params and tags
        store; an empty segment yields an empty string rather than being dropped,
        so the set of keys is the same for every fold and the resulting table has
        no ragged rows.
        """
        flat: dict[str, str] = {"fold_index": str(self.index)}
        for name, segment in (("train", self.train), ("val", self.val), ("test", self.test)):
            flat[f"{name}_start"] = _iso(segment.start)
            flat[f"{name}_end"] = _iso(segment.end)
            flat[f"{name}_size"] = str(segment.size)
        return flat

    def describe(self) -> str:
        """One line naming the fold and its three date ranges."""
        return (
            f"fold {self.index}: "
            f"train {_iso(self.train.start)}..{_iso(self.train.end)} ({self.train.size}) | "
            f"val {_iso(self.val.start)}..{_iso(self.val.end)} ({self.val.size}) | "
            f"test {_iso(self.test.start)}..{_iso(self.test.end)} ({self.test.size})"
        )


@dataclass(frozen=True)
class FoldBounds:
    """One fold as three half-open position ranges into an ordered date sequence.

    The ranges are contiguous and non-overlapping but need not be adjacent: under
    walk-forward the purged periods sit in the gaps between them and belong to no
    segment.
    """

    index: int
    train: range
    val: range
    test: range

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            span: range = getattr(self, name)
            if span.step != 1:
                raise AdapterError(
                    f"fold {self.index}: the {name} range must be contiguous; got step {span.step}."
                )
            if span.start < 0:
                raise AdapterError(
                    f"fold {self.index}: the {name} range starts before the sample "
                    f"(position {span.start})."
                )
            if span.stop < span.start:
                # Empty is fine — a single split may have no test segment — but
                # reversed is a construction error that would otherwise pass as
                # empty and take a whole segment silently out of the fold.
                raise AdapterError(
                    f"fold {self.index}: the {name} range ends before it starts "
                    f"({span.start} -> {span.stop})."
                )

    @property
    def start(self) -> int:
        """First position the fold touches."""
        return self.train.start

    @property
    def stop(self) -> int:
        """One past the last position the fold touches."""
        return self.test.stop

    @property
    def sizes(self) -> dict[str, int]:
        """Segment lengths, keyed by segment name."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def shift(self, offset: int) -> FoldBounds:
        """The same fold, re-based by ``offset`` positions."""
        if offset == 0:
            return self
        return FoldBounds(
            index=self.index,
            train=_shift(self.train, offset),
            val=_shift(self.val, offset),
            test=_shift(self.test, offset),
        )

    def window(self, dates: pd.Index) -> FoldWindow:
        """Resolve this fold's positions against ``dates``."""
        if self.stop > len(dates):
            raise AdapterError(
                f"fold {self.index} ends at position {self.stop} but only {len(dates)} "
                "dates were supplied; the fold was cut over a different sequence."
            )
        return FoldWindow(
            index=self.index,
            train=_segment(dates, self.train),
            val=_segment(dates, self.val),
            test=_segment(dates, self.test),
        )


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #


def generate_folds(num_periods: int, cfg: WalkForwardConfig) -> tuple[FoldBounds, ...]:
    """Sweep the configured window across ``num_periods`` positions.

    Fold ``k`` begins ``k * step`` positions into the sample. Under ``"rolling"``
    the train window moves with it; under ``"expanding"`` the train window stays
    anchored at position ``0`` and grows instead. Both place the validation and
    test windows identically, separated from what precedes them by
    :attr:`~WalkForwardConfig.purge` positions.

    Only whole folds are produced: the sweep stops at the last position where a
    complete train/validation/test window still fits, so a partial tail is left
    unused rather than silently scored on fewer periods than the others. The count
    is ``floor((num_periods - span) / step) + 1`` with ``span`` the total length of
    one fold including both purges.
    """
    span = cfg.train_size + cfg.val_size + cfg.test_size + 2 * cfg.purge
    if num_periods < span:
        raise AdapterError(
            f"a walk-forward fold spans {span} periods (train {cfg.train_size} + "
            f"val {cfg.val_size} + test {cfg.test_size} + 2 x purge {cfg.purge}) but only "
            f"{num_periods} are available; shorten a window, lower split.burn_in, or "
            "lengthen the sample."
        )
    step = cfg.step if cfg.step is not None else cfg.test_size
    folds = [
        _fold_at(index, offset, cfg)
        for index, offset in enumerate(range(0, num_periods - span + 1, step))
    ]
    if cfg.max_folds is not None:
        folds = folds[: cfg.max_folds]
    return tuple(folds)


def plan_folds(dates: pd.Index, cfg: SplitConfig) -> tuple[FoldBounds, ...]:
    """The fold schedule ``cfg`` describes over ``dates``.

    ``dates`` must be the sorted, unique dates of whatever sequence the caller will
    slice — see the module docstring on why that is the caller's choice to make.
    Returns one fold under ``"single"`` and as many as the sample admits under
    ``"walk_forward"``.
    """
    _require_ordered(dates)
    if cfg.burn_in >= len(dates):
        raise AdapterError(
            f"split.burn_in ({cfg.burn_in}) discards the whole sample of {len(dates)} "
            "dates; lower it."
        )
    usable = dates[cfg.burn_in :] if cfg.burn_in else dates
    bounds = (
        generate_folds(len(usable), cfg.walk_forward)
        if cfg.scheme == "walk_forward"
        else (_single_fold(usable, cfg),)
    )
    return tuple(fold.shift(cfg.burn_in) for fold in bounds)


def rebase_split(cfg: SplitConfig, panel_dates: pd.Index, target_dates: pd.Index) -> SplitConfig:
    """``cfg`` re-expressed so cutting ``target_dates`` means what cutting the panel meant.

    ``target_dates`` is a subsequence of ``panel_dates``: the periods a model can
    actually use, once the ones it cannot price have been dropped. Two settings are
    stated in units that do not survive that change of sequence, and both are
    converted here.

    :attr:`SplitConfig.burn_in` is a count of leading dates to discard, set from
    the longest trailing window a featurizer uses — a property of the panel's
    calendar, so it counts *panel* dates. The equivalent count over the target
    sequence is however many of its dates fall before the first panel date that
    survives the burn-in, which is generally a smaller number. Applying the same
    integer to both sequences would move the start of training.

    The ``single`` scheme's cut dates, when derived from fractions, are
    ``dates[int(len(dates) * fraction)]`` — a different date over a subsequence
    than over the panel. Resolving them against the panel and cutting the
    subsequence at *those timestamps* is what keeps a run's fold 0 identical to
    what :func:`~nekron.data_adapter.build_datasets` produces.

    A no-op when nothing needs converting: no burn-in, and either the walk-forward
    scheme or a single split that already names both dates.
    """
    if cfg.burn_in >= len(panel_dates):
        raise AdapterError(
            f"split.burn_in ({cfg.burn_in}) discards the whole panel of "
            f"{len(panel_dates)} dates; lower it."
        )
    burn_in = (
        int(target_dates.searchsorted(panel_dates[cfg.burn_in], side="left")) if cfg.burn_in else 0
    )
    return replace(pin_cut_dates(cfg, panel_dates), burn_in=burn_in)


def pin_cut_dates(cfg: SplitConfig, dates: pd.Index) -> SplitConfig:
    """``cfg`` with the single scheme's two cut dates resolved against ``dates``.

    Resolves the fractions to timestamps so that a later cut over a *different*
    sequence lands on the same calendar dates. It does **not** convert
    :attr:`SplitConfig.burn_in`, which is also sequence-dependent — use
    :func:`rebase_split` when the schedule will be cut over a subsequence, which is
    the usual case for a model that discards unusable periods.

    A no-op for ``walk_forward``, which is defined by positions and has no cut
    dates, and for a ``single`` config that already names both.
    """
    if cfg.scheme == "walk_forward" or (cfg.train_end and cfg.val_end):
        return cfg
    if cfg.burn_in >= len(dates):
        raise AdapterError(
            f"split.burn_in ({cfg.burn_in}) discards the whole sample of {len(dates)} "
            "dates; lower it."
        )
    train_end, val_end = resolve_cut_dates(dates[cfg.burn_in :] if cfg.burn_in else dates, cfg)
    # isoformat, not date(): truncating to the day would move the cut to midnight
    # on a panel whose periods are finer than daily.
    return replace(cfg, train_end=train_end.isoformat(), val_end=val_end.isoformat())


def resolve_cut_dates(dates: pd.Index, cfg: SplitConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The two timestamps a single split cuts at, from dates or from fractions."""
    train_end = (
        pd.Timestamp(cfg.train_end)
        if cfg.train_end
        else pd.Timestamp(dates[int(len(dates) * cfg.train_fraction)])
    )
    val_end = (
        pd.Timestamp(cfg.val_end)
        if cfg.val_end
        else pd.Timestamp(dates[int(len(dates) * cfg.val_fraction)])
    )
    if train_end > val_end:
        raise AdapterError(f"train_end ({train_end}) must not exceed val_end ({val_end}).")
    return train_end, val_end


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _fold_at(index: int, offset: int, cfg: WalkForwardConfig) -> FoldBounds:
    """One position of the sweep, with the purges placed at both inner boundaries."""
    train_stop = offset + cfg.train_size
    train_start = offset if cfg.mode == "rolling" else 0
    val_start = train_stop + cfg.purge
    val_stop = val_start + cfg.val_size
    test_start = val_stop + cfg.purge
    return FoldBounds(
        index=index,
        train=range(train_start, train_stop),
        val=range(val_start, val_stop),
        test=range(test_start, test_start + cfg.test_size),
    )


def _single_fold(dates: pd.Index, cfg: SplitConfig) -> FoldBounds:
    """The one fold a ``"single"`` schedule produces, cut at the two resolved dates."""
    train_end, val_end = resolve_cut_dates(dates, cfg)
    # side="right" counts the dates at or before the cut, which is the half of the
    # boundary the single split has always assigned to the earlier segment.
    train_stop = int(dates.searchsorted(train_end, side="right"))
    val_stop = int(dates.searchsorted(val_end, side="right"))
    if train_stop == 0:
        raise AdapterError("train split is empty; check split.train_end against the data dates.")
    return FoldBounds(
        index=0,
        train=range(0, train_stop),
        val=range(train_stop, val_stop),
        test=range(val_stop, len(dates)),
    )


def _require_ordered(dates: pd.Index) -> None:
    """Reject a date index the position arithmetic would silently misread."""
    if len(dates) == 0:
        raise AdapterError("cannot build a fold schedule over an empty set of dates.")
    if dates.hasnans:
        raise AdapterError(
            "the dates a fold schedule is cut over contain missing values; a row whose "
            "date is NaT belongs to no period and must be dropped upstream."
        )
    if not dates.is_monotonic_increasing:
        raise AdapterError("the dates a fold schedule is cut over must be sorted ascending.")
    if not dates.is_unique:
        raise AdapterError("the dates a fold schedule is cut over must be unique.")


def _shift(span: range, offset: int) -> range:
    return range(span.start + offset, span.stop + offset)


def _segment(dates: pd.Index, span: range) -> Segment:
    if len(span) == 0:
        return Segment(start=None, end=None, size=0)
    return Segment(
        start=pd.Timestamp(dates[span.start]),
        end=pd.Timestamp(dates[span.stop - 1]),
        size=len(span),
    )


def _iso(value: pd.Timestamp | None) -> str:
    return "" if value is None else str(pd.Timestamp(value).date())
