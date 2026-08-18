"""Tests for :mod:`nekron.data_adapter.splitting` — the fold schedule layer.

A fold is arithmetic over positions, so most of what follows needs nothing but a
date index: how many folds a sample admits, where the purged periods land, and
which schedules are refused before any data is read. The panel fixtures appear
only in the last section, where the question is whether a schedule and the frames
sliced from it agree — and whether the one-fold ``single`` schedule still returns
exactly what ``build_datasets`` has always returned.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nekron.align.config import AlignmentConfig
from nekron.cache import CacheConfig
from nekron.data.config import IngestionConfig
from nekron.data_adapter import (
    AdapterError,
    DataAdapterConfig,
    SplitConfig,
    WalkForwardConfig,
    assemble_panel,
    build_datasets,
    build_folds,
    split_panel,
)
from nekron.data_adapter.splitting import (
    FoldBounds,
    Segment,
    generate_folds,
    plan_folds,
    resolve_cut_dates,
)

Ingestion = Callable[..., IngestionConfig]

# The twenty-date panel section D builds, and the two entities it carries.
NUM_DATES = 20
ENTITIES = (10001, 10002)


def business_days(count: int, start: str = "2020-01-01") -> pd.Index:
    """``count`` consecutive business days: the sequence a fold's positions index."""
    return pd.Index(pd.bdate_range(start, periods=count))


def walk_forward(**overrides: Any) -> WalkForwardConfig:
    """A small sweep — 5 train, 2 val, 2 test — since the defaults are five years."""
    params: dict[str, Any] = {"train_size": 5, "val_size": 2, "test_size": 2}
    return WalkForwardConfig(**(params | overrides))


def covered(fold: FoldBounds) -> set[int]:
    """Every position the fold assigns to one of its three segments."""
    return set(fold.train) | set(fold.val) | set(fold.test)


def dates_of(index: pd.Index, span: range) -> list[pd.Timestamp]:
    """The dates ``span`` covers, sliced positionally as the fold arithmetic means it."""
    return list(index[span.start : span.stop])


# --------------------------------------------------------------------------- #
# generate_folds: how many folds, and where
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("num_periods", "train", "val", "test", "purge", "step", "expected"),
    [
        (9, 5, 2, 2, 0, None, 1),  # span 9: an exact fit is one fold
        (10, 5, 2, 2, 0, None, 1),  # one short of a second fold
        (11, 5, 2, 2, 0, None, 2),  # the period that completes it
        (20, 5, 2, 2, 0, None, 6),
        (20, 5, 2, 2, 1, None, 5),  # two purges lengthen the span to 11
        (20, 5, 2, 2, 0, 1, 12),  # a step of one moves the fold one period
        (20, 5, 2, 2, 0, 3, 4),
        (20, 5, 2, 3, 0, None, 4),  # step defaults to test_size, so 3 here
        (12, 4, 3, 3, 1, None, 1),  # span 12: exactly the sample
        (100, 60, 12, 12, 5, None, 1),  # span 94, step 12: no room for a second
    ],
)
def test_fold_count_follows_the_span_and_step(
    num_periods: int, train: int, val: int, test: int, purge: int, step: int | None, expected: int
) -> None:
    """``floor((num_periods - span) / step) + 1``, with ``span`` including both purges.

    Every one of these numbers is a decision about how much out-of-sample evidence
    a run produces, so they are pinned rather than recomputed from the formula the
    implementation itself uses.
    """
    cfg = walk_forward(train_size=train, val_size=val, test_size=test, purge=purge, step=step)
    assert len(generate_folds(num_periods, cfg)) == expected


def test_rolling_train_windows_are_fixed_length_and_advance_by_the_step() -> None:
    cfg = walk_forward(step=3)
    folds = generate_folds(30, cfg)

    assert len(folds) > 1
    assert [len(fold.train) for fold in folds] == [5] * len(folds)
    assert [fold.train.start for fold in folds] == [3 * k for k in range(len(folds))]


def test_expanding_train_windows_stay_anchored_and_grow_by_the_step() -> None:
    cfg = walk_forward(mode="expanding", step=3)
    folds = generate_folds(30, cfg)

    assert len(folds) > 1
    assert [fold.train.start for fold in folds] == [0] * len(folds)
    assert [len(fold.train) for fold in folds] == [5 + 3 * k for k in range(len(folds))]


def test_expanding_places_validation_and_test_exactly_where_rolling_does() -> None:
    """The two modes differ only in where the train window starts.

    Scores from a rolling sweep and an expanding one are compared against each
    other; that comparison is meaningless the moment the two are scored on
    different dates, so the invariant is asserted directly rather than implied by
    the two mode tests above.
    """
    rolling = generate_folds(40, walk_forward(purge=2, step=3))
    expanding = generate_folds(40, walk_forward(mode="expanding", purge=2, step=3))

    assert len(rolling) == len(expanding)
    for slid, grown in zip(rolling, expanding, strict=True):
        assert grown.val == slid.val
        assert grown.test == slid.test
        assert grown.train.stop == slid.train.stop


def test_purged_positions_belong_to_no_segment() -> None:
    """The purge is what keeps a forward-return label out of the next segment.

    A gap that is merely narrower than configured, or one whose positions are
    still handed to a segment, leaks the label it exists to withhold.
    """
    cfg = walk_forward(train_size=6, val_size=3, test_size=3, purge=4)
    folds = generate_folds(40, cfg)

    assert folds
    for fold in folds:
        after_train = set(range(fold.train.stop, fold.val.start))
        after_val = set(range(fold.val.stop, fold.test.start))
        assert len(after_train) == 4
        assert len(after_val) == 4
        assert not after_train & covered(fold)
        assert not after_val & covered(fold)
        # No position is claimed twice either, so the segments really partition.
        assert len(covered(fold)) == 6 + 3 + 3


def test_the_default_step_tiles_the_test_windows_edge_to_edge() -> None:
    """Stepping by ``test_size`` scores every date once: no overlap and no gap.

    That is what makes the folds concatenate into one continuous out-of-sample
    series, which is how the results table is read.
    """
    folds = generate_folds(40, walk_forward(train_size=8, val_size=3, test_size=4, purge=1))

    assert len(folds) > 1
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test.stop == later.test.start
        assert not set(earlier.test) & set(later.test)


def test_a_step_below_the_test_size_overlaps_the_test_windows() -> None:
    folds = generate_folds(40, walk_forward(train_size=8, val_size=3, test_size=4, step=2))

    assert len(folds) > 1
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test.start < earlier.test.stop
        assert set(earlier.test) & set(later.test)


def test_a_step_above_the_test_size_leaves_dates_untested() -> None:
    folds = generate_folds(40, walk_forward(train_size=8, val_size=3, test_size=4, step=6))

    assert len(folds) > 1
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test.start > earlier.test.stop


def test_max_folds_truncates_the_sweep_from_the_front() -> None:
    every = generate_folds(40, walk_forward(step=3))
    capped = generate_folds(40, walk_forward(step=3, max_folds=2))

    assert len(every) > 2
    assert capped == every[:2]


def test_a_sample_shorter_than_one_fold_names_the_span_it_needs() -> None:
    """The message has to say what is missing, not merely that something is."""
    cfg = walk_forward(purge=1)  # span 5 + 2 + 2 + 2 x 1 = 11

    with pytest.raises(AdapterError, match="spans 11 periods") as raised:
        generate_folds(10, cfg)

    assert "train 5 + val 2 + test 2 + 2 x purge 1" in str(raised.value)
    assert "only 10 are available" in str(raised.value)


def test_the_sweep_stops_before_a_partial_fold() -> None:
    """A short final fold would be scored on fewer periods than every other one.

    Leaving the tail unused is the choice that keeps the folds comparable, so the
    last fold must be full size and one more must not fit.
    """
    cfg = walk_forward(step=3)  # span 9
    folds = generate_folds(20, cfg)

    for fold in folds:
        assert fold.sizes == {"train": 5, "val": 2, "test": 2}
        assert fold.stop <= 20
    assert folds[-1].stop + 3 > 20


def test_fold_indices_are_consecutive_from_zero() -> None:
    folds = generate_folds(40, walk_forward(step=3))
    assert [fold.index for fold in folds] == list(range(len(folds)))


# --------------------------------------------------------------------------- #
# plan_folds: the two schemes over a real date index
# --------------------------------------------------------------------------- #


def test_the_single_scheme_yields_one_fold_that_partitions_every_position() -> None:
    """A single split leaves nothing out: it is the degenerate one-fold schedule."""
    index = business_days(50)

    (fold,) = plan_folds(index, SplitConfig())

    assert fold.index == 0
    assert fold.train.start == 0
    assert fold.train.stop == fold.val.start
    assert fold.val.stop == fold.test.start
    assert fold.test.stop == len(index)
    assert covered(fold) == set(range(len(index)))


def test_explicit_cut_dates_reproduce_the_boolean_mask_semantics() -> None:
    """``date <= train_end`` -> train, ``date > val_end`` -> test, as it always was.

    The single split moved from masks to positions; a boundary that shifted by one
    date would silently move a date from train into validation on every existing
    model, so it is checked against the masks it replaced.
    """
    index = business_days(60)
    train_end = pd.Timestamp("2020-02-14")
    val_end = pd.Timestamp("2020-03-06")
    cfg = SplitConfig(train_end="2020-02-14", val_end="2020-03-06")

    (fold,) = plan_folds(index, cfg)

    assert dates_of(index, fold.train) == list(index[index <= train_end])
    assert dates_of(index, fold.val) == list(index[(index > train_end) & (index <= val_end)])
    assert dates_of(index, fold.test) == list(index[index > val_end])


def test_a_cut_date_the_sample_does_not_carry_still_cuts_between_two_dates() -> None:
    """A weekend cut date belongs to no row, and must not be rounded onto one."""
    index = business_days(30)
    cfg = SplitConfig(train_end="2020-01-11", val_end="2020-01-19")  # a Saturday, a Sunday

    (fold,) = plan_folds(index, cfg)

    assert dates_of(index, fold.train)[-1] == pd.Timestamp("2020-01-10")
    assert dates_of(index, fold.val)[0] == pd.Timestamp("2020-01-13")
    assert dates_of(index, fold.val)[-1] == pd.Timestamp("2020-01-17")
    assert dates_of(index, fold.test)[0] == pd.Timestamp("2020-01-20")


def test_fractional_cuts_land_on_the_quantile_dates() -> None:
    """With no cut dates the fractions pick positions, and the cut date joins train."""
    index = business_days(50)
    cfg = SplitConfig(train_fraction=0.6, val_fraction=0.8)

    train_end, val_end = resolve_cut_dates(index, cfg)
    (fold,) = plan_folds(index, cfg)

    assert train_end == index[int(len(index) * cfg.train_fraction)]
    assert val_end == index[int(len(index) * cfg.val_fraction)]
    # side="right": the date a cut names is the last date of the earlier segment.
    assert fold.train.stop == int(len(index) * cfg.train_fraction) + 1
    assert fold.val.stop == int(len(index) * cfg.val_fraction) + 1


def test_a_cut_at_the_last_date_leaves_an_empty_test_segment() -> None:
    """Reachable, not hypothetical — which is why an empty segment has to resolve."""
    index = pd.Index(pd.to_datetime([f"2020-01-{day:02d}" for day in range(1, 11)]))
    cfg = SplitConfig(train_end="2020-01-05", val_end="2020-01-10")

    (fold,) = plan_folds(index, cfg)

    assert fold.test == range(10, 10)
    assert fold.window(index).test == Segment(start=None, end=None, size=0)


def test_the_walk_forward_scheme_ignores_the_cut_dates() -> None:
    index = business_days(40)
    cfg = SplitConfig(
        scheme="walk_forward",
        train_end="2020-01-10",
        val_end="2020-01-20",
        walk_forward=walk_forward(),
    )

    folds = plan_folds(index, cfg)

    assert len(folds) == len(generate_folds(len(index), cfg.walk_forward))
    assert folds[0].train == range(0, 5)


@pytest.mark.parametrize(
    ("scheme", "mode"),
    [("single", "rolling"), ("walk_forward", "rolling"), ("walk_forward", "expanding")],
)
def test_burn_in_shifts_the_whole_schedule_forward(scheme: str, mode: str) -> None:
    """Burned-in dates leave the sample, they are not merely withheld from train.

    Positions are read against the *full* date index downstream, so a schedule cut
    over the surviving dates has to be re-based; forgetting the shift puts every
    boundary ``burn_in`` dates too early. An expanding window is the case that
    would go unnoticed longest, since its anchor is a literal ``0``.
    """
    index = business_days(60)
    cfg = SplitConfig(
        scheme=scheme,
        burn_in=7,
        walk_forward=walk_forward(mode=mode, train_size=10, val_size=4, test_size=4),
    )

    folds = plan_folds(index, cfg)

    assert folds == tuple(fold.shift(7) for fold in plan_folds(index[7:], replace(cfg, burn_in=0)))
    assert folds[0].start == 7
    assert min(min(covered(fold)) for fold in folds) == 7
    assert max(max(covered(fold)) for fold in folds) < len(index)


def test_a_burn_in_that_consumes_the_sample_is_refused() -> None:
    with pytest.raises(AdapterError, match="discards the whole sample of 10 dates"):
        plan_folds(business_days(10), SplitConfig(burn_in=10))


def test_unsorted_dates_are_refused() -> None:
    """Position arithmetic over an unsorted index is wrong in a way nothing reports."""
    index = pd.Index(pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-06"]))

    with pytest.raises(AdapterError, match="must be sorted ascending"):
        plan_folds(index, SplitConfig())


def test_duplicated_dates_are_refused() -> None:
    index = pd.Index(pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]))

    with pytest.raises(AdapterError, match="must be unique"):
        plan_folds(index, SplitConfig())


def test_an_empty_date_index_is_refused() -> None:
    with pytest.raises(AdapterError, match="empty set of dates"):
        plan_folds(pd.Index([], dtype="datetime64[ns]"), SplitConfig())


def test_a_train_end_after_val_end_is_refused() -> None:
    cfg = SplitConfig(train_end="2020-02-01", val_end="2020-01-10")

    with pytest.raises(AdapterError, match="must not exceed val_end"):
        plan_folds(business_days(40), cfg)

    with pytest.raises(AdapterError, match="must not exceed val_end"):
        resolve_cut_dates(business_days(40), cfg)


def test_a_train_end_before_the_sample_is_refused() -> None:
    """An empty train set is a misconfigured cut date, not a schedule worth running."""
    cfg = SplitConfig(train_end="1999-01-01", val_end="2020-01-10")

    with pytest.raises(AdapterError, match="train split is empty"):
        plan_folds(business_days(40), cfg)


# --------------------------------------------------------------------------- #
# FoldBounds, FoldWindow and Segment
# --------------------------------------------------------------------------- #


def test_window_resolves_positions_to_the_first_and_last_date_of_each_range() -> None:
    index = business_days(20)
    bounds = FoldBounds(index=3, train=range(0, 8), val=range(9, 12), test=range(13, 17))

    window = bounds.window(index)

    assert window.index == 3
    assert (window.train.start, window.train.end, window.train.size) == (index[0], index[7], 8)
    assert (window.val.start, window.val.end, window.val.size) == (index[9], index[11], 3)
    assert (window.test.start, window.test.end, window.test.size) == (index[13], index[16], 4)


def test_an_empty_segment_reports_no_dates_at_all() -> None:
    """The single scheme can hand a reporter an empty val or test segment.

    Inventing a boundary date for it would put a date in a results table that no
    row of that segment ever had, because the segment has no rows.
    """
    index = business_days(10)
    bounds = FoldBounds(index=0, train=range(0, 10), val=range(10, 10), test=range(10, 10))

    window = bounds.window(index)

    assert window.val == Segment(start=None, end=None, size=0)
    assert window.test == Segment(start=None, end=None, size=0)
    assert window.to_dict()["val_start"] == ""
    assert window.to_dict()["val_end"] == ""
    assert window.to_dict()["val_size"] == "0"


def test_to_dict_is_all_strings_with_one_key_set_however_empty_the_fold_is() -> None:
    """Ragged rows are what a params table cannot have, whatever a fold looks like."""
    index = business_days(10)
    full = FoldBounds(index=0, train=range(0, 4), val=range(4, 7), test=range(7, 10))
    hollow = FoldBounds(index=1, train=range(0, 10), val=range(10, 10), test=range(10, 10))

    flat = full.window(index).to_dict()
    empty = hollow.window(index).to_dict()

    assert set(flat) == {
        "fold_index",
        "train_start",
        "train_end",
        "train_size",
        "val_start",
        "val_end",
        "val_size",
        "test_start",
        "test_end",
        "test_size",
    }
    assert set(flat) == set(empty)
    assert all(isinstance(value, str) for value in flat.values())
    assert all(isinstance(value, str) for value in empty.values())
    assert flat["fold_index"] == "0"
    assert flat["train_start"] == "2020-01-01"
    assert flat["test_end"] == "2020-01-14"


def test_describe_names_the_fold_and_its_three_ranges() -> None:
    index = business_days(10)
    bounds = FoldBounds(index=2, train=range(0, 4), val=range(4, 7), test=range(7, 10))

    line = bounds.window(index).describe()

    assert line.startswith("fold 2: ")
    assert "train 2020-01-01..2020-01-06 (4)" in line
    assert "test 2020-01-10..2020-01-14 (3)" in line


def test_shift_moves_all_three_ranges_and_keeps_the_index() -> None:
    bounds = FoldBounds(index=2, train=range(0, 5), val=range(6, 8), test=range(9, 11))

    moved = bounds.shift(4)

    assert moved.index == 2
    assert moved.train == range(4, 9)
    assert moved.val == range(10, 12)
    assert moved.test == range(13, 15)
    assert moved.sizes == bounds.sizes
    assert (moved.start, moved.stop) == (4, 15)
    assert bounds.shift(0) is bounds


def test_a_window_resolved_against_a_shorter_date_index_is_refused() -> None:
    """A fold cut over the surviving periods, read against every date, reports lies.

    The length check turns that into an error instead of boundaries that are
    plausible and wrong.
    """
    bounds = FoldBounds(index=0, train=range(0, 5), val=range(5, 7), test=range(7, 9))

    with pytest.raises(AdapterError, match="ends at position 9 but only 8 dates"):
        bounds.window(business_days(8))


def test_a_strided_range_is_refused_at_construction() -> None:
    """Every consumer slices these ranges as contiguous spans of dates."""
    with pytest.raises(AdapterError, match="the val range must be contiguous"):
        FoldBounds(index=0, train=range(0, 5), val=range(5, 9, 2), test=range(9, 11))


def test_a_range_starting_before_the_sample_is_refused() -> None:
    with pytest.raises(AdapterError, match="the train range starts before the sample"):
        FoldBounds(index=0, train=range(-1, 5), val=range(5, 7), test=range(7, 9))


# --------------------------------------------------------------------------- #
# FoldPlan: the schedule against a real panel
# --------------------------------------------------------------------------- #


def _prices_csv(dates: pd.DatetimeIndex) -> str:
    """The conftest price source rewritten over a longer calendar."""
    rows = ["PERMNO,DlyCalDt,DlyClose,DlyVol"]
    for entity in ENTITIES:
        rows += [
            f"{entity},{date.strftime('%d/%m/%Y')},{10.0 + position},{100 + position}"
            for position, date in enumerate(dates)
        ]
    return "\n".join(rows) + "\n"


@pytest.fixture
def long_prices(sources: dict[str, Path]) -> pd.Index:
    """Rewrite the price source over twenty business days and return those dates.

    The conftest panel is four dates long, which no walk-forward window with a
    purge fits inside; a schedule needs a sample it can actually sweep.
    """
    dates = pd.bdate_range("2020-01-01", periods=NUM_DATES)
    sources["prices"].write_text(_prices_csv(dates), encoding="utf-8")
    return pd.Index(dates)


def _adapter(ingestion_config: IngestionConfig, split: SplitConfig) -> DataAdapterConfig:
    """The smallest pipeline that reaches the split: one panel, no joins, no cache."""
    return DataAdapterConfig(
        ingestion=ingestion_config,
        alignment=AlignmentConfig(spine="prices", joins=[]),
        split=split,
        cache=CacheConfig(enabled=False),
    )


def _swept() -> SplitConfig:
    """Five folds over twenty dates: span 11, stepped by the test size of 2."""
    return SplitConfig(
        scheme="walk_forward",
        walk_forward=WalkForwardConfig(train_size=5, val_size=2, test_size=2, purge=1),
    )


def test_build_folds_under_the_single_scheme_reproduces_build_datasets(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    """The backward-compatibility guarantee: a single split is fold zero, unchanged.

    Every existing model still calls ``build_datasets``; if the one-fold schedule
    disagreed with it by so much as a row, the fold layer would have silently
    changed what those models train on.
    """
    cfg = _adapter(ingestion("prices"), SplitConfig(train_fraction=0.5, val_fraction=0.75))

    plan = build_folds(cfg)
    expected = build_datasets(cfg)
    actual = plan.splits(0)

    assert len(plan) == 1
    pd.testing.assert_frame_equal(actual.train, expected.train)
    pd.testing.assert_frame_equal(actual.val, expected.val)
    pd.testing.assert_frame_equal(actual.test, expected.test)
    assert actual.feature_columns == expected.feature_columns == ("DlyClose", "DlyVol")
    assert actual.train_end == expected.train_end
    assert actual.val_end == expected.val_end


def test_build_folds_sweeps_the_configured_window_across_the_panel(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    cfg = _adapter(ingestion("prices"), _swept())

    plan = build_folds(cfg)

    assert len(plan) == 5
    assert list(plan.dates) == list(long_prices)
    for position, fold in enumerate(plan):
        offset = 2 * position
        assert fold.train == range(offset, offset + 5)
        assert fold.val == range(offset + 6, offset + 8)
        assert fold.test == range(offset + 9, offset + 11)


def test_a_folds_frames_are_disjoint_and_omit_exactly_the_purged_dates(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    """The frames must honour the schedule the positions describe.

    Segments that overlap by a date, or a purged date that reaches a frame anyway,
    are the two ways the plan and the data it slices can disagree — and neither
    shows up as an error, only as an optimistic score.
    """
    cfg = _adapter(ingestion("prices"), _swept())
    plan = build_folds(cfg)

    for position, fold in enumerate(plan):
        splits = plan.splits(position)
        train, val, test = (
            set(frame.index.get_level_values("date"))
            for frame in (splits.train, splits.val, splits.test)
        )
        purged = {long_prices[fold.train.stop], long_prices[fold.val.stop]}

        assert len(train) == 5
        assert len(val) == 2
        assert len(test) == 2
        assert not train & val
        assert not val & test
        assert not train & test
        assert not purged & (train | val | test)
        assert train | val | test | purged == set(long_prices[fold.start : fold.stop])
        # Both entities survive the slice: the mask is over dates, not rows.
        assert len(splits.train) == 5 * len(ENTITIES)


def test_burn_in_keeps_the_leading_dates_out_of_every_frame(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    """A burned-in date leaves the sample, so no frame of any fold may carry one.

    The schedule and the slicing read the same date index, and a burn_in that
    moved the positions without moving the rows would put the warm-up dates —
    whose long-window features are entirely missing — back into training.
    """
    cfg = _adapter(ingestion("prices"), SplitConfig(burn_in=5, train_fraction=0.5))
    plan = build_folds(cfg)

    splits = plan.splits(0)
    seen = {
        date
        for frame in (splits.train, splits.val, splits.test)
        for date in frame.index.get_level_values("date")
    }

    assert plan.bounds[0].start == 5
    assert seen == set(long_prices[5:])


def test_split_panel_refuses_a_schedule_of_more_than_one_fold(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    """A single train/val/test answer does not exist for a sweep, so none is invented."""
    cfg = _adapter(ingestion("prices"), _swept())
    panel = assemble_panel(cfg)

    with pytest.raises(AdapterError, match=r"Use build_folds\(\)") as raised:
        split_panel(panel, cfg.split)

    assert "produced 5 folds" in str(raised.value)


def test_windows_returns_one_window_per_fold_in_order(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    cfg = _adapter(ingestion("prices"), _swept())
    plan = build_folds(cfg)

    windows = plan.windows()

    assert len(windows) == len(plan)
    assert [window.index for window in windows] == list(range(len(plan)))
    for position, window in enumerate(windows):
        fold = plan.bounds[position]
        assert window == plan.window(position)
        assert window.train.start == long_prices[fold.train.start]
        assert window.train.end == long_prices[fold.train.stop - 1]
        assert window.test.start == long_prices[fold.test.start]
        assert window.test.end == long_prices[fold.test.stop - 1]


def test_a_walk_forward_fold_reports_the_last_date_of_each_segment_as_its_cut(
    ingestion: Ingestion, long_prices: pd.Index
) -> None:
    """A swept fold has no cut dates, so it reports the bound its positions imply."""
    cfg = _adapter(ingestion("prices"), _swept())
    plan = build_folds(cfg)

    for position, fold in enumerate(plan):
        splits = plan.splits(position)
        assert splits.train_end == long_prices[fold.train.stop - 1]
        assert splits.val_end == long_prices[fold.val.stop - 1]
        assert splits.train.index.get_level_values("date").max() <= splits.train_end
        assert splits.test.index.get_level_values("date").min() > splits.val_end


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "sliding"}, "split.walk_forward.mode must be one of"),
        ({"train_size": 0}, "split.walk_forward.train_size must be positive"),
        ({"val_size": 0}, "split.walk_forward.val_size must be positive"),
        ({"test_size": -1}, "split.walk_forward.test_size must be positive"),
        ({"purge": -1}, "split.walk_forward.purge must not be negative"),
        ({"step": 0}, "split.walk_forward.step must be positive"),
        ({"max_folds": 0}, "split.walk_forward.max_folds must be positive"),
    ],
)
def test_walk_forward_config_names_the_field_it_rejects(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        WalkForwardConfig(**overrides)


def test_walk_forward_config_accepts_its_boundary_values() -> None:
    """``purge=0``, ``step=1`` and ``max_folds=1`` are the smallest legal sweep."""
    cfg = WalkForwardConfig(train_size=1, val_size=1, test_size=1, purge=0, step=1, max_folds=1)

    assert generate_folds(3, cfg) == (
        FoldBounds(index=0, train=range(0, 1), val=range(1, 2), test=range(2, 3)),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scheme": "expanding"}, "split.scheme must be one of"),
        ({"burn_in": -1}, "split.burn_in must not be negative"),
        (
            {"train_fraction": 0.9, "val_fraction": 0.5},
            "require 0 < train_fraction <= val_fraction < 1",
        ),
        ({"train_fraction": 0.0}, "require 0 < train_fraction <= val_fraction < 1"),
        ({"val_fraction": 1.0}, "require 0 < train_fraction <= val_fraction < 1"),
    ],
)
def test_split_config_names_the_field_it_rejects(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        SplitConfig(**overrides)


def test_split_config_accepts_both_schemes_and_a_zero_burn_in() -> None:
    for scheme in ("single", "walk_forward"):
        assert SplitConfig(scheme=scheme, burn_in=0).scheme == scheme
