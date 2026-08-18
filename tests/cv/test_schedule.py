"""Tests for cutting a schedule over the periods a model can actually use.

The failure this module exists to prevent is silent: a boundary that is off by the
number of dates a model discarded produces a perfectly plausible sweep whose folds
train on slightly the wrong data, and no test of the model itself would notice. So
every assertion here compares a schedule cut over a *subsequence* against the
dates it should land on, rather than against a restatement of the arithmetic.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from nekron.cv import plan_schedule
from nekron.data_adapter import SplitConfig, WalkForwardConfig
from nekron.data_adapter.splitting import plan_folds

PANEL = pd.Index(pd.date_range("2020-01-01", periods=60, freq="B"))


def surviving(drop: set[int]) -> pd.Index:
    """The panel minus some positions, as a model that discards periods would see it."""
    return pd.Index([d for i, d in enumerate(PANEL) if i not in drop])


def cut_dates(bounds: tuple, dates: pd.Index) -> list[tuple]:
    return [
        (dates[b.train.start], dates[b.train.stop - 1], dates[b.test.start], dates[b.test.stop - 1])
        for b in bounds
    ]


# --------------------------------------------------------------------------- #
# The whole panel: plan_schedule must agree with plan_folds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "split",
    [
        SplitConfig(scheme="single"),
        SplitConfig(scheme="single", train_end="2020-02-14", val_end="2020-03-06"),
        SplitConfig(
            scheme="walk_forward",
            walk_forward=WalkForwardConfig(train_size=20, val_size=6, test_size=6),
        ),
    ],
)
def test_nothing_discarded_is_the_plain_schedule(split: SplitConfig) -> None:
    assert plan_schedule(PANEL, PANEL, split) == plan_folds(PANEL, split)


# --------------------------------------------------------------------------- #
# A subsequence: the boundaries must not move
# --------------------------------------------------------------------------- #


def test_an_explicit_single_split_cuts_at_the_same_dates_either_way() -> None:
    """The cut is a timestamp, so discarding periods must not move it."""
    split = SplitConfig(scheme="single", train_end="2020-02-14", val_end="2020-03-06")
    kept = surviving({5, 17, 40})
    (fold,) = plan_schedule(kept, PANEL, split)

    assert kept[fold.train.stop - 1] <= pd.Timestamp("2020-02-14")
    assert kept[fold.train.stop] > pd.Timestamp("2020-02-14")
    assert kept[fold.val.stop - 1] <= pd.Timestamp("2020-03-06")
    assert kept[fold.val.stop] > pd.Timestamp("2020-03-06")


def test_a_fraction_split_lands_on_the_panels_date_not_the_subsequences() -> None:
    """The reason the panel is passed at all: a fraction of a subsequence is a
    different date than the same fraction of the panel."""
    split = SplitConfig(scheme="single", train_fraction=0.6, val_fraction=0.8)
    kept = surviving(set(range(0, 30)))  # half the panel gone, all from the front

    (rebased,) = plan_schedule(kept, PANEL, split)
    (naive,) = plan_folds(kept, split)

    panel_cut = PANEL[int(len(PANEL) * 0.6)]
    # The cut is inclusive -- ``date <= train_end`` goes to train -- so the panel's
    # own cut date is the last training date, not the first validation one.
    assert kept[rebased.train.stop - 1] == panel_cut
    assert kept[rebased.train.stop] > panel_cut
    # Cutting the subsequence by its own fractions would land somewhere else.
    assert naive.train.stop != rebased.train.stop


def test_positions_index_the_surviving_sequence() -> None:
    """A fold's positions must be usable against the sequence it was cut over."""
    kept = surviving({3, 11, 29})
    split = SplitConfig(
        scheme="walk_forward",
        walk_forward=WalkForwardConfig(train_size=15, val_size=5, test_size=5),
    )
    for fold in plan_schedule(kept, PANEL, split):
        assert fold.stop <= len(kept)
        window = fold.window(kept)
        assert window.train.start == kept[fold.train.start]
        assert window.test.end == kept[fold.test.stop - 1]


def test_a_burn_in_is_converted_into_the_surviving_sequences_units() -> None:
    """``burn_in`` counts *panel* dates; applying that integer to a shorter
    sequence would discard more history than was asked for."""
    split = SplitConfig(scheme="single", train_end="2020-03-06", val_end="2020-03-20", burn_in=10)
    kept = surviving({1, 2, 3, 4})  # four of the first ten panel dates are gone

    (fold,) = plan_schedule(kept, PANEL, split)
    # The fold starts at the first surviving date at or after panel date 10 ...
    assert kept[fold.train.start] >= PANEL[10]
    # ... which is fewer than 10 positions into the surviving sequence.
    assert fold.train.start < 10


def test_walk_forward_ignores_the_cut_dates_entirely() -> None:
    split = SplitConfig(
        scheme="walk_forward",
        train_end="2020-01-10",
        val_end="2020-01-20",
        walk_forward=WalkForwardConfig(train_size=20, val_size=6, test_size=6),
    )
    kept = surviving({7})
    assert plan_schedule(kept, PANEL, split) == plan_folds(kept, split)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_discarded_dates_are_logged_with_what_they_failed_to_produce(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="nekron.cv.schedule"):
        plan_schedule(surviving({1, 2, 3}), PANEL, SplitConfig(), what="complete window")
    assert "3 of 60 dates produced no complete window" in caplog.text


def test_nothing_is_logged_when_nothing_was_discarded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="nekron.cv.schedule"):
        plan_schedule(PANEL, PANEL, SplitConfig())
    assert not caplog.records


# --------------------------------------------------------------------------- #
# Import cost
# --------------------------------------------------------------------------- #


def test_the_policy_can_be_imported_without_torch_pandas_or_mlflow() -> None:
    """The property `nekron.cv.config` is written for, in a fresh interpreter.

    Checked in a subprocess because this one has already imported everything. The
    driver is resolved lazily precisely so that this stays true; importing the
    package eagerly would make it false without any test noticing.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "from nekron.cv.config import CrossValidationConfig;"
        "assert CrossValidationConfig().seed_mode == 'fixed';"
        "heavy = [m for m in ('torch', 'pandas', 'mlflow') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"heavy modules imported: {result.stdout.strip()}"


def test_the_driver_is_still_reachable_from_the_package() -> None:
    """Laziness must not cost the ordinary spelling."""
    from nekron import cv

    assert callable(cv.run_sweep)
    assert callable(cv.plan_schedule)
    assert not [name for name in cv.__all__ if not hasattr(cv, name)]


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    from nekron import cv

    with pytest.raises(AttributeError, match="no attribute 'not_a_thing'"):
        cv.not_a_thing  # noqa: B018
