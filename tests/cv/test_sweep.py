"""Tests for the shared cross-validation driver, against no model at all.

Every invariant here is exercised through a fake fold body of a dozen lines, which
is the point: the driver's contract is what it does *around* a model, and pinning
it through a real one buys nothing and costs a fitted network per assertion. The
two model packages have their own sweep tests for the parts that are theirs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nekron.cv import (
    CrossValidationConfig,
    FoldOutcome,
    FoldRecord,
    SweepError,
    SweepReport,
    aggregate_convergence,
    aggregate_spread,
    fold_checkpoint_dir,
    fold_seed,
    run_sweep,
    sweep_tags,
    unplaced_window,
    warn_on_overlap,
)
from nekron.data_adapter import SplitConfig, WalkForwardConfig
from nekron.data_adapter.splitting import FoldBounds
from nekron.tracking import MlflowConfig, MlflowTracker

DATES = pd.Index(pd.date_range("2020-01-01", periods=40, freq="B"))


def bounds(count: int = 3, *, size: int = 4) -> tuple[FoldBounds, ...]:
    """A tiled schedule: ``count`` folds of train/val/test blocks of ``size``."""
    return tuple(
        FoldBounds(
            index=i,
            train=range(i * size, i * size + size),
            val=range(i * size + size, i * size + 2 * size),
            test=range(i * size + 2 * size, i * size + 3 * size),
        )
        for i in range(count)
    )


def tracker() -> MlflowTracker:
    return MlflowTracker(MlflowConfig(enabled=False))


def rows_for(index: int, *, n: int = 4, total: float = 8.0) -> pd.DataFrame:
    return pd.DataFrame({"n": [n], "total": [total]})


def pool(rows: pd.DataFrame) -> Mapping[str, float]:
    """A ratio of sums, so pooled and averaged genuinely differ."""
    if not len(rows) or "n" not in rows.columns:
        return {"score": float("nan")}
    return {"score": float(rows["total"].sum()) / float(rows["n"].sum())}


def sweep(
    body: object,
    *,
    schedule: Sequence[FoldBounds] | None = None,
    policy: CrossValidationConfig | None = None,
    **kwargs: object,
) -> object:
    options: dict[str, object] = {"pool": pool, **kwargs}
    return run_sweep(
        schedule if schedule is not None else bounds(),
        DATES,
        body,  # type: ignore[arg-type]
        policy=policy or CrossValidationConfig(),
        tracker=tracker(),
        seed=7,
        **options,  # type: ignore[arg-type]
    )


def simple_body(index_scores: dict[int, float] | None = None) -> object:
    scores = index_scores or {}

    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        value = scores.get(fold.index, float(fold.index))
        return FoldOutcome(
            scores={"test_score": value},
            rows=rows_for(fold.index, n=4, total=value * 4),
            best_epoch=fold.index,
            best_metric=value,
        )

    return body


# --------------------------------------------------------------------------- #
# The shape of a finished sweep
# --------------------------------------------------------------------------- #


def test_every_fold_produces_a_record_and_a_summary_row() -> None:
    result = sweep(simple_body())

    assert len(result.folds) == 3
    assert result.completed == result.folds
    assert list(result.summary["fold_index"]) == [0, 1, 2]
    assert (result.summary["status"] == "completed").all()


def test_the_summary_carries_the_bodys_own_column_names() -> None:
    """The driver never invents a metric name; it copies whatever the body returned."""
    result = sweep(simple_body())
    assert "test_score" in result.summary.columns
    assert list(result.summary["test_score"]) == [0.0, 1.0, 2.0]


def test_the_fold_index_is_an_int_in_the_first_column_of_the_rows() -> None:
    """`FoldWindow.to_dict` is string-valued; the results table wants numbers back."""
    result = sweep(simple_body())
    assert list(result.periods.columns)[0] == "fold_index"
    assert list(result.periods["fold_index"]) == [0, 1, 2]
    assert result.summary["fold_index"].dtype.kind == "i"
    assert result.summary["train_size"].dtype.kind == "i"


def test_a_body_that_already_labelled_its_rows_is_not_relabelled() -> None:
    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        rows = rows_for(fold.index)
        rows.insert(0, "fold_index", 99)
        return FoldOutcome(scores={"s": 1.0}, rows=rows)

    assert set(sweep(body).periods["fold_index"]) == {99}


def test_a_body_may_return_no_rows_at_all() -> None:
    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del fold, window, run, num_folds
        return FoldOutcome(scores={"s": 1.0})

    result = sweep(body)
    assert len(result.periods) == 0
    assert np.isnan(result.pooled["score"])
    assert list(result.summary["test_periods"]) == [0, 0, 0]


# --------------------------------------------------------------------------- #
# Pooled, never averaged
# --------------------------------------------------------------------------- #


def test_the_headline_number_is_pooled_over_rows_not_averaged_over_folds() -> None:
    """The discipline the driver exists to enforce."""

    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        # A thick fold scoring 1.0 and two thin ones scoring 10.0.
        n, value = (1000, 1.0) if fold.index == 0 else (1, 10.0)
        return FoldOutcome(
            scores={"test_score": value}, rows=pd.DataFrame({"n": [n], "total": [value * n]})
        )

    result = sweep(body)
    assert result.pooled["score"] == pytest.approx(1020.0 / 1002.0)
    mean_of_folds = float(np.mean([1.0, 10.0, 10.0]))
    assert abs(result.pooled["score"] - mean_of_folds) > 5.0  # they are not close


def test_the_pooler_sees_every_completed_folds_rows_at_once() -> None:
    seen: list[int] = []

    def counting_pool(rows: pd.DataFrame) -> Mapping[str, float]:
        seen.append(len(rows))
        return pool(rows)

    sweep(simple_body(), pool=counting_pool)  # type: ignore[arg-type]
    assert seen == [3]  # one call, over the concatenation


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def test_each_fold_is_seeded_before_its_body_runs() -> None:
    """Weight init draws from the global RNG, so the seed must land first."""
    draws: list[float] = []

    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        draws.append(float(torch.rand(1)))
        return FoldOutcome(scores={"s": float(fold.index)})

    torch.manual_seed(999)  # poisoned state the driver must overwrite
    sweep(body)
    assert draws[0] == draws[1] == draws[2]  # fixed seeding -> identical draws


def test_per_fold_seeding_makes_each_fold_draw_differently() -> None:
    draws: list[float] = []

    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        draws.append(float(torch.rand(1)))
        return FoldOutcome(scores={"s": float(fold.index)})

    sweep(body, policy=CrossValidationConfig(seed_mode="per_fold"))
    assert len(set(draws)) == 3


def test_a_sweep_is_a_function_of_its_seed() -> None:
    def make() -> tuple[object, list[float]]:
        draws: list[float] = []

        def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
            del fold, window, run, num_folds
            draws.append(float(torch.rand(1)))
            return FoldOutcome(scores={"s": 1.0})

        return body, draws

    first_body, first = make()
    second_body, second = make()
    sweep(first_body)
    sweep(second_body)
    assert first == second


@pytest.mark.parametrize(("mode", "expected"), [("fixed", [5, 5, 5]), ("per_fold", [5, 6, 7])])
def test_fold_seed_policy(mode: str, expected: list[int]) -> None:
    assert [fold_seed(5, i, mode) for i in range(3)] == expected


def test_fold_checkpoint_dir_leaves_a_one_fold_run_where_it_was() -> None:
    assert fold_checkpoint_dir("checkpoints", 0, 1) == "checkpoints"
    assert fold_checkpoint_dir("checkpoints", 0, 3) == str(Path("checkpoints", "fold-00"))
    assert fold_checkpoint_dir("checkpoints", 12, 20) == str(Path("checkpoints", "fold-12"))


# --------------------------------------------------------------------------- #
# Failure policy
# --------------------------------------------------------------------------- #


def failing_body(index: int | None) -> object:
    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        del window, run, num_folds
        if index is None or fold.index == index:
            raise RuntimeError(f"fold {fold.index} is too thin")
        return FoldOutcome(scores={"test_score": 1.0}, rows=rows_for(fold.index))

    return body


def test_a_failing_fold_costs_only_itself_under_skip() -> None:
    result = sweep(failing_body(1), policy=CrossValidationConfig(on_fold_error="skip"))

    assert [fold.index for fold in result.completed] == [0, 2]
    failed = result.folds[1]
    assert not failed.completed
    assert failed.error is not None and "too thin" in failed.error
    assert failed.rows is None and not failed.scores
    row = result.summary.iloc[1]
    assert row["status"] == "failed"
    assert row["test_periods"] == 0
    assert row["best_epoch"] == -1
    assert np.isnan(row["best_metric"])
    assert set(result.periods["fold_index"]) == {0, 2}


def test_the_raise_policy_propagates_the_original_exception_unwrapped() -> None:
    with pytest.raises(RuntimeError, match="fold 1 is too thin"):
        sweep(failing_body(1), policy=CrossValidationConfig(on_fold_error="raise"))


def test_a_sweep_in_which_every_fold_failed_still_raises() -> None:
    with pytest.raises(SweepError, match="every one of the 3 folds failed"):
        sweep(failing_body(None), policy=CrossValidationConfig(on_fold_error="skip"))


def test_an_empty_schedule_is_refused_before_anything_runs() -> None:
    ran = False

    def body(fold: FoldBounds, window: object, run: object, num_folds: int) -> FoldOutcome:
        nonlocal ran
        ran = True
        del fold, window, run, num_folds
        return FoldOutcome()

    with pytest.raises(SweepError, match="schedule is empty"):
        sweep(body, schedule=())
    assert not ran


def test_a_fold_cut_over_the_wrong_sequence_fails_as_one_fold() -> None:
    """Resolving a fold's dates sits inside the guard, so it is skippable too."""
    beyond = (
        *bounds(1),
        FoldBounds(index=1, train=range(0, 4), val=range(4, 8), test=range(8, 400)),
    )
    result = sweep(simple_body(), schedule=beyond)

    assert [fold.index for fold in result.completed] == [0]
    assert result.folds[1].error is not None
    # It could not be located, so it reports an empty window rather than inventing one.
    assert result.folds[1].window.train.size == 0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_describe_names_the_models_own_statistic() -> None:
    result = sweep(
        simple_body({0: 1.0, 1: 2.0, 2: 3.0}),
        report=SweepReport(headline="test R^2", spread_column="test_score", spread_label="R^2"),
        spread_columns=("test_score",),
    )
    described = result.describe()
    assert "folds: 3 completed of 3 planned" in described
    assert "pooled test R^2 -> score=" in described
    assert "per-fold R^2 -> median=2.0000" in described


def test_describe_omits_the_spread_line_when_no_column_is_named() -> None:
    described = sweep(simple_body()).describe()
    assert "per-fold" not in described


def test_a_single_completed_fold_reports_no_spread() -> None:
    """A standard deviation over one number is not a distribution."""
    result = sweep(simple_body(), schedule=bounds(1), spread_columns=("test_score",))
    assert "per-fold" not in result.describe()


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def test_aggregate_spread_reports_the_six_statistics() -> None:
    summary = pd.DataFrame({"status": ["completed"] * 4, "test_score": [1.0, 2.0, 3.0, 4.0]})
    metrics = aggregate_spread(summary, ("test_score",))
    assert metrics["test/score_mean"] == pytest.approx(2.5)
    assert metrics["test/score_median"] == pytest.approx(2.5)
    assert metrics["test/score_min"] == 1.0
    assert metrics["test/score_max"] == 4.0
    assert metrics["test/score_std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert metrics["test/score_iqr"] == pytest.approx(1.5)


def test_aggregate_spread_excludes_failed_folds() -> None:
    summary = pd.DataFrame(
        {"status": ["completed", "failed", "completed"], "test_score": [1.0, np.nan, 3.0]}
    )
    assert aggregate_spread(summary, ("test_score",))["test/score_mean"] == pytest.approx(2.0)


def test_aggregate_spread_skips_std_and_iqr_for_a_single_fold() -> None:
    summary = pd.DataFrame({"status": ["completed"], "test_score": [1.0]})
    metrics = aggregate_spread(summary, ("test_score",))
    assert metrics["test/score_mean"] == 1.0
    assert "test/score_std" not in metrics
    assert "test/score_iqr" not in metrics


@pytest.mark.parametrize(
    "summary",
    [
        pd.DataFrame({"test_score": [1.0]}),  # no status column
        pd.DataFrame({"status": ["completed"]}),  # column absent
        pd.DataFrame({"status": ["completed"], "test_score": [np.nan]}),  # all NaN
    ],
)
def test_aggregate_spread_returns_nothing_rather_than_failing(summary: pd.DataFrame) -> None:
    assert aggregate_spread(summary, ("test_score",)) == {}


def test_aggregate_spread_with_no_columns_is_empty() -> None:
    assert aggregate_spread(pd.DataFrame({"status": ["completed"]}), ()) == {}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def test_sweep_tags_describe_a_walk_forward_schedule() -> None:
    split = SplitConfig(scheme="walk_forward", burn_in=3, walk_forward=WalkForwardConfig(purge=5))
    assert sweep_tags(split, 7) == {
        "cv_scheme": "walk_forward",
        "cv_mode": "rolling",
        "cv_folds": "7",
        "cv_purge": "5",
        "cv_burn_in": "3",
    }


def test_sweep_tags_leave_the_walk_forward_fields_empty_for_a_single_split() -> None:
    tags = sweep_tags(SplitConfig(scheme="single"), 1)
    assert tags["cv_scheme"] == "single"
    assert tags["cv_mode"] == "" and tags["cv_purge"] == ""


def test_unplaced_window_is_empty_in_every_segment() -> None:
    window = unplaced_window(4)
    assert window.index == 4
    assert (window.train.size, window.val.size, window.test.size) == (0, 0, 0)
    assert window.to_dict()["train_start"] == ""


def test_warn_on_overlap_fires_only_when_test_windows_share_a_date(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        warn_on_overlap(bounds(3))
    assert not caplog.records

    overlapping = (
        FoldBounds(index=0, train=range(0, 4), val=range(4, 8), test=range(8, 16)),
        FoldBounds(index=1, train=range(2, 6), val=range(6, 10), test=range(10, 18)),
    )
    with caplog.at_level("WARNING"):
        warn_on_overlap(overlapping)
    assert "overlap" in caplog.text


def test_aggregate_spread_names_the_namespace_after_the_columns_first_word() -> None:
    """The summary table and the metric tree stay in step without knowing each other."""
    summary = pd.DataFrame(
        {
            "status": ["completed"] * 3,
            "val_total_r2": [0.10, 0.12, 0.14],
            "test_total_r2": [0.05, 0.06, 0.07],
        }
    )
    metrics = aggregate_spread(summary, ("val_total_r2", "test_total_r2"))

    assert metrics["val/total_r2_mean"] == pytest.approx(0.12)
    assert metrics["test/total_r2_mean"] == pytest.approx(0.06)


def test_aggregate_spread_reports_the_count_and_the_standard_error() -> None:
    """A dispersion over nine folds and one over three are different evidence."""
    summary = pd.DataFrame({"status": ["completed"] * 4, "test_score": [1.0, 2.0, 3.0, 4.0]})
    metrics = aggregate_spread(summary, ("test_score",))

    assert metrics["test/score_n"] == 4.0
    assert metrics["test/score_sem"] == pytest.approx(metrics["test/score_std"] / 2.0)


def test_aggregate_spread_omits_the_standard_error_for_a_single_fold() -> None:
    summary = pd.DataFrame({"status": ["completed"], "test_score": [1.0]})
    metrics = aggregate_spread(summary, ("test_score",))

    assert metrics["test/score_n"] == 1.0
    assert "test/score_sem" not in metrics


def _converged(index: int, **metrics: float) -> FoldRecord:
    return FoldRecord(
        index=index,
        window=unplaced_window(index),
        run_id=None,
        outcome=FoldOutcome(metrics=metrics),
    )


def test_aggregate_convergence_counts_folds_rather_than_averaging_flags() -> None:
    """ "Two of three folds hit the cap" is the actionable form; a fraction hides it."""
    records = (
        _converged(0, **{"fold/hit_epoch_cap": 1.0, "fold/diverged": 0.0}),
        _converged(1, **{"fold/hit_epoch_cap": 1.0, "fold/diverged": 0.0}),
        _converged(2, **{"fold/hit_epoch_cap": 0.0, "fold/diverged": 1.0}),
    )
    metrics = aggregate_convergence(records)

    assert metrics["folds/hit_epoch_cap"] == 2.0
    assert metrics["folds/diverged"] == 1.0


def test_aggregate_convergence_sums_cost_and_takes_the_median_of_the_rest() -> None:
    records = (
        _converged(0, **{"fold/fit_seconds": 10.0, "fold/epochs_trained": 40.0}),
        _converged(1, **{"fold/fit_seconds": 30.0, "fold/epochs_trained": 60.0}),
    )
    metrics = aggregate_convergence(records)

    assert metrics["fit/fit_seconds_total"] == pytest.approx(40.0)
    assert metrics["fit/epochs_trained_median"] == pytest.approx(50.0)


def test_aggregate_convergence_is_silent_for_a_model_that_reports_none() -> None:
    assert aggregate_convergence((_converged(0, **{"test/loss": 1.0}),)) == {}


def test_aggregate_convergence_ignores_failed_folds() -> None:
    failed = FoldRecord(index=1, window=unplaced_window(1), run_id=None, error="boom")
    records = (_converged(0, **{"fold/hit_epoch_cap": 1.0}), failed)

    assert aggregate_convergence(records)["folds/hit_epoch_cap"] == 1.0
