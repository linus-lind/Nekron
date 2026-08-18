"""Tests for :mod:`nekron.nn.training`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nekron.nn.config import TrainConfig
from nekron.nn.training import EarlyStopping, FitResult


def _model() -> nn.Module:
    return nn.Linear(2, 2)


def _run(stopper: EarlyStopping, metrics: list[float]) -> list[bool]:
    model = _model()
    return [stopper.update(metric, epoch, model) for epoch, metric in enumerate(metrics)]


# --------------------------------------------------------------------------- #
# min_delta
# --------------------------------------------------------------------------- #


def test_gains_below_min_delta_do_not_reset_patience() -> None:
    """The case the threshold exists for: a metric drifting inside its own noise.

    Without it, every one of these epochs counts as an improvement and patience
    never expires — which is how a run selects epoch 938 of 1000.
    """
    noise = [0.100, 0.1001, 0.1002, 0.1003, 0.1004]
    assert _run(EarlyStopping(patience=3, min_delta=1e-3), noise) == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert not any(_run(EarlyStopping(patience=3), noise))


def test_sub_threshold_gains_are_not_recorded_as_best() -> None:
    """``min_delta`` gates the recorded state too, not only the counter."""
    stopper = EarlyStopping(patience=10, min_delta=1e-3)
    _run(stopper, [0.100, 0.1001, 0.1002])
    assert stopper.best_epoch == 0
    assert stopper.best_metric == pytest.approx(0.100)


def test_a_gain_above_min_delta_resets_patience_and_records() -> None:
    stopper = EarlyStopping(patience=3, min_delta=1e-3)
    stops = _run(stopper, [0.10, 0.1001, 0.20, 0.2001, 0.2002, 0.2003])
    assert stops == [False, False, False, False, False, True]
    assert stopper.best_epoch == 2
    assert stopper.best_metric == pytest.approx(0.20)


def test_the_first_finite_metric_is_recorded_whatever_the_threshold() -> None:
    """``best_metric`` starts at ``-inf``, which no threshold can lift."""
    stopper = EarlyStopping(patience=2, min_delta=1e9)
    assert stopper.update(-5.0, 0, _model()) is False
    assert stopper.best_epoch == 0
    assert stopper.best_metric == pytest.approx(-5.0)


def test_min_delta_defaults_to_accepting_any_strict_gain() -> None:
    stopper = EarlyStopping(patience=2)
    assert stopper.min_delta == 0.0
    assert not any(_run(stopper, [0.1, 0.1 + 1e-12, 0.1 + 2e-12]))


def test_min_delta_is_applied_in_the_maximizing_direction() -> None:
    """A caller selecting on a loss negates the loss, not the threshold."""
    stopper = EarlyStopping(patience=2, min_delta=0.5)
    losses = [10.0, 9.9, 9.0]
    stops = [stopper.update(-loss, epoch, _model()) for epoch, loss in enumerate(losses)]
    assert stops == [False, False, False]
    assert stopper.best_epoch == 2  # 9.9 was not enough; 9.0 was


# --------------------------------------------------------------------------- #
# Interaction with the existing contract
# --------------------------------------------------------------------------- #


def test_non_finite_metrics_are_never_recorded_but_still_exhaust_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=1e-3)
    assert _run(stopper, [math.nan, math.inf]) == [False, True]
    assert stopper.best_state is None
    assert stopper.best_epoch == -1


def test_restore_loads_the_recorded_state() -> None:
    model = _model()
    stopper = EarlyStopping(patience=5, min_delta=1e-3)
    stopper.update(1.0, 0, model)
    best = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    stopper.restore(model)
    for name, tensor in model.state_dict().items():
        assert tensor.equal(best[name])


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_train_config_rejects_a_negative_min_delta() -> None:
    with pytest.raises(ValueError, match="train.min_delta must not be negative"):
        TrainConfig(min_delta=-1e-6)


@pytest.mark.parametrize("value", [0.0, 1e-4, 1.0])
def test_train_config_accepts_a_non_negative_min_delta(value: float) -> None:
    assert TrainConfig(min_delta=value).min_delta == value


# --------------------------------------------------------------------------- #
# What a fit reports about how it terminated
# --------------------------------------------------------------------------- #


def _fit(*, epochs: int, budget: int, best: int, metric: float = 0.5) -> FitResult:
    history = [{"val/total_r2": 0.1 * i, "train/total_r2": 0.2 * i} for i in range(epochs)]
    return FitResult(
        best_metric=metric,
        best_epoch=best,
        epoch_budget=budget,
        fit_seconds=10.0,
        history=history,
    )


def test_a_run_that_used_its_whole_budget_is_marked_capped_not_converged() -> None:
    """A capped run is unfinished evidence, not weak evidence, about a configuration."""
    result = _fit(epochs=250, budget=250, best=249)

    assert result.hit_epoch_cap
    assert not result.stopped_early
    assert result.convergence_metrics()["fold/hit_epoch_cap"] == 1.0


def test_a_run_that_ran_out_of_patience_is_marked_early_stopped() -> None:
    result = _fit(epochs=80, budget=250, best=55)

    assert result.stopped_early
    assert not result.hit_epoch_cap
    assert result.epochs_trained == 80


def test_divergence_is_a_flag_rather_than_a_gap_in_the_curve() -> None:
    """Non-finite metrics are dropped on the way to the tracker, so nothing else says so."""
    assert _fit(epochs=30, budget=250, best=-1, metric=math.nan).diverged
    assert _fit(epochs=30, budget=250, best=5, metric=math.inf).diverged
    assert not _fit(epochs=30, budget=250, best=5).diverged


def test_the_best_row_is_the_selected_epoch_and_not_the_last_one() -> None:
    """The last row is the model ``patience`` epochs after the one that was kept."""
    result = _fit(epochs=80, budget=250, best=10)

    assert result.best_row["val/total_r2"] == pytest.approx(1.0)
    assert result.convergence_metrics()["best/val_total_r2"] == pytest.approx(1.0)


def test_a_best_epoch_outside_the_history_yields_no_row_rather_than_an_error() -> None:
    assert _fit(epochs=3, budget=250, best=-1).best_row == {}
    assert _fit(epochs=3, budget=250, best=99).best_row == {}


def test_convergence_metrics_report_the_cost_per_epoch() -> None:
    metrics = _fit(epochs=20, budget=250, best=9).convergence_metrics()

    assert metrics["fold/fit_seconds"] == pytest.approx(10.0)
    assert metrics["fold/epoch_seconds"] == pytest.approx(0.5)


def test_convergence_metric_names_stay_one_path_segment_deep() -> None:
    """MLflow keys are flat; a nested epoch key would read as two levels here."""
    metrics = _fit(epochs=5, budget=250, best=2).convergence_metrics()

    assert "best/val_total_r2" in metrics
    assert not any(key.count("/") > 1 for key in metrics)
