"""Reusable training-loop helpers: what every fit does the same way.

Early stopping, building the optimizer, deciding which epochs to record, and
writing the selected model to disk are the same four operations in every model
here, and each has one detail that is easy to get wrong in a fresh copy — the
probe has to own the clip *and* the step, the last epoch has to be logged whatever
the interval, and the checkpoint has to carry enough to be interpreted later.
They are written once here so a new model inherits the details rather than
rediscovering them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nekron.tracking import MlflowTracker

from .config import OptimConfig, TrainConfig
from .diagnostics import OptimizerProbe
from .proximal import ProximalL1

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


FOLD_PREFIX = "fold/"
"""Namespace for the per-fold scalars a fit reports about itself."""

BEST_PREFIX = "best/"
"""Namespace for the epoch metrics as they stood at the selected epoch."""

FOLD_FLAGS: tuple[str, ...] = (
    f"{FOLD_PREFIX}hit_epoch_cap",
    f"{FOLD_PREFIX}early_stopped",
    f"{FOLD_PREFIX}diverged",
)
"""Zero-or-one fold metrics a sweep counts folds over rather than averaging."""

FOLD_TOTALS: tuple[str, ...] = (f"{FOLD_PREFIX}fit_seconds",)
"""Fold metrics a sweep sums."""

FOLD_MEDIANS: tuple[str, ...] = (
    f"{FOLD_PREFIX}epochs_trained",
    f"{FOLD_PREFIX}epoch_seconds",
)
"""Fold metrics a sweep reports the median of."""


@dataclass
class FitResult:
    """What a completed fit reports: the selected epoch and the curve behind it.

    :attr:`best_metric` is in whatever direction the model selects on — a
    higher-is-better R-squared for one model, a lower-is-better loss for another —
    and is reported as the model itself would state it, never as the internally
    negated value early stopping maximizes.

    :attr:`epoch_budget` and :attr:`fit_seconds` are recorded rather than derived
    because neither survives the fit. The budget is what separates a run that
    converged from one that merely ran out of epochs — the same curve, and two
    entirely different pieces of evidence about the configuration that produced
    it — and it cannot be read off the history, which stops at the same place in
    both cases.
    """

    best_metric: float
    best_epoch: int
    epoch_budget: int
    fit_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def epochs_trained(self) -> int:
        """Epochs that actually ran; one row of :attr:`history` each."""
        return len(self.history)

    @property
    def hit_epoch_cap(self) -> bool:
        """Whether the loop ended because the budget ran out rather than because it converged.

        A capped run is not a bad run and not a good one: it is an *unfinished*
        one, and comparing it with a converged run compares budgets as much as
        configurations. Configurations differ systematically in how many epochs
        they need — a lower learning rate, a heavier penalty and a wider network
        all need more — so silently discarding capped runs would bias a search
        toward whatever converges fastest. It is recorded so the run can be given
        the budget it asked for and repeated.
        """
        return self.epoch_budget > 0 and self.epochs_trained >= self.epoch_budget

    @property
    def stopped_early(self) -> bool:
        """Whether patience was exhausted before the budget was."""
        return self.epochs_trained > 0 and not self.hit_epoch_cap

    @property
    def diverged(self) -> bool:
        """Whether no epoch ever produced a finite selectable score.

        Distinct from a merely poor fit, and invisible in the logged curve: a run
        whose metrics went non-finite stops being plotted rather than being marked,
        because non-finite metrics are dropped on the way to the tracker.
        """
        return self.best_epoch < 0 or not math.isfinite(self.best_metric)

    @property
    def best_row(self) -> dict[str, float]:
        """The epoch metrics as they stood at the selected epoch.

        The state of the model that was actually kept. The last row of the history
        is a different thing — the state ``patience`` epochs later, after the run
        had stopped improving — and reading one for the other is how a diagnostic
        ends up describing a model nobody uses.
        """
        if not 0 <= self.best_epoch < len(self.history):
            return {}
        return dict(self.history[self.best_epoch])

    def convergence_metrics(self) -> dict[str, float]:
        """How the fit terminated, plus every epoch metric at the selected epoch.

        Flat, scalar and model-agnostic, so a fold can merge it into whatever it
        reports without the caller naming any of it. The ``best/`` half is
        whatever the model happened to record per epoch, which is what makes a
        run's health checkable from the run itself instead of from a downloaded
        curve.
        """
        metrics = {
            f"{FOLD_PREFIX}epochs_trained": float(self.epochs_trained),
            f"{FOLD_PREFIX}hit_epoch_cap": float(self.hit_epoch_cap),
            f"{FOLD_PREFIX}early_stopped": float(self.stopped_early),
            f"{FOLD_PREFIX}diverged": float(self.diverged),
            f"{FOLD_PREFIX}fit_seconds": float(self.fit_seconds),
        }
        if self.epochs_trained:
            metrics[f"{FOLD_PREFIX}epoch_seconds"] = float(self.fit_seconds / self.epochs_trained)
        # Slashes flattened, so ``grad/norm_p90`` at the selected epoch reads as
        # ``best/grad_norm_p90`` and stays one path segment deep like every other
        # metric the tracker holds.
        metrics.update(
            {f"{BEST_PREFIX}{key.replace('/', '_')}": value for key, value in self.best_row.items()}
        )
        return metrics


@dataclass(frozen=True)
class Optimization:
    """The optimizer, its schedule, and the probe that steps it."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    probe: OptimizerProbe

    @property
    def learning_rate(self) -> float:
        """The rate the *current* epoch trained at.

        Read before the scheduler steps: an epoch's metrics have to be read
        against the rate that produced them, not the one the next epoch will use.
        """
        return float(self.scheduler.get_last_lr()[0])


def build_optimization(
    model: nn.Module,
    optim: OptimConfig,
    train: TrainConfig,
    *,
    proximal: ProximalL1 | None = None,
) -> Optimization:
    """AdamW, an exponential decay, and the probe that clips, steps and shrinks.

    The probe replaces the ``clip_grad_norm_`` / ``optimizer.step()`` pair rather
    than wrapping it, because each of its measurements has exactly one valid
    observation point — the gradient norm before clipping rescales it, and the
    update across the step itself. ``proximal`` joins the step for the same
    reason: its shrinkage is part of the update, so it has to fall inside the
    window ``opt/update_ratio`` measures.

    ``AdamW`` and ``Adam`` coincide exactly at ``weight_decay: 0.0``; see
    :class:`~nekron.nn.config.OptimConfig` for why the difference matters once it
    is not zero.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim.lr,
        betas=(optim.betas[0], optim.betas[1]),
        weight_decay=optim.weight_decay,
    )
    return Optimization(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=optim.lr_decay_gamma),
        probe=OptimizerProbe(
            model,
            optimizer,
            train.diagnostics,
            clip_norm=train.grad_clip_norm,
            proximal=proximal,
        ),
    )


def should_log_epoch(epoch: int, *, interval: int, epochs: int) -> bool:
    """Whether epoch ``epoch`` is recorded, given a logging interval.

    The final epoch is always recorded, whatever the interval, so a run's last
    logged metrics are the ones early stopping actually acted on.
    """
    return epoch % max(1, interval) == 0 or epoch == epochs - 1


def save_checkpoint(
    model: nn.Module,
    *,
    directory: str,
    filename: str,
    config: DataclassInstance,
    best_epoch: int,
    best_metric: float,
    tracker: MlflowTracker,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    """Write the selected model, then attach it to the run.

    ``extras`` is whatever the model needs in order to be *interpreted* months
    later — the column order its input widths were inferred from, the window
    length it was built for. A checkpoint without that is a tensor of weights
    whose dimensions no longer mean anything once the configuration has moved on,
    so it is a parameter rather than an afterthought.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            **dict(extras or {}),
        },
        path,
    )
    tracker.log_artifact(path)
    tracker.log_model(model)
    return path


class EarlyStopping:
    """Tracks the best-scoring model state and signals when to stop.

    A higher metric is better. :meth:`update` records a deep copy of the model
    state whenever the metric improves on the best seen by more than
    ``min_delta``, and returns whether patience has been exhausted;
    :meth:`restore` loads the best recorded state back into the model.

    ``min_delta`` gates both decisions rather than only the counter: an epoch that
    fails it is neither recorded nor treated as progress. Splitting the two would
    let a run keep collecting sub-threshold bests while the counter ignores them,
    which is the behaviour the threshold exists to remove; and the state it can
    cost is bounded by ``min_delta``, which is the noise level by construction. It
    is expressed in the units of whatever metric is passed in and in the
    maximizing direction, so a caller selecting on a loss negates the loss and not
    the threshold.
    """

    def __init__(self, patience: int, *, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric: float = -math.inf
        self.best_epoch: int = -1
        self.best_state: dict[str, Any] | None = None
        self._wait = 0

    def update(self, metric: float, epoch: int, model: nn.Module) -> bool:
        """Record the model if ``metric`` gained more than ``min_delta``; stop?"""
        # ``best_metric`` starts at -inf, and -inf + min_delta is still -inf, so the
        # first finite metric is always an improvement whatever the threshold.
        if math.isfinite(metric) and metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state = deepcopy(model.state_dict())
            self._wait = 0
            return False
        self._wait += 1
        return self._wait >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best recorded state dict into the model, if one was recorded."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
