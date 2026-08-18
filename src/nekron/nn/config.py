"""Configuration shared by every model's training loop.

Optimization and training control are the same question for every model in this
project — which optimizer, how long, how hard to clip, what to record about the
run — so they are declared once here and embedded by each model's root config.
What stays model-specific is anything that names a *statistic*: which metric
selects a model, and in which direction, is a property of what the model computes
and belongs in its own package.

These are plain dataclasses with no behaviour, so a model composes them through
Hydra exactly as if it had declared them itself, and a model that needs an extra
field subclasses rather than redeclares — see
:class:`~nekron.asset_pricing.conditional_autoencoder.config.TrainConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .diagnostics import DiagnosticsConfig


@dataclass
class OptimConfig:
    """AdamW with an exponentially decaying learning rate.

    ``AdamW`` rather than ``Adam`` because the two differ only in how
    :attr:`weight_decay` reaches the parameters, and only one of the two ways is
    the one the number describes. ``Adam`` adds ``weight_decay * theta`` to the
    gradient, where Adam's own normalization by gradient RMS divides the magnitude
    back out — a decay term is consistent in sign, so it normalizes toward ``+-1``
    and moves the parameter by roughly the full ``lr`` regardless of the
    coefficient. ``AdamW`` applies ``theta <- theta - lr * weight_decay * theta``
    outside the adaptive step, so the coefficient sets the rate as written. At
    ``weight_decay: 0.0`` the two are identical, which is why this can change
    underneath an existing configuration without moving any result.

    :attr:`weight_decay` is an L2 pull toward zero, proportional to the parameter,
    so it shrinks weights without ever reaching zero and cannot by itself kill a
    unit. That is what makes it the safe regularizer to add first: it settles
    ``weights/norm`` at the level where decay balances the gradient, and a norm
    still climbing at the end of a run is the reading that it is too weak or
    absent. Contrast :attr:`l1_lambda` on a model's loss config, which is a
    constant pull that does reach zero and does remove units.

    :attr:`lr_decay_gamma` defaults to ``1.0``, which is no schedule at all: the
    adaptive step is usually the only one these models need, and an explicit decay
    is exposed for tuning rather than assumed. It matters more than it looks —
    Adam's step is ``lr`` times a quantity normalized to roughly unit scale, so
    unlike SGD it does not slow down as gradients shrink, and ``opt/update_ratio``
    pinned flat across a run is what that looks like in the metrics.
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    lr_decay_gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"optim.lr must be positive; got {self.lr}.")
        if len(self.betas) != 2:
            raise ValueError(f"optim.betas must hold exactly two values; got {self.betas}.")


@dataclass
class TrainConfig:
    """Training-loop control, including what the loop records about itself.

    Parameters
    ----------
    epochs:
        Maximum epochs; early stopping normally ends a run before this.
    early_stopping_patience:
        Epochs without an improvement before the run stops.
    min_delta:
        How much the selection metric has to gain for an epoch to count as an
        improvement at all. ``0.0`` accepts any strict gain, which lets a metric
        wandering inside its own noise reset the patience counter indefinitely: a
        validation R-squared moving in 1e-4 steps under ``early_stopping_patience:
        100`` will run to the epoch cap without ever having improved on anything.
        Set it to the size of that noise, read off a run's own validation curve.
    grad_clip_norm:
        ``0.0`` disables clipping. Set it from a run's ``grad/norm_p90``, which is
        the 90th percentile of the gradient norms that run actually produced;
        ``grad/clipped_frac`` then says how often the threshold bites. Past ~0.5 it
        has stopped being clipping and become an obscure way of lowering the
        learning rate.
    seed:
        Seeds Python, NumPy and Torch, and under a sweep is offset per fold.
    device:
        ``"auto"`` selects CUDA, then Apple-silicon MPS, then CPU.
    checkpoint_dir:
        Where the best model is written. A multi-fold sweep gives each fold its own
        subdirectory, because the checkpoint's *name* is a constant inside each
        trainer.
    diagnostics:
        Per-epoch optimizer instrumentation — gradient norms, the update-to-weight
        ratio that reads the learning rate, weight sparsity, activation saturation.
        Turning it off leaves the fit metrics alone and skips only the
        instrumentation.
    """

    epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 0.0
    grad_clip_norm: float = 0.0
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"train.epochs must be positive; got {self.epochs}.")
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"train.early_stopping_patience must be positive; got "
                f"{self.early_stopping_patience}."
            )
        if self.min_delta < 0.0:
            raise ValueError(f"train.min_delta must not be negative; got {self.min_delta}.")
        if self.grad_clip_norm < 0.0:
            raise ValueError(
                f"train.grad_clip_norm must not be negative; got {self.grad_clip_norm}."
            )
