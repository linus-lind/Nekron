"""Proximal operators: the penalties an adaptive optimizer cannot be told through a loss.

Adding ``lambda * sum(|W|)`` to a loss is the textbook way to write down a LASSO,
and under plain SGD it behaves as written: the penalty contributes a gradient of
``lambda * sign(w)`` and the resulting step is ``lr * lambda``. Under Adam it does
not. Adam divides each gradient by its own running RMS, so a term whose magnitude
is constant and whose sign is stable — which is exactly what ``lambda * sign(w)``
is — normalizes to ``+-1`` and moves the weight by the full ``lr``, whatever
``lambda`` was set to. The coefficient stops being a strength and becomes a
threshold on *which* weights are penalized, while ``lr`` silently decides how fast
they are erased.

The fix is to keep the non-smooth term out of the gradient entirely and apply its
proximal operator after the step, which is proximal gradient descent (ISTA). For
the L1 norm that operator is the soft-threshold

    prox(w) = sign(w) * max(|w| - lr * lambda, 0),

the closed-form minimizer of ``0.5 * (u - w)^2 + lr * lambda * |u|``. It restores
the intended ``lr * lambda`` rate, produces exact zeros rather than a cloud of
values dithering around zero at the scale of ``lr``, and — because a weight parked
at zero is still one MSE gradient step away from leaving it — keeps the selection
reversible instead of one-way.

The rate difference is not subtle. At ``lr=1e-3`` and ``lambda=1e-4`` the
subgradient formulation retires a weight of magnitude 0.5 in about 500 steps; the
proximal one takes five million. A run switching from one to the other therefore
has to re-tune ``lambda``, upward by roughly ``1/lambda``, and that is the point:
afterwards the number means what it says.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProximalL1:
    """Soft-thresholds weights toward zero at a rate of ``lr * lam`` per step.

    Applied by :class:`~nekron.nn.diagnostics.OptimizerProbe` immediately after
    ``optimizer.step()``, so the shrinkage is inside the window ``opt/update_ratio``
    measures: it is part of the update, and an update ratio that excluded it would
    describe a step the model never took.

    ``lr`` is read from the optimizer at call time rather than stored, so the
    shrinkage follows the schedule automatically. That coupling is the correct one
    — the proximal operator's argument *is* the gradient step size — and it means a
    decaying learning rate anneals the penalty alongside everything else.
    """

    lam: float

    def __post_init__(self) -> None:
        if self.lam < 0.0:
            raise ValueError(f"ProximalL1 lam must not be negative; got {self.lam}.")

    @torch.no_grad()
    def __call__(self, weights: Iterable[torch.Tensor], lr: float) -> None:
        """Shrink ``weights`` in place by ``lr * lam``, clamping at exactly zero."""
        if self.lam == 0.0:
            return
        threshold = lr * self.lam
        for weight in weights:
            weight.copy_(weight.sign() * (weight.abs() - threshold).clamp_min(0.0))

    @staticmethod
    def penalized(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        """The parameters an L1 penalty applies to: the weight matrices.

        ``ndim >= 2`` selects the linear maps and leaves biases and normalization
        scales alone, matching :meth:`~nekron.asset_pricing.conditional_autoencoder
        .losses.CaeLoss.l1_penalty` and :func:`~nekron.nn.diagnostics
        .weight_summary` so that the penalty, the metric that reports it and the
        operator that applies it all describe the same set of tensors.
        """
        return [p for p in parameters if p.ndim >= 2]


def value(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """``sum(|W|)`` over the penalized weights: the penalty's value, not its gradient.

    Reported as ``loss/l1`` whether the penalty reaches the optimizer through the
    loss or through :class:`ProximalL1`. Under the proximal formulation it is no
    longer a term of the objective being differentiated, but it is still the
    quantity being penalized, and watching it fall is how the penalty is seen to be
    working at all.
    """
    if not weights:
        return torch.zeros(())
    return torch.stack([w.abs().sum() for w in weights]).sum()
