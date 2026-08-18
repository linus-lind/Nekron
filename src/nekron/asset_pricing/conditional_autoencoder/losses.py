"""Training objective: mean squared pricing error plus an L1 penalty.

    L = MSE(r, r_hat) + l1_lambda * sum(|W|)

The mean squared error is taken over all stock-period observations in the batch;
the L1 (LASSO) penalty is summed over the linear weight matrices of both networks.

The objective is the same under either :attr:`~..config.LossConfig.l1_mode`; what
differs is where the penalty is applied. Under ``"proximal"`` — the default — the
penalty is *not* part of the tensor this module hands to ``backward()``: it is
applied to the weights after the optimizer step by
:class:`nekron.nn.ProximalL1`, because an adaptive optimizer normalizes the
magnitude out of an L1 subgradient and would shrink at ``lr`` rather than at
``lr * l1_lambda``. Its value is still computed and reported as ``loss/l1``, since
watching the penalized quantity fall is how the penalty is seen to work at all,
but ``loss/total`` then equals ``loss/mse`` and is the quantity actually
differentiated. Under ``"subgradient"`` the penalty is added to the loss in the
textbook way, and ``loss/total`` includes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from nekron.nn.proximal import ProximalL1

from .config import LossConfig


@dataclass
class LossBreakdown:
    """Total loss plus its components, retaining the graph for backprop on ``total``."""

    total: torch.Tensor
    mse: torch.Tensor
    l1: torch.Tensor

    def detached_components(self) -> dict[str, torch.Tensor]:
        """Detached, device-resident component tensors (no host-device sync)."""
        return {
            "loss/total": self.total.detach(),
            "loss/mse": self.mse.detach(),
            "loss/l1": self.l1.detach(),
        }

    def to_log_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.detached_components().items()}


class CaeLoss:
    """Callable composite loss over reconstructed and realized returns."""

    def __init__(self, cfg: LossConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def mean_squared_error(fitted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared pricing error over all observations."""
        return ((fitted - target) ** 2).mean()

    @staticmethod
    def l1_penalty(model_parameters: Iterable[torch.Tensor]) -> torch.Tensor:
        """Sum of absolute values over the linear weight matrices (``ndim >= 2``)."""
        weights = ProximalL1.penalized(model_parameters)
        return torch.stack([w.abs().sum() for w in weights]).sum()

    def proximal(self) -> ProximalL1 | None:
        """The operator applying this penalty after the step, if that is the mode.

        ``None`` under ``"subgradient"``, where the penalty travels through the
        loss instead. The trainer hands the result to
        :func:`~nekron.nn.build_optimization`, so exactly one of the two paths is
        ever active and neither configuration applies the penalty twice.
        """
        if self.cfg.l1_mode != "proximal":
            return None
        return ProximalL1(self.cfg.l1_lambda)

    def __call__(
        self,
        fitted: torch.Tensor,
        target: torch.Tensor,
        model_parameters: Iterable[torch.Tensor],
    ) -> LossBreakdown:
        mse = self.mean_squared_error(fitted, target)
        l1 = self.l1_penalty(model_parameters)
        # Under "proximal" the penalty is reported but not differentiated: it
        # reaches the weights through ProximalL1 after the step. Adding it here as
        # well would apply it twice, once at the wrong rate.
        total = mse if self.cfg.l1_mode == "proximal" else mse + self.cfg.l1_lambda * l1
        return LossBreakdown(total=total, mse=mse, l1=l1.detach())
