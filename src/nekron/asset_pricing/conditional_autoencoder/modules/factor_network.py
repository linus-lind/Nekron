"""Factor network."""

from __future__ import annotations

import torch
from torch import nn

from nekron.nn import MLP

from ..config import FactorNetworkConfig


class FactorNetwork(nn.Module):
    """Maps the characteristic-managed portfolios to latent factors.

    The managed portfolios ``x = (Z^T Z)^{-1} Z^T r`` (one entry per portfolio
    characteristic) are fed through an MLP producing the ``K`` latent factors for
    the period. The input width equals the number of portfolios ``P_port``.
    """

    def __init__(self, cfg: FactorNetworkConfig, *, num_portfolios: int, num_factors: int) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=num_portfolios,
            hidden_dims=cfg.hidden_dims,
            out_dim=num_factors,
            activation=cfg.activation,
            batch_norm=cfg.batch_norm,
            dropout=cfg.dropout,
            bias=cfg.bias,
            track_running_stats=cfg.track_running_stats,
        )

    def forward(self, managed_portfolios: torch.Tensor) -> torch.Tensor:
        factors: torch.Tensor = self.mlp(managed_portfolios)
        return factors
