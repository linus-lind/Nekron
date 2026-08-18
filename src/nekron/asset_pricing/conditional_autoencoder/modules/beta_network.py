"""Factor-loading (beta) network."""

from __future__ import annotations

import torch
from torch import nn

from nekron.nn import MLP

from ..config import BetaNetworkConfig


class BetaNetwork(nn.Module):
    """Maps per-stock beta inputs to factor loadings.

    A shared MLP is applied row-wise to every stock in a cross-section, turning an
    ``[N, P]`` beta-input matrix into an ``[N, K]`` matrix of factor loadings.
    """

    def __init__(self, cfg: BetaNetworkConfig, *, num_beta_columns: int, num_factors: int) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=num_beta_columns,
            hidden_dims=cfg.hidden_dims,
            out_dim=num_factors,
            activation=cfg.activation,
            batch_norm=cfg.batch_norm,
            dropout=cfg.dropout,
            bias=cfg.bias,
            track_running_stats=cfg.track_running_stats,
        )

    def forward(self, beta_inputs: torch.Tensor) -> torch.Tensor:
        loadings: torch.Tensor = self.mlp(beta_inputs)
        return loadings
