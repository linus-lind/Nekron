"""The score head: pooled window representation to scores."""

from __future__ import annotations

import torch
from torch import nn

from nekron.nn import MLP

from ..config import ScoreHeadConfig


class ScoreHead(nn.Module):
    """Maps a pooled window vector to ``out_dim`` scores.

    A plain :class:`~nekron.nn.mlp.MLP`, so the last layer is a bare linear map:
    the head emits unbounded real numbers and any squashing belongs to the
    objective, which is the only thing that knows what the scores are supposed to
    mean.

    The input width is the encoder's output width and is passed in, never
    configured — the pooling kind decides it, and a setting that could disagree
    would only be a way to get it wrong.
    """

    def __init__(self, cfg: ScoreHeadConfig, *, in_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.out_dim,
            activation=cfg.activation,
            batch_norm=cfg.batch_norm,
            dropout=cfg.dropout,
            bias=cfg.bias,
            track_running_stats=cfg.track_running_stats,
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        scores: torch.Tensor = self.mlp(pooled)
        return scores
