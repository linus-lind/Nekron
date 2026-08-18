"""Configurable multilayer perceptron building block."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .utils import build_activation


class MLP(nn.Module):
    """Fully connected stack ``in_dim -> hidden_dims -> out_dim``.

    Each hidden layer applies a linear map, an optional ``BatchNorm1d``, the chosen
    activation and optional dropout; the output layer is a bare linear map with no
    activation or normalization. An empty ``hidden_dims`` yields a single linear
    layer. Batch normalization requires a leading batch dimension in the input.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str,
        batch_norm: bool,
        dropout: float,
        bias: bool,
        track_running_stats: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(width, hidden, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden, track_running_stats=track_running_stats))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, out_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out
