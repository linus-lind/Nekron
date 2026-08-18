"""The encoder: a residual window in, one vector out.

Three stages, each a configurable block from :mod:`nekron.nn`, in the order the
model reads them:

1. a bidirectional :class:`~nekron.nn.lstm.LSTMEncoder` over the raw window, which
   turns each step into a state summarizing the window around it;
2. a :class:`~nekron.nn.transformer.TransformerEncoder` over that hidden-state
   sequence, which lets any step attend to any other rather than only to what the
   recurrence carried forward;
3. a :class:`~nekron.nn.pooling.TemporalPooling` reduction to one vector.

The two widths that connect them are never configured. The transformer is told
the LSTM's output width and projects it to ``d_model`` only if they differ, and
the pooling is told the transformer's — so no setting can claim a width that
disagrees with the one the layer before it actually produces.
"""

from __future__ import annotations

import torch
from torch import nn

from nekron.nn import LSTMEncoder, TemporalPooling, TransformerEncoder

from ..config import ModelConfig


class SequenceEncoder(nn.Module):
    """Encode ``[B, seq_len, channels]`` residual windows into ``[B, out_dim]``."""

    def __init__(self, cfg: ModelConfig, *, in_dim: int, seq_len: int) -> None:
        super().__init__()
        self.lstm = LSTMEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.lstm.hidden_dim,
            num_layers=cfg.lstm.num_layers,
            bias=cfg.lstm.bias,
            dropout=cfg.lstm.dropout,
            bidirectional=cfg.lstm.bidirectional,
            output_dropout=cfg.lstm.output_dropout,
        )
        self.transformer = TransformerEncoder(
            in_dim=self.lstm.out_dim,
            d_model=cfg.transformer.d_model,
            num_layers=cfg.transformer.num_layers,
            num_heads=cfg.transformer.num_heads,
            ff_dim=cfg.transformer.ff_dim,
            activation=cfg.transformer.activation,
            dropout=cfg.transformer.dropout,
            bias=cfg.transformer.bias,
            norm_position=cfg.transformer.norm_position,
            positional_encoding=cfg.transformer.positional_encoding,
            rope_base=cfg.transformer.rope_base,
            causal=cfg.transformer.causal,
            # The window length *is* the longest sequence the positional tables
            # ever see, so it is passed rather than configured a second time.
            max_len=seq_len,
        )
        self.pooling = TemporalPooling(
            in_dim=self.transformer.out_dim,
            kind=cfg.pooling.kind,
            attention_dim=cfg.pooling.attention_dim,
            dropout=cfg.pooling.dropout,
        )

    @property
    def out_dim(self) -> int:
        """Width of the pooled vector the score head reads."""
        return self.pooling.out_dim

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Reduce ``[B, L, C]`` to ``[B, out_dim]``."""
        hidden = self.lstm(sequences)
        attended = self.transformer(hidden)
        pooled: torch.Tensor = self.pooling(attended)
        return pooled
