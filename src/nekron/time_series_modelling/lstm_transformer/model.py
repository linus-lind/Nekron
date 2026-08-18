"""The LSTM-Transformer residual model: encoder plus score head as one ``nn.Module``.

Forward pass over a batch of residual windows:

    residual window [B, L, C]  --> bidirectional LSTM  --> hidden states [B, L, H]
    hidden states              --> transformer stack   --> attended  [B, L, D]
    attended                   --> temporal pooling    --> pooled    [B, P]
    pooled                     --> MLP score head      --> scores    [B, K]

Every width after the first is determined by the stage before it; the only one
that comes from outside is ``C``, the number of channels a residual window
carries, which the data decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nekron.nn import count_parameters

from .config import LstmTransformerConfig, ModelConfig
from .modules.score_head import ScoreHead
from .modules.sequence_encoder import SequenceEncoder


@dataclass
class ScoreOutput:
    """Tensors produced by a forward pass over one batch of windows."""

    scores: torch.Tensor  # [B, K] the head's output
    pooled: torch.Tensor  # [B, P] the encoder's window representation


class LstmTransformer(nn.Module):
    """Scores a fixed-length window of one entity's autoencoder residuals.

    The window is assumed to lie entirely in the past of whatever the score is
    for, which is what makes a bidirectional recurrence and unrestricted attention
    legitimate here: both mix information *within* the window, never across its
    leading edge. Restricting the attention as well is
    ``model.transformer.causal``; note that it constrains the attention only, so a
    model that is causal step by step also needs
    ``model.lstm.bidirectional=false``.
    """

    def __init__(self, cfg: ModelConfig, *, in_dim: int, seq_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.encoder = SequenceEncoder(cfg, in_dim=in_dim, seq_len=seq_len)
        self.head = ScoreHead(cfg.head, in_dim=self.encoder.out_dim)

    @classmethod
    def from_config(
        cls, cfg: LstmTransformerConfig, *, in_dim: int, seq_len: int
    ) -> LstmTransformer:
        """Construct the model with the input width inferred from the windows."""
        return cls(cfg.model, in_dim=in_dim, seq_len=seq_len)

    def forward(self, sequences: torch.Tensor) -> ScoreOutput:
        """Score ``[B, seq_len, in_dim]`` residual windows."""
        if sequences.ndim != 3:
            raise ValueError(
                f"expected [batch, time, channel] windows; got {sequences.ndim} dimensions."
            )
        if sequences.shape[1] != self.seq_len:
            raise ValueError(
                f"the model was built for windows of {self.seq_len} steps; got "
                f"{sequences.shape[1]}."
            )
        pooled = self.encoder(sequences)
        return ScoreOutput(scores=self.head(pooled), pooled=pooled)

    def num_parameters(self) -> int:
        return count_parameters(self)
