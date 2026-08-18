"""Configurable LSTM sequence encoder."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """Multi-layer, optionally bidirectional LSTM over ``[B, L, C]`` sequences.

    Returns the *whole* hidden-state sequence ``[B, L, out_dim]`` rather than a
    single summary vector: what follows an encoder in this project is attention,
    which needs one vector per step. Reducing the sequence to a vector is a
    separate, separately configurable decision — see
    :class:`~nekron.nn.pooling.TemporalPooling`.

    ``out_dim`` is :attr:`hidden_dim` doubled when :attr:`bidirectional`, because
    the two directions are concatenated per step. It is a property rather than a
    setting so nothing downstream can declare a width that disagrees with the one
    the layer actually produces.

    Parameters
    ----------
    in_dim:
        Channels per step of the input sequence.
    hidden_dim:
        Hidden size *per direction*.
    num_layers:
        Stacked LSTM layers.
    bias:
        Whether the recurrent cells carry bias terms.
    dropout:
        Dropout applied *between* stacked layers — torch's own ``dropout``, which
        does nothing at all with a single layer. A positive value with
        ``num_layers=1`` is dropped with a warning rather than passed on, which is
        what keeps a sweep over ``num_layers`` from emitting torch's warning on
        every one-layer configuration.
    bidirectional:
        Whether to run a second pass in reverse and concatenate it.
    output_dropout:
        Dropout applied to the returned sequence. Distinct from :attr:`dropout`,
        which torch applies only between layers and therefore never after the
        last one.

    Notes
    -----
    ``batch_first=True`` is fixed, not configured: every sequence tensor in this
    project is ``[batch, time, channel]``, and a per-model switch would make the
    layout a property of the configuration rather than of the code.

    Sequences are assumed to be equal-length and gap-free, so no packing is done.
    That holds by construction for the windowed datasets in
    :mod:`nekron.time_series_modelling` — a window exists only where the whole
    span is present — and a caller with ragged sequences must pack them itself.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        bidirectional: bool,
        output_dropout: float,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive; got {in_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive; got {num_layers}.")
        recurrent_dropout = dropout
        if num_layers == 1 and dropout > 0.0:
            logger.warning(
                "lstm.dropout=%.3g is ignored with num_layers=1: torch applies it between "
                "stacked layers only. Use output_dropout to drop the returned sequence.",
                dropout,
            )
            recurrent_dropout = 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(output_dropout) if output_dropout > 0.0 else nn.Identity()
        self._out_dim = hidden_dim * (2 if bidirectional else 1)

    @property
    def out_dim(self) -> int:
        """Channels per step of the returned sequence."""
        return self._out_dim

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, L, in_dim]`` into ``[B, L, out_dim]``."""
        hidden, _ = self.lstm(sequences)
        out: torch.Tensor = self.dropout(hidden)
        return out
