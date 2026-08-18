"""Configurable temporal aggregation: a sequence ``[B, L, D]`` to one vector ``[B, P]``.

The last step of a sequence encoder, and a real hyperparameter rather than a
formality — the reductions below disagree about *where in the window* the signal
is, and which is right is a property of the data:

``"last"``
    The final step alone. The natural read-out when the encoder is recurrent or
    causal, since that step is the only one that has seen the whole window; it is
    also the one that throws away the most.
``"mean"``
    Every step weighted equally. Robust and low-variance, but it dilutes a signal
    that lives in a short stretch of the window.
``"max"``
    Per channel, the largest value over the window. Picks out whether a feature
    ever fired, at the cost of ignoring how long it did.
``"attention"``
    Learned additive attention over steps: the model is given the weights instead
    of being told them. The only kind with parameters, and the only one that can
    put the mass on a stretch of the window that moves between examples.
``"last_mean"``, ``"mean_max"``
    Two reductions concatenated, so the head sees both. They double the output
    width, which :attr:`TemporalPooling.out_dim` reports.

Every reduction here is over the whole window, including under a causal encoder:
causality constrains what step ``t`` may have *seen*, and the whole window is
already in the past of whatever the pooled vector is used to score.
"""

from __future__ import annotations

import torch
from torch import nn

POOLING_KINDS: tuple[str, ...] = ("last", "mean", "max", "attention", "last_mean", "mean_max")
"""Supported temporal reductions; see the module docstring."""

_CONCATENATING: frozenset[str] = frozenset({"last_mean", "mean_max"})
"""Kinds whose output is two reductions side by side, hence twice as wide."""


class TemporalPooling(nn.Module):
    """Reduce ``[B, L, in_dim]`` over time to ``[B, out_dim]``.

    Parameters
    ----------
    in_dim:
        Channels per step of the input sequence.
    kind:
        One of :data:`POOLING_KINDS`.
    attention_dim:
        Width of the learned attention's hidden layer. Ignored by every other
        kind.
    dropout:
        Applied to the pooled vector. Zero disables it.
    """

    def __init__(self, *, in_dim: int, kind: str, attention_dim: int, dropout: float) -> None:
        super().__init__()
        if kind not in POOLING_KINDS:
            raise ValueError(f"pooling kind must be one of {list(POOLING_KINDS)}; got {kind!r}.")
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive; got {in_dim}.")
        if kind == "attention" and attention_dim < 1:
            raise ValueError(f"attention_dim must be positive; got {attention_dim}.")
        self.kind = kind
        self._out_dim = in_dim * (2 if kind in _CONCATENATING else 1)
        # A single scalar score per step, from a one-hidden-layer network: the
        # additive ("Bahdanau") form, which needs no notion of a query and so has
        # nothing to attend *from* — appropriate when the whole point is to
        # summarize a sequence rather than to align two of them.
        self.score: nn.Module = (
            nn.Sequential(
                nn.Linear(in_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1, bias=False),
            )
            if kind == "attention"
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def out_dim(self) -> int:
        """Width of the pooled vector — twice ``in_dim`` for the concatenating kinds."""
        return self._out_dim

    def _reduce(self, sequences: torch.Tensor) -> torch.Tensor:
        if self.kind == "last":
            return sequences[:, -1, :]
        if self.kind == "mean":
            return sequences.mean(dim=1)
        if self.kind == "max":
            return sequences.amax(dim=1)
        if self.kind == "attention":
            weights = torch.softmax(self.score(sequences), dim=1)  # [B, L, 1]
            return (sequences * weights).sum(dim=1)
        if self.kind == "last_mean":
            return torch.cat((sequences[:, -1, :], sequences.mean(dim=1)), dim=-1)
        return torch.cat((sequences.mean(dim=1), sequences.amax(dim=1)), dim=-1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, L, in_dim]`` down to ``[B, out_dim]``."""
        if sequences.ndim != 3:
            raise ValueError(
                f"temporal pooling needs a [batch, time, channel] tensor; got "
                f"{sequences.ndim} dimensions."
            )
        if sequences.shape[1] == 0:
            raise ValueError("cannot pool a sequence of length zero.")
        out: torch.Tensor = self.dropout(self._reduce(sequences))
        return out
