"""Configurable Transformer encoder: attention, position, normalization placement.

A thin, configured wrapper around :class:`torch.nn.TransformerEncoderLayer` and
:class:`torch.nn.TransformerEncoder`. The blocks, the attention, the feed-forward
sublayer, the residual connections, the normalization placement and the parameter
initialization are all torch's; what is added here is the input projection, the
positional encoding, the causal mask, and the configuration surface that binds
them.

Everything a run would want to vary is a constructor argument: the width and
depth, the head count, the feed-forward width and activation, where the
normalization sits, which positional information is injected, and whether
attention may look ahead.

Positional information
----------------------
Three choices, and they differ in *where* they act, not only in how:

``"none"``
    Nothing is injected. Self-attention is then permutation-equivariant, so the
    stack sees the sequence as a set. That is a real configuration — an LSTM
    upstream has already ordered the sequence — not a degenerate one.
``"sinusoidal"``
    The classic fixed absolute encoding of Vaswani et al., *added to the input*
    once before the first block.
``"rope"``
    Rotary embeddings, applied to the queries and keys *inside every block*.
    Position enters the attention score as a rotation, so a score depends on the
    two positions only through their difference — the property absolute encodings
    do not have.

The one place torch cannot be used as-is
----------------------------------------
:class:`torch.nn.MultiheadAttention` projects, splits and attends in a single
call, and rotary embeddings have to be applied to the queries and keys *between*
the projection and the attention. There is no hook there. RoPE therefore needs
:class:`RotaryEncoderLayer`, which subclasses torch's encoder layer and replaces
one method — its attention block — while inheriting the layer's parameters,
initialization, feed-forward sublayer, norms and dropouts unchanged. The
attention itself is still :func:`torch.nn.functional.scaled_dot_product_attention`.

That subclass also overrides ``forward``, and *must*. Torch's own ``forward``
takes a fused fast path when the module is in evaluation mode under
``no_grad`` — the exact conditions inference runs under — and that path calls a
single fused kernel instead of ``_sa_block``. A layer that only replaced
``_sa_block`` would apply its rotation while training and silently drop it while
scoring. The override is torch's own slow path, verbatim.

Normalization placement
-----------------------
``norm_position="pre"`` maps to torch's ``norm_first=True``: each sublayer
normalizes its input, leaving an unnormalized path from input to output, so deep
stacks train without a warm-up. ``"post"`` normalizes the residual sum instead,
the original formulation. A pre-norm stack is given a final
:class:`~torch.nn.LayerNorm` through torch's ``norm`` argument, since otherwise
the last block's residual branch is never normalized; a post-norm stack already
ends in one.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import build_activation

POSITIONAL_ENCODINGS: tuple[str, ...] = ("none", "sinusoidal", "rope")
"""How position enters the stack; see the module docstring."""

NORM_POSITIONS: tuple[str, ...] = ("pre", "post")
"""Whether each sublayer normalizes its input or its residual sum."""


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal absolute positional encoding, added to the input.

    Position ``p``, dimension ``2i`` is ``sin(p / 10000^(2i/d))`` and dimension
    ``2i+1`` the matching cosine, so each dimension is a sinusoid whose wavelength
    grows geometrically from ``2*pi`` to ``10000 * 2*pi``. Torch ships no
    equivalent, so the table is built here.

    It is a *non-persistent* buffer: a deterministic function of ``(max_len,
    dim)`` that costs nothing to rebuild, so writing it into every checkpoint
    would only create a stale copy that a later change to ``max_len`` would fail
    to load.
    """

    def __init__(self, *, dim: int, max_len: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"sinusoidal encoding needs an even dim; got {dim}.")
        if max_len < 1:
            raise ValueError(f"max_len must be positive; got {max_len}.")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # exp(-log(10000) * 2i/d) rather than a direct power: the division is done
        # in log space, which keeps the smallest frequency exact at float32.
        frequency = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        table = torch.zeros(max_len, dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("table", table, persistent=False)
        self.max_len = max_len

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Add the encoding of the first ``L`` positions to ``[B, L, dim]``."""
        length = sequences.shape[1]
        if length > self.max_len:
            raise ValueError(
                f"sequence of length {length} exceeds the positional table's max_len "
                f"{self.max_len}."
            )
        table: torch.Tensor = self.get_buffer("table")
        return sequences + table[:length].to(sequences.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings applied to queries and keys.

    Each pair of head dimensions is rotated by an angle proportional to the
    position, with the pairs taken *half against half* — dimension ``i`` is paired
    with ``i + head_dim/2`` — which is the convention used by essentially every
    current PyTorch implementation. The defining property, verified in the tests,
    is that ``<rotate(q, m), rotate(k, n)>`` depends on ``m`` and ``n`` only
    through ``m - n``.

    Applied to ``q`` and ``k`` only, never to ``v``: position is meant to shape
    *which* values are attended to, not the values themselves.
    """

    def __init__(self, *, head_dim: int, max_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"rotary embeddings need an even head dimension; got {head_dim}. "
                "Choose d_model and num_heads so that d_model / num_heads is even."
            )
        if max_len < 1:
            raise ValueError(f"max_len must be positive; got {max_len}.")
        if base <= 1.0:
            raise ValueError(f"rope_base must exceed 1; got {base}.")
        # Angles are built in float64 and stored as float32: the smallest inverse
        # frequency is ~1e-4 and multiplying it by a position of several hundred
        # in float32 loses bits the invariance above is defined by.
        inverse = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        )  # [head_dim/2]
        angles = torch.outer(torch.arange(max_len, dtype=torch.float64), inverse)
        doubled = torch.cat((angles, angles), dim=-1)  # [max_len, head_dim]
        self.register_buffer("cos", doubled.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin", doubled.sin().to(torch.float32), persistent=False)
        self.max_len = max_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """``[x1, x2] -> [-x2, x1]`` over the trailing dimension's two halves."""
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``[B, H, L, head_dim]`` queries and keys by their position."""
        length = queries.shape[-2]
        if length > self.max_len:
            raise ValueError(
                f"sequence of length {length} exceeds the rotary table's max_len {self.max_len}."
            )
        cos = self.get_buffer("cos")[:length].to(queries.dtype).view(1, 1, length, -1)
        sin = self.get_buffer("sin")[:length].to(queries.dtype).view(1, 1, length, -1)
        rotated_q = queries * cos + self._rotate_half(queries) * sin
        rotated_k = keys * cos + self._rotate_half(keys) * sin
        return rotated_q, rotated_k


class RotaryEncoderLayer(nn.TransformerEncoderLayer):
    """:class:`torch.nn.TransformerEncoderLayer` with rotary queries and keys.

    Only the attention block differs. The layer's parameters, their
    initialization, the feed-forward sublayer, both layer norms, all three
    dropouts and the residual structure are torch's, inherited unchanged — so a
    rotary stack and an ordinary one differ in exactly one operation and cannot
    drift apart in any other.

    :meth:`forward` is torch's own slow path, restated. It has to be: torch's
    ``forward`` dispatches to a fused kernel under evaluation mode and
    ``no_grad``, which is precisely when a model is scored, and that kernel does
    not call :meth:`_sa_block`. Inheriting it would give a layer that rotates
    while training and does not while predicting.
    """

    def __init__(self, *args: object, rotary: RotaryPositionalEmbedding, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.rotary = rotary

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """One block, along torch's unfused path so the attention override applies."""
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            return x + self._ff_block(self.norm2(x))
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
        out: torch.Tensor = self.norm2(x + self._ff_block(x))
        return out

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-attention with the queries and keys rotated by their position.

        The projection uses :class:`~torch.nn.MultiheadAttention`'s own packed
        weights, so this layer holds exactly the parameters torch gave it and a
        state dict is interchangeable with an unrotated layer's.
        """
        if key_padding_mask is not None:
            raise NotImplementedError(
                "rotary attention does not take a key-padding mask; the sequences this "
                "encoder is built for are equal-length and never padded."
            )
        attention = self.self_attn
        batch, length, dim = x.shape
        head_dim = dim // attention.num_heads
        # [B, L, 3D] -> [3, B, H, Dh, L]: one reshape and one permute, matching the
        # (q, k, v) packing order of MultiheadAttention's in_proj_weight.
        packed = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        queries, keys, values = (
            packed.view(batch, length, 3, attention.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        ).unbind(0)
        queries, keys = self.rotary(queries, keys)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            # SDPA has no notion of train/eval, so the rate is gated here.
            dropout_p=attention.dropout if self.training else 0.0,
            # An explicit mask already encodes causality; passing both is an error.
            is_causal=is_causal and attn_mask is None,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, dim)
        dropped: torch.Tensor = self.dropout1(attention.out_proj(merged))
        return dropped


class TransformerEncoder(nn.Module):
    """A configured :class:`torch.nn.TransformerEncoder` over ``[B, L, in_dim]``.

    Returns the full sequence ``[B, L, d_model]``; reducing it to one vector is
    :class:`~nekron.nn.pooling.TemporalPooling`'s job.

    ``in_dim`` is the width of whatever produced the sequence — an
    :class:`~nekron.nn.lstm.LSTMEncoder`'s ``out_dim``, say — and is projected to
    ``d_model`` only when the two differ, so a stack configured to match its input
    carries no redundant identity map.

    ``max_len`` bounds the positional tables and the causal mask, and is taken
    from the data (the window length) rather than configured separately: two
    settings that must agree are one setting that can disagree.

    Parameters
    ----------
    in_dim:
        Channels per step of the input sequence.
    d_model:
        The stack's working width.
    num_layers:
        Number of blocks. Zero is allowed, carries no parameters at all, and
        yields the input projection alone — the control a sweep over depth needs.
    num_heads:
        Attention heads per block. Must divide ``d_model``, and under
        ``positional_encoding="rope"`` must divide it into an even head width.
    ff_dim:
        Width of each block's feed-forward hidden layer (torch's
        ``dim_feedforward``). The convention is ``4 * d_model``.
    activation:
        Feed-forward activation, resolved by
        :func:`~nekron.nn.utils.build_activation` and handed to torch as a module
        rather than by name. That keeps the whole activation registry available
        where torch's own string form accepts only ``relu`` and ``gelu``, and it
        keeps the activation visible in the module tree, which is what
        :class:`~nekron.nn.diagnostics.ActivationProbe` hooks.
    dropout:
        Torch's single dropout rate: applied to the attention weights, inside the
        feed-forward net, and to each sublayer's output before it re-enters the
        residual stream.
    bias:
        Whether the projections and norms carry bias terms.
    norm_position:
        ``"pre"`` or ``"post"``; see the module docstring.
    positional_encoding:
        One of :data:`POSITIONAL_ENCODINGS`.
    rope_base:
        Geometric base of the rotary frequencies. Ignored unless RoPE is used.
    causal:
        Whether a step may attend to later steps. ``False`` — the default — lets
        every step see the whole window, which is correct when the window lies
        entirely in the past of whatever the model is scoring. ``True`` builds
        torch's square subsequent mask and restricts each step to itself and its
        predecessors.
    max_len:
        Longest sequence the positional tables and the causal mask must cover.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        bias: bool,
        norm_position: str,
        positional_encoding: str,
        rope_base: float,
        causal: bool,
        max_len: int,
    ) -> None:
        super().__init__()
        if positional_encoding not in POSITIONAL_ENCODINGS:
            raise ValueError(
                f"positional_encoding must be one of {list(POSITIONAL_ENCODINGS)}; "
                f"got {positional_encoding!r}."
            )
        if norm_position not in NORM_POSITIONS:
            raise ValueError(
                f"norm_position must be one of {list(NORM_POSITIONS)}; got {norm_position!r}."
            )
        if num_layers < 0:
            raise ValueError(f"num_layers must not be negative; got {num_layers}.")
        if num_heads < 1:
            raise ValueError(f"num_heads must be positive; got {num_heads}.")
        if ff_dim < 1:
            raise ValueError(f"ff_dim must be positive; got {ff_dim}.")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.max_len = max_len
        self.causal = causal
        norm_first = norm_position == "pre"

        self.input_projection: nn.Module = (
            nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model, bias=bias)
        )
        self.positional_encoding: nn.Module = (
            SinusoidalPositionalEncoding(dim=d_model, max_len=max_len)
            if positional_encoding == "sinusoidal"
            else nn.Identity()
        )
        # One rotary module shared by every block: the tables depend only on the
        # head width and the length, and they are buffers, so a copy per block
        # would be identical numbers repeated `num_layers` times.
        self.rotary = (
            RotaryPositionalEmbedding(
                head_dim=d_model // num_heads, max_len=max_len, base=rope_base
            )
            if positional_encoding == "rope"
            else None
        )
        self.encoder = (
            self._build_stack(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                activation=activation,
                dropout=dropout,
                bias=bias,
                norm_first=norm_first,
            )
            if num_layers > 0
            # Not an empty torch stack: constructing one still builds its prototype
            # layer, so a zero-depth encoder would carry a block's worth of
            # parameters that nothing ever uses.
            else nn.Identity()
        )
        if causal:
            # Torch's own additive mask: 0 on and below the diagonal, -inf above.
            self.register_buffer(
                "causal_mask",
                nn.Transformer.generate_square_subsequent_mask(max_len),
                persistent=False,
            )

    def _build_stack(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        bias: bool,
        norm_first: bool,
    ) -> nn.TransformerEncoder:
        """Torch's encoder stack, with the rotary layer swapped in when needed."""
        arguments = {
            "d_model": d_model,
            "nhead": num_heads,
            "dim_feedforward": ff_dim,
            "dropout": dropout,
            "activation": build_activation(activation),
            "batch_first": True,
            "norm_first": norm_first,
            "bias": bias,
        }
        layer = (
            nn.TransformerEncoderLayer(**arguments)  # type: ignore[arg-type]
            if self.rotary is None
            else RotaryEncoderLayer(rotary=self.rotary, **arguments)
        )
        return nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            # A pre-norm stack leaves its last residual branch unnormalized; a
            # post-norm stack ends in a LayerNorm already.
            norm=nn.LayerNorm(d_model, bias=bias) if norm_first else None,
            # The windows this encoder is built for are equal-length and never
            # padded, so the nested-tensor path has nothing to skip; asking for it
            # only earns a warning under norm_first.
            enable_nested_tensor=False,
        )

    @property
    def out_dim(self) -> int:
        """Channels per step of the returned sequence."""
        return self.d_model

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, L, in_dim]`` into ``[B, L, d_model]``."""
        if sequences.ndim != 3:
            raise ValueError(
                f"expected [batch, time, channel] sequences; got {sequences.ndim} dimensions."
            )
        length = sequences.shape[1]
        if length > self.max_len:
            raise ValueError(
                f"sequence of length {length} exceeds max_len {self.max_len}; the encoder's "
                "positional tables were built for shorter windows."
            )
        hidden: torch.Tensor = self.positional_encoding(self.input_projection(sequences))
        if not isinstance(self.encoder, nn.TransformerEncoder):
            return hidden  # a zero-depth stack is the projection alone
        mask = self.get_buffer("causal_mask")[:length, :length] if self.causal else None
        out: torch.Tensor = self.encoder(hidden, mask=mask, is_causal=self.causal)
        return out
