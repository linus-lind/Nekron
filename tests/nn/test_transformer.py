"""Tests for :mod:`nekron.nn.transformer`.

The interesting properties here are not shapes. They are the three things a
transformer implementation gets silently wrong: whether the positional encoding
actually encodes position the way it claims to, whether a causal stack really
cannot see ahead, and whether the two normalization placements are the two
formulations they are named after rather than one of them twice.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nekron.nn.transformer import (
    NORM_POSITIONS,
    POSITIONAL_ENCODINGS,
    RotaryEncoderLayer,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEncoder,
)


def _encoder(**overrides: object) -> TransformerEncoder:
    kwargs: dict[str, object] = {
        "in_dim": 8,
        "d_model": 8,
        "num_layers": 2,
        "num_heads": 2,
        "ff_dim": 16,
        "activation": "gelu",
        "dropout": 0.0,
        "bias": True,
        "norm_position": "pre",
        "positional_encoding": "none",
        "rope_base": 10000.0,
        "causal": False,
        "max_len": 12,
    }
    kwargs.update(overrides)
    return TransformerEncoder(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Positional encodings
# --------------------------------------------------------------------------- #


def test_sinusoidal_table_matches_the_published_formula() -> None:
    encoding = SinusoidalPositionalEncoding(dim=8, max_len=6)
    table = encoding.get_buffer("table")
    for position in range(6):
        for pair in range(4):
            angle = position / (10000.0 ** (2 * pair / 8))
            assert table[position, 2 * pair] == pytest.approx(math.sin(angle), abs=1e-6)
            assert table[position, 2 * pair + 1] == pytest.approx(math.cos(angle), abs=1e-6)


def test_sinusoidal_encoding_is_additive_and_length_sliced() -> None:
    encoding = SinusoidalPositionalEncoding(dim=4, max_len=10)
    sequences = torch.zeros(2, 3, 4)
    encoded = encoding(sequences)
    assert torch.equal(encoded[0], encoding.get_buffer("table")[:3])
    assert torch.equal(encoded[0], encoded[1])


def test_sinusoidal_rejects_a_sequence_longer_than_its_table() -> None:
    encoding = SinusoidalPositionalEncoding(dim=4, max_len=3)
    with pytest.raises(ValueError, match="exceeds"):
        encoding(torch.zeros(1, 4, 4))


def test_rope_scores_depend_only_on_relative_position() -> None:
    """The defining property: <rot(q, m), rot(k, n)> is a function of m - n alone."""
    rotary = RotaryPositionalEmbedding(head_dim=8, max_len=32, base=10000.0)
    torch.manual_seed(0)
    query = torch.randn(1, 1, 1, 8, dtype=torch.float32)
    key = torch.randn(1, 1, 1, 8, dtype=torch.float32)

    def score(left: int, right: int) -> float:
        cos = rotary.get_buffer("cos")
        sin = rotary.get_buffer("sin")
        rotated_q = query * cos[left] + rotary._rotate_half(query) * sin[left]
        rotated_k = key * cos[right] + rotary._rotate_half(key) * sin[right]
        return float((rotated_q * rotated_k).sum())

    same_gap = [score(m, m - 3) for m in (3, 7, 11, 20)]
    assert max(same_gap) - min(same_gap) < 1e-5
    assert abs(score(10, 6) - same_gap[0]) > 1e-3  # a different gap is a different score


def test_rope_needs_an_even_head_dim() -> None:
    with pytest.raises(ValueError, match="even head dimension"):
        RotaryPositionalEmbedding(head_dim=7, max_len=8, base=10000.0)


def test_positional_tables_stay_out_of_the_state_dict() -> None:
    """They are a pure function of (max_len, dim); a saved copy could only go stale."""
    for kind in ("sinusoidal", "rope"):
        state = _encoder(positional_encoding=kind).state_dict()
        assert not [name for name in state if name.endswith(("table", "cos", "sin"))]


def test_rope_extends_past_the_lengths_it_was_built_for_only_up_to_max_len() -> None:
    encoder = _encoder(positional_encoding="rope", max_len=6)
    with pytest.raises(ValueError, match="exceeds max_len"):
        encoder(torch.randn(1, 7, 8))


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


def test_a_plain_stack_is_torch_s_own_encoder() -> None:
    """The non-rotary path adds no attention code at all; it configures torch's."""
    encoder = _encoder(positional_encoding="none")
    assert isinstance(encoder.encoder, nn.TransformerEncoder)
    assert type(encoder.encoder.layers[0]) is nn.TransformerEncoderLayer
    assert encoder.rotary is None


def test_a_rotary_stack_is_torch_s_layer_with_one_method_replaced() -> None:
    encoder = _encoder(positional_encoding="rope")
    layer = encoder.encoder.layers[0]
    assert isinstance(layer, RotaryEncoderLayer)
    assert isinstance(layer, nn.TransformerEncoderLayer)
    # It holds exactly the parameters torch gave it: same names, same shapes.
    plain = _encoder(positional_encoding="none").encoder.layers[0]
    assert {n: tuple(p.shape) for n, p in layer.named_parameters()} == {
        n: tuple(p.shape) for n, p in plain.named_parameters()
    }


def test_the_rotary_layer_is_not_bypassed_by_torch_s_fused_fast_path() -> None:
    """The failure this guards against is silent and only appears at inference.

    Torch's ``TransformerEncoderLayer.forward`` dispatches to a fused kernel in
    evaluation mode under ``no_grad`` — exactly how a model is scored — and that
    kernel never calls ``_sa_block``. A layer that replaced only ``_sa_block``
    would rotate while training and quietly stop while predicting, so the two
    modes must agree.
    """
    encoder = _encoder(positional_encoding="rope").eval()
    sequences = torch.randn(2, 9, 8)
    with torch.no_grad():
        fused_path = encoder(sequences)
    unfused_path = encoder(sequences).detach()
    assert torch.allclose(fused_path, unfused_path, atol=1e-6)

    calls = 0
    original = type(encoder.encoder.layers[0])._sa_block

    def counting(self: object, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkey = type(encoder.encoder.layers[0])
    monkey._sa_block = counting  # type: ignore[assignment,method-assign]
    try:
        with torch.no_grad():
            encoder(sequences)
    finally:
        monkey._sa_block = original  # type: ignore[method-assign]
    assert calls == 2, f"the rotary attention ran {calls} times, expected once per layer"


def test_rope_changes_the_output_so_it_is_demonstrably_applied() -> None:
    torch.manual_seed(0)
    sequences = torch.randn(2, 9, 8)
    plain = _encoder(positional_encoding="none").eval()
    rotary = _encoder(positional_encoding="rope").eval()
    # Same parameters, so any difference is the rotation and nothing else.
    rotary.load_state_dict(plain.state_dict())
    with torch.no_grad():
        assert not torch.allclose(plain(sequences), rotary(sequences), atol=1e-4)


def test_a_rotary_state_dict_is_interchangeable_with_a_plain_one() -> None:
    """Same parameters under the same names, so a checkpoint survives the switch."""
    plain = _encoder(positional_encoding="none")
    rotary = _encoder(positional_encoding="rope")
    assert set(plain.state_dict()) == set(rotary.state_dict())
    rotary.load_state_dict(plain.state_dict())


def test_rotary_attention_refuses_a_key_padding_mask() -> None:
    """Silently ignoring one would be worse; these windows are never padded."""
    layer = _encoder(positional_encoding="rope").encoder.layers[0]
    with pytest.raises(NotImplementedError, match="key-padding mask"):
        layer(torch.randn(1, 4, 8), src_key_padding_mask=torch.zeros(1, 4, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("positional_encoding", POSITIONAL_ENCODINGS)
@pytest.mark.parametrize("norm_position", NORM_POSITIONS)
def test_causal_stack_cannot_see_ahead(positional_encoding: str, norm_position: str) -> None:
    """Rewriting the tail of a window must leave every earlier step untouched."""
    encoder = _encoder(
        causal=True, positional_encoding=positional_encoding, norm_position=norm_position
    ).eval()
    first = torch.randn(1, 10, 8)
    second = first.clone()
    second[:, 5:] = torch.randn(1, 5, 8)
    with torch.no_grad():
        assert torch.allclose(encoder(first)[:, :5], encoder(second)[:, :5], atol=1e-6)


def test_non_causal_stack_does_see_ahead() -> None:
    """Otherwise the causal flag would be indistinguishable from a no-op.

    The tail is *replaced*, not shifted. Adding a constant to every channel of a
    step is invisible here: layer normalization subtracts each step's own mean, so
    a uniform shift is exactly the perturbation this stack is built to ignore.
    """
    torch.manual_seed(0)
    encoder = _encoder(causal=False).eval()
    first = torch.randn(1, 10, 8)
    second = first.clone()
    second[:, 5:] = torch.randn(1, 5, 8)
    with torch.no_grad():
        assert not torch.allclose(encoder(first)[:, :5], encoder(second)[:, :5], atol=1e-3)


# --------------------------------------------------------------------------- #
# Normalization placement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("norm_position", "norm_first"), [("pre", True), ("post", False)])
def test_norm_position_maps_onto_torch_s_norm_first(norm_position: str, norm_first: bool) -> None:
    """The two placements are torch's two, not one of them spelled twice."""
    encoder = _encoder(norm_position=norm_position)
    assert encoder.encoder.layers[0].norm_first is norm_first


@pytest.mark.parametrize("norm_position", NORM_POSITIONS)
def test_a_stack_matches_torch_s_own_encoder_built_the_same_way(norm_position: str) -> None:
    """Weights copied across, the two must agree to floating-point noise."""
    ours = _encoder(
        in_dim=8, d_model=8, num_layers=2, norm_position=norm_position, positional_encoding="none"
    ).eval()
    layer = nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        activation=nn.GELU(),
        batch_first=True,
        norm_first=norm_position == "pre",
        bias=True,
    )
    theirs = nn.TransformerEncoder(
        layer,
        num_layers=2,
        norm=nn.LayerNorm(8) if norm_position == "pre" else None,
        enable_nested_tensor=False,
    ).eval()
    theirs.load_state_dict(ours.encoder.state_dict())
    sequences = torch.randn(3, 7, 8)
    with torch.no_grad():
        assert torch.allclose(ours(sequences), theirs(sequences), atol=1e-6)


def test_only_a_pre_norm_stack_carries_a_final_layer_norm() -> None:
    assert isinstance(_encoder(norm_position="pre").encoder.norm, nn.LayerNorm)
    assert _encoder(norm_position="post").encoder.norm is None


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_input_projection_is_identity_only_when_widths_agree() -> None:
    assert isinstance(_encoder(in_dim=8, d_model=8).input_projection, nn.Identity)
    assert isinstance(_encoder(in_dim=5, d_model=8).input_projection, nn.Linear)


def test_empty_stack_still_projects_to_d_model() -> None:
    encoder = _encoder(in_dim=5, d_model=8, num_layers=0)
    assert encoder(torch.randn(2, 4, 5)).shape == (2, 4, 8)


def test_an_empty_stack_carries_no_block_parameters() -> None:
    """Torch would still build its prototype layer; those weights must not linger."""
    empty = _encoder(in_dim=8, d_model=8, num_layers=0)
    assert list(empty.parameters()) == []


@pytest.mark.parametrize("positional_encoding", POSITIONAL_ENCODINGS)
@pytest.mark.parametrize("norm_position", NORM_POSITIONS)
def test_every_combination_runs_and_backpropagates(
    positional_encoding: str, norm_position: str
) -> None:
    encoder = _encoder(
        in_dim=6,
        d_model=8,
        dropout=0.1,
        positional_encoding=positional_encoding,
        norm_position=norm_position,
    )
    out = encoder(torch.randn(3, 7, 6))
    assert out.shape == (3, 7, 8)
    out.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())


def test_out_dim_reports_d_model() -> None:
    assert _encoder(d_model=8).out_dim == 8


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_rejects_heads_that_do_not_divide_the_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _encoder(d_model=8, num_heads=3)


def test_rejects_an_unknown_positional_encoding() -> None:
    with pytest.raises(ValueError, match="positional_encoding"):
        _encoder(positional_encoding="learned")


def test_rejects_an_unknown_norm_position() -> None:
    with pytest.raises(ValueError, match="norm_position"):
        _encoder(norm_position="middle")


def test_rejects_rope_with_an_odd_head_width() -> None:
    with pytest.raises(ValueError, match="even head dimension"):
        _encoder(d_model=6, num_heads=2, positional_encoding="rope")
