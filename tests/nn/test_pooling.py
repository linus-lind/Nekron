"""Tests for :mod:`nekron.nn.pooling`."""

from __future__ import annotations

import pytest
import torch

from nekron.nn.pooling import POOLING_KINDS, TemporalPooling

# One batch, three steps, two channels — small enough to write the answers down.
SEQUENCE = torch.tensor([[[1.0, -1.0], [2.0, 5.0], [3.0, 0.0]]])


def _pool(kind: str, *, in_dim: int = 2, attention_dim: int = 4) -> TemporalPooling:
    return TemporalPooling(in_dim=in_dim, kind=kind, attention_dim=attention_dim, dropout=0.0)


# --------------------------------------------------------------------------- #
# What each reduction actually computes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("last", [3.0, 0.0]),
        ("mean", [2.0, 4.0 / 3.0]),
        ("max", [3.0, 5.0]),
        ("last_mean", [3.0, 0.0, 2.0, 4.0 / 3.0]),
        ("mean_max", [2.0, 4.0 / 3.0, 3.0, 5.0]),
    ],
)
def test_reduction_values(kind: str, expected: list[float]) -> None:
    pooled = _pool(kind)(SEQUENCE)
    assert pooled.shape == (1, len(expected))
    assert torch.allclose(pooled, torch.tensor([expected]), atol=1e-6)


def test_attention_reduces_to_the_mean_when_every_step_scores_alike() -> None:
    """A zeroed score network makes the softmax uniform, which is the mean."""
    pooling = _pool("attention")
    with torch.no_grad():
        for parameter in pooling.score.parameters():
            parameter.zero_()
    assert torch.allclose(pooling(SEQUENCE), SEQUENCE.mean(dim=1), atol=1e-6)


def test_attention_weights_are_a_distribution_over_time() -> None:
    pooling = _pool("attention")
    weights = torch.softmax(pooling.score(SEQUENCE), dim=1)
    assert weights.shape == (1, 3, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1, 1), atol=1e-6)


def test_attention_can_put_its_mass_on_one_step() -> None:
    """Otherwise it would only ever be an expensive mean."""
    pooling = _pool("attention")
    with torch.no_grad():
        for parameter in pooling.score.parameters():
            parameter.zero_()
        # Score the second channel only, hugely: step 1 has the largest value there.
        pooling.score[0].weight[0, 1] = 50.0
        pooling.score[2].weight[0, 0] = 50.0
    assert torch.allclose(pooling(SEQUENCE), SEQUENCE[:, 1, :], atol=1e-4)


# --------------------------------------------------------------------------- #
# Declared width
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", POOLING_KINDS)
def test_out_dim_matches_what_the_layer_produces(kind: str) -> None:
    pooling = _pool(kind, in_dim=5)
    assert pooling(torch.randn(4, 7, 5)).shape == (4, pooling.out_dim)


@pytest.mark.parametrize("kind", POOLING_KINDS)
def test_only_learned_attention_has_parameters(kind: str) -> None:
    parameters = list(_pool(kind).parameters())
    assert bool(parameters) == (kind == "attention")


@pytest.mark.parametrize("kind", POOLING_KINDS)
def test_gradients_flow_through_every_kind(kind: str) -> None:
    sequences = torch.randn(2, 6, 3, requires_grad=True)
    _pool(kind, in_dim=3)(sequences).sum().backward()
    assert sequences.grad is not None
    assert torch.isfinite(sequences.grad).all()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="pooling kind"):
        _pool("median")


def test_rejects_a_non_positive_attention_width() -> None:
    with pytest.raises(ValueError, match="attention_dim"):
        _pool("attention", attention_dim=0)


def test_rejects_an_unbatched_sequence() -> None:
    with pytest.raises(ValueError, match=r"\[batch, time, channel\]"):
        _pool("mean")(torch.randn(4, 2))


def test_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="length zero"):
        _pool("mean")(torch.randn(1, 0, 2))
