"""Tests for :mod:`nekron.nn.utils`."""

from __future__ import annotations

import pytest
import torch

from nekron.nn.utils import build_activation, count_parameters, same_device

# --------------------------------------------------------------------------- #
# Device comparison
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("left", "right"),
    [("cpu", "cpu"), ("mps", "mps:0"), ("mps:0", "mps"), ("cuda", "cuda:0"), ("cuda:1", "cuda:1")],
)
def test_an_unset_index_matches_the_default_one(left: str, right: str) -> None:
    """A tensor reports ``mps:0`` while ``resolve_device`` returns ``mps``."""
    assert same_device(torch.device(left), torch.device(right))


@pytest.mark.parametrize(("left", "right"), [("cpu", "mps"), ("cuda:0", "cuda:1"), ("cpu", "meta")])
def test_different_devices_do_not_match(left: str, right: str) -> None:
    assert not same_device(torch.device(left), torch.device(right))


def test_a_real_tensor_matches_the_spec_it_was_placed_with() -> None:
    tensor = torch.zeros(2, device=torch.device("cpu"))
    assert same_device(tensor.device, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Activations and counting
# --------------------------------------------------------------------------- #


def test_unknown_activation_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="choose from"):
        build_activation("swish")


def test_activation_lookup_is_case_insensitive() -> None:
    assert isinstance(build_activation("GeLU"), torch.nn.GELU)


def test_count_parameters_counts_only_trainable_ones() -> None:
    layer = torch.nn.Linear(3, 2, bias=True)
    assert count_parameters(layer) == 8
    layer.bias.requires_grad_(False)
    assert count_parameters(layer) == 6
