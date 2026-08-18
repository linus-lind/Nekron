"""Tests for :mod:`nekron.nn.lstm`."""

from __future__ import annotations

import warnings

import pytest
import torch

from nekron.nn.lstm import LSTMEncoder


def _encoder(**overrides: object) -> LSTMEncoder:
    kwargs: dict[str, object] = {
        "in_dim": 3,
        "hidden_dim": 8,
        "num_layers": 2,
        "bias": True,
        "dropout": 0.0,
        "bidirectional": False,
        "output_dropout": 0.0,
    }
    kwargs.update(overrides)
    return LSTMEncoder(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("bidirectional", "expected"), [(False, 8), (True, 16)])
def test_out_dim_doubles_when_bidirectional(bidirectional: bool, expected: int) -> None:
    encoder = _encoder(bidirectional=bidirectional)
    assert encoder.out_dim == expected


def test_returns_the_whole_hidden_sequence() -> None:
    encoder = _encoder(bidirectional=True)
    out = encoder(torch.randn(4, 11, 3))
    assert out.shape == (4, 11, encoder.out_dim)


def test_out_dim_matches_what_the_layer_actually_produces() -> None:
    """The property is what downstream widths are derived from; it must not lie."""
    for bidirectional in (False, True):
        for hidden in (1, 5, 8):
            encoder = _encoder(hidden_dim=hidden, bidirectional=bidirectional)
            assert encoder(torch.randn(2, 6, 3)).shape[-1] == encoder.out_dim


# --------------------------------------------------------------------------- #
# Directionality
# --------------------------------------------------------------------------- #


def test_unidirectional_output_is_prefix_stable() -> None:
    """A forward-only LSTM's step ``t`` cannot depend on anything after ``t``."""
    encoder = _encoder(bidirectional=False).eval()
    first = torch.randn(1, 10, 3)
    second = first.clone()
    second[:, 5:] = torch.randn(1, 5, 3)
    with torch.no_grad():
        assert torch.allclose(encoder(first)[:, :5], encoder(second)[:, :5], atol=1e-6)


def test_bidirectional_output_is_not_prefix_stable() -> None:
    """The reverse pass mixes the whole window, which is why it needs a caveat."""
    torch.manual_seed(0)
    encoder = _encoder(bidirectional=True).eval()
    first = torch.randn(1, 10, 3)
    second = first.clone()
    second[:, 5:] = torch.randn(1, 5, 3)
    with torch.no_grad():
        assert not torch.allclose(encoder(first)[:, :5], encoder(second)[:, :5], atol=1e-4)


# --------------------------------------------------------------------------- #
# Dropout handling
# --------------------------------------------------------------------------- #


def test_single_layer_dropout_is_dropped_without_torch_warning() -> None:
    """torch warns that its ``dropout`` is inert with one layer; we warn instead."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder = _encoder(num_layers=1, dropout=0.5)
    assert not [w for w in caught if "recurrent layer" in str(w.message)]
    assert encoder.lstm.dropout == 0.0


def test_multi_layer_dropout_is_passed_through() -> None:
    assert _encoder(num_layers=2, dropout=0.25).lstm.dropout == 0.25


def test_output_dropout_is_identity_when_zero() -> None:
    """Zero must not leave a Dropout module in the graph that eval mode has to undo."""
    assert isinstance(_encoder(output_dropout=0.0).dropout, torch.nn.Identity)
    assert isinstance(_encoder(output_dropout=0.1).dropout, torch.nn.Dropout)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [("in_dim", 0), ("hidden_dim", 0), ("num_layers", 0), ("num_layers", -1)],
)
def test_rejects_non_positive_dimensions(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        _encoder(**{field: value})
