"""Reusable neural-network training utilities: reproducibility, device, activations."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec ('auto' | 'cuda' | 'mps' | 'cpu' | ...) to a device.

    ``'auto'`` prefers CUDA, then Apple-silicon MPS, then CPU.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def same_device(left: torch.device, right: torch.device) -> bool:
    """Whether two device specs name the same device.

    ``torch.device("mps")`` and ``torch.device("mps:0")`` are the same device and
    compare unequal, because one carries an index and the other does not: a tensor
    reports the resolved form while :func:`resolve_device` returns the bare one.
    Comparing them with ``==`` is the reason a correct configuration can be
    rejected as a device mismatch. An unset index matches any index, since it
    means "the default one".
    """
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


def build_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name."""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
