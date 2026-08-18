"""Tests for :mod:`nekron.nn.proximal`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nekron.nn.config import OptimConfig, TrainConfig
from nekron.nn.diagnostics import DiagnosticsConfig, OptimizerProbe
from nekron.nn.proximal import ProximalL1, value
from nekron.nn.training import build_optimization

# --------------------------------------------------------------------------- #
# The operator itself
# --------------------------------------------------------------------------- #


def test_soft_threshold_shrinks_by_lr_times_lam_and_clamps_at_zero() -> None:
    weight = torch.tensor([[1.0, 0.05, -0.05, -1.0, 0.0]])
    ProximalL1(0.1)([weight], lr=1.0)
    assert torch.allclose(weight, torch.tensor([[0.9, 0.0, 0.0, -0.9, 0.0]]), atol=1e-6)


def test_the_threshold_scales_with_the_learning_rate() -> None:
    """``lr`` is the proximal operator's step, so a decayed rate anneals the pull."""
    weight = torch.tensor([[1.0]])
    ProximalL1(0.1)([weight], lr=0.5)
    assert weight.item() == pytest.approx(0.95)


def test_a_weight_never_crosses_zero() -> None:
    """The clamp is what distinguishes this from a subgradient step."""
    weight = torch.tensor([[0.01, -0.01]])
    ProximalL1(1.0)([weight], lr=1.0)
    assert torch.equal(weight, torch.zeros_like(weight))


def test_zero_lam_is_a_no_op() -> None:
    weight = torch.tensor([[0.3, -0.7]])
    ProximalL1(0.0)([weight], lr=1.0)
    assert torch.allclose(weight, torch.tensor([[0.3, -0.7]]))


def test_a_negative_lam_is_rejected() -> None:
    with pytest.raises(ValueError, match="lam must not be negative"):
        ProximalL1(-1e-6)


def test_penalized_selects_the_weight_matrices_only() -> None:
    layer = nn.Linear(3, 2)
    penalized = ProximalL1.penalized(layer.parameters())
    assert len(penalized) == 1
    assert penalized[0] is layer.weight


def test_value_sums_absolute_weights() -> None:
    weights = [torch.tensor([[1.0, -2.0]]), torch.tensor([[3.0]])]
    assert value(weights).item() == pytest.approx(6.0)
    assert value([]).item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Why it exists: the rate an adaptive optimizer actually applies
# --------------------------------------------------------------------------- #


def _shrink_with_zero_data_gradient(*, proximal: bool, steps: int = 200) -> float:
    """Magnitude left after ``steps`` on a weight whose only pressure is the penalty."""
    lr, lam = 1e-3, 1e-4
    # Double precision: a proximal step of lr * lam = 1e-7 is below float32's
    # resolution at 0.5 (eps ~ 6e-8), so single precision loses roughly a tenth of
    # the shrinkage to rounding and the rate cannot be read cleanly.
    layer = nn.Linear(4, 4, bias=False).double()
    with torch.no_grad():
        layer.weight.fill_(0.5)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=0.0)
    operator = ProximalL1(lam)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (layer.weight * 0.0).sum()
        if not proximal:
            loss = loss + lam * layer.weight.abs().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if proximal:
            operator([layer.weight], lr)
    return float(layer.weight.detach().abs().mean())


def test_a_subgradient_penalty_shrinks_at_lr_not_lr_times_lam() -> None:
    """The defect the proximal operator exists to avoid.

    An L1 subgradient has constant magnitude and stable sign, so Adam's
    normalization by gradient RMS divides ``lam`` out: the weight moves by the full
    ``lr`` each step. Over 200 steps at ``lr=1e-3`` that is 0.2 of magnitude, all of
    a weight that started at 0.5 being 40% erased by a penalty configured at 1e-4.
    """
    moved = 0.5 - _shrink_with_zero_data_gradient(proximal=False)
    assert moved == pytest.approx(200 * 1e-3, rel=0.05)


def test_the_proximal_operator_shrinks_at_the_configured_rate() -> None:
    moved = 0.5 - _shrink_with_zero_data_gradient(proximal=True)
    assert moved == pytest.approx(200 * 1e-3 * 1e-4, rel=0.05)


def test_a_zeroed_weight_can_leave_zero_again() -> None:
    """Reversibility: the property that keeps unit selection from being one-way."""
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=0.0)
    operator = ProximalL1(1e-4)
    for _ in range(10):
        optimizer.zero_grad()
        (layer.weight - 1.0).pow(2).sum().backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        operator([layer.weight], 1e-2)
    assert layer.weight.abs().min() > 0.0


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_build_optimization_uses_adamw() -> None:
    optimization = build_optimization(nn.Linear(2, 2), OptimConfig(), TrainConfig())
    assert isinstance(optimization.optimizer, torch.optim.AdamW)


def test_adamw_matches_adam_when_weight_decay_is_zero() -> None:
    """Why the swap can land under an existing configuration without moving it."""

    def run(cls: type[torch.optim.Optimizer]) -> torch.Tensor:
        torch.manual_seed(0)
        layer = nn.Linear(3, 3)
        optimizer = cls(layer.parameters(), lr=1e-2, weight_decay=0.0)  # type: ignore[call-arg]
        for _ in range(20):
            optimizer.zero_grad()
            layer.weight.pow(2).sum().backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        return layer.weight.detach().clone()

    assert torch.allclose(run(torch.optim.AdamW), run(torch.optim.Adam), atol=1e-7)


def test_the_probe_applies_the_operator_after_the_step() -> None:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.5)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
    probe = OptimizerProbe(
        layer,
        optimizer,
        DiagnosticsConfig(enabled=True),
        clip_norm=0.0,
        proximal=ProximalL1(0.1),
    )
    layer.weight.grad = torch.zeros_like(layer.weight)
    probe.step()
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.4))


def test_the_update_ratio_spans_the_proximal_shrinkage() -> None:
    """The shrinkage is part of the update, so the metric has to see it."""
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    # lr=1.0 with a zero gradient: the gradient step is a no-op, so the shrinkage
    # is the entire update. lr cannot be 0.0 here -- the probe reads it from the
    # optimizer, so it would zero the proximal threshold too.
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
    probe = OptimizerProbe(
        layer,
        optimizer,
        DiagnosticsConfig(enabled=True),
        clip_norm=0.0,
        proximal=ProximalL1(0.1),
    )
    layer.weight.grad = torch.zeros_like(layer.weight)
    probe.step()
    assert probe.summary()["opt/update_ratio"] == pytest.approx(0.1, rel=1e-4)


def test_no_proximal_operator_leaves_the_step_untouched() -> None:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.5)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
    probe = OptimizerProbe(layer, optimizer, DiagnosticsConfig(enabled=True), clip_norm=0.0)
    layer.weight.grad = torch.zeros_like(layer.weight)
    probe.step()
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.5))
