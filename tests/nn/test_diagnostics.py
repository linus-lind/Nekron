"""Tests for :mod:`nekron.nn.diagnostics`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nekron.nn.diagnostics import (
    _SATURATION,
    ActivationProbe,
    DiagnosticsConfig,
    OptimizerProbe,
    dead_fraction,
    saturated_fraction,
    weight_summary,
)


def _linear(weight: list[list[float]]) -> nn.Linear:
    """A bias-free linear layer with the given weight, for exact arithmetic."""
    layer = nn.Linear(len(weight[0]), len(weight), bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight))
    return layer


def _probe(model: nn.Module, *, clip_norm: float = 0.0, **kwargs: object) -> OptimizerProbe:
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    cfg = DiagnosticsConfig(**kwargs)  # type: ignore[arg-type]
    return OptimizerProbe(model, optimizer, cfg, clip_norm=clip_norm)


def _set_grad(layer: nn.Linear, grad: list[list[float]]) -> None:
    layer.weight.grad = torch.tensor(grad)


# --------------------------------------------------------------------------- #
# Activation saturation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(cls.__name__ for cls in _SATURATION))
def test_saturation_tests_agree_with_the_activations_own_gradient(name: str) -> None:
    """The output-space test must select the units that pass (almost) no gradient.

    The whole point of the metric is "how much of this layer is not learning", so
    the cheap output-space predicate has to track the expensive ground truth —
    ``|f'(x)|`` small — rather than merely correlate with it. Checked over the input
    range the activations actually see.
    """
    module = getattr(nn, name)()
    x = torch.linspace(-8.0, 8.0, 20001, requires_grad=True)
    y = module(x)
    (slope,) = torch.autograd.grad(y.sum(), x)

    # Leaky ReLU's negative branch has slope 0.01: heavily attenuated rather than
    # flat, which is what the metric is meant to surface, so it is compared against
    # a threshold above its own slope.
    cutoff = 0.02 if name == "LeakyReLU" else 1e-2
    flat = slope.abs() < cutoff
    predicted = _SATURATION[type(module)](module, y.detach(), 1e-2)

    agreement = (flat == predicted).to(torch.float64).mean()
    assert float(agreement) > 0.95


def test_saturated_fraction_counts_a_fully_dead_relu() -> None:
    relu = nn.ReLU()
    assert float(saturated_fraction(relu, relu(torch.full((4, 8), -3.0)), 1e-2)) == 1.0
    assert float(saturated_fraction(relu, relu(torch.full((4, 8), 3.0)), 1e-2)) == 0.0


def test_saturated_fraction_rejects_an_unregistered_activation() -> None:
    with pytest.raises(KeyError, match="no saturation test registered"):
        saturated_fraction(nn.Identity(), torch.zeros(4), 1e-2)


def test_dead_fraction_counts_units_no_example_activates() -> None:
    """The sharper reading: a unit off for the whole batch gets no gradient at all.

    Both columns here are half zero, so the per-output saturated fraction cannot
    tell them apart from a layer whose second unit never fires.
    """
    relu = nn.ReLU()
    # Column 0 fires for one row; column 1 never fires.
    out = relu(torch.tensor([[1.0, -1.0], [-1.0, -1.0]]))

    assert float(saturated_fraction(relu, out, 1e-2)) == pytest.approx(0.75)
    assert float(dead_fraction(relu, out, 1e-2)) == pytest.approx(0.5)


def test_dead_fraction_needs_a_batch_to_be_dead_across() -> None:
    relu = nn.ReLU()
    with pytest.raises(ValueError, match="batched output"):
        dead_fraction(relu, relu(torch.tensor([-1.0, 1.0])), 1e-2)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def test_weight_summary_ignores_biases_and_counts_zeros() -> None:
    """Biases must not dilute the sparsity a LASSO penalty is judged by."""
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0, 0.0, 0.0]]))
        model.bias.copy_(torch.tensor([7.0]))

    summary = weight_summary(model, DiagnosticsConfig())

    assert summary["weights/norm"] == pytest.approx(5.0)  # the bias is excluded
    assert summary["weights/zero_frac"] == pytest.approx(0.5)


def test_weight_summary_is_inert_when_disabled() -> None:
    assert weight_summary(nn.Linear(2, 2), DiagnosticsConfig(enabled=False)) == {}


# --------------------------------------------------------------------------- #
# Optimizer probe
# --------------------------------------------------------------------------- #


def test_reported_gradient_norm_is_the_one_before_clipping() -> None:
    """Reading the norm after clipping would report the threshold back at itself.

    That is the failure that makes the metric useless: every clipped step would
    read exactly ``grad_clip_norm``, so the distribution the threshold is supposed
    to be chosen from is precisely the part that has been erased.
    """
    layer = _linear([[1.0, 1.0]])
    probe = _probe(layer, clip_norm=1.0)
    _set_grad(layer, [[6.0, 8.0]])  # norm 10, well past the threshold

    probe.step()
    summary = probe.summary()

    assert summary["grad/norm"] == pytest.approx(10.0)
    assert summary["grad/clipped_frac"] == pytest.approx(1.0)


def test_clipping_actually_rescales_the_gradient() -> None:
    layer = _linear([[1.0, 1.0]])
    probe = _probe(layer, clip_norm=1.0)
    _set_grad(layer, [[6.0, 8.0]])

    probe.step()

    assert layer.weight.grad is not None
    assert float(layer.weight.grad.norm()) == pytest.approx(1.0, rel=1e-5)


def test_no_clipping_leaves_the_gradient_alone_but_still_measures_it() -> None:
    layer = _linear([[1.0, 1.0]])
    probe = _probe(layer, clip_norm=0.0)
    _set_grad(layer, [[6.0, 8.0]])

    probe.step()
    summary = probe.summary()

    assert layer.weight.grad is not None
    assert float(layer.weight.grad.norm()) == pytest.approx(10.0)
    assert summary["grad/norm"] == pytest.approx(10.0)
    # An inert threshold has no share of steps to report.
    assert "grad/clipped_frac" not in summary


def test_gradient_percentile_is_the_threshold_to_configure() -> None:
    """``grad/norm_p90`` is read straight back into ``grad_clip_norm``."""
    layer = _linear([[1.0, 0.0]])
    probe = _probe(layer, clip_norm=0.0)
    for norm in range(1, 11):  # norms 1..10
        _set_grad(layer, [[float(norm), 0.0]])
        probe.step()

    summary = probe.summary()

    assert summary["grad/norm"] == pytest.approx(5.5)
    assert summary["grad/norm_p90"] == pytest.approx(9.1)


def test_largest_gradient_norm_says_whether_clipping_is_needed_at_all() -> None:
    layer = _linear([[1.0, 0.0]])
    probe = _probe(layer, clip_norm=0.0)
    for norm in (1.0, 1.0, 1.0, 50.0):
        _set_grad(layer, [[float(norm), 0.0]])
        probe.step()

    assert probe.summary()["grad/norm_max"] == pytest.approx(50.0)


def test_a_nonfinite_gradient_is_reported_instead_of_poisoning_the_statistics() -> None:
    """A NaN norm would make every gradient statistic NaN, and the tracker drops NaNs.

    The chart would then simply stop, with nothing saying why.
    """
    layer = _linear([[1.0, 0.0]])
    probe = _probe(layer, clip_norm=0.0)
    _set_grad(layer, [[3.0, 0.0]])
    probe.step()
    _set_grad(layer, [[float("nan"), 0.0]])
    probe.step()

    summary = probe.summary()

    assert summary["grad/norm"] == pytest.approx(3.0)
    assert summary["grad/nonfinite_frac"] == pytest.approx(0.5)


def test_update_ratio_measures_the_step_the_optimizer_took() -> None:
    """The scale-free learning-rate reading, against an exactly known SGD step."""
    layer = _linear([[3.0, 4.0]])  # ||W|| = 5
    probe = _probe(layer, clip_norm=0.0)  # SGD, lr=1 -> update is -grad
    _set_grad(layer, [[0.03, 0.04]])  # ||dW|| = 0.05

    probe.step()

    assert probe.summary()["opt/update_ratio"] == pytest.approx(0.01)


def test_biases_are_kept_out_of_the_update_ratio() -> None:
    """A bias has a small norm and so a huge relative update; it would drown the ratio."""
    layer = nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
        layer.bias.copy_(torch.tensor([1e-6]))
    probe = _probe(layer, clip_norm=0.0)
    layer.weight.grad = torch.tensor([[0.03, 0.04]])
    layer.bias.grad = torch.tensor([1.0])

    probe.step()

    assert probe.summary()["opt/update_ratio"] == pytest.approx(0.01)
    # The step itself still moved the bias.
    assert float(layer.bias.detach()) == pytest.approx(1e-6 - 1.0)


def test_summary_resets_between_epochs() -> None:
    layer = _linear([[1.0, 0.0]])
    probe = _probe(layer, clip_norm=0.0)
    _set_grad(layer, [[10.0, 0.0]])
    probe.step()
    probe.summary()

    _set_grad(layer, [[2.0, 0.0]])
    probe.step()

    assert probe.summary()["grad/norm"] == pytest.approx(2.0)


def test_summary_is_empty_before_any_step() -> None:
    assert _probe(_linear([[1.0]]), clip_norm=0.0).summary() == {}


def test_every_n_steps_subsamples_the_measurements() -> None:
    layer = _linear([[1.0, 0.0]])
    probe = _probe(layer, clip_norm=0.0, every_n_steps=2)
    for norm in (10.0, 1000.0, 20.0, 2000.0):  # only the 1st and 3rd are measured
        _set_grad(layer, [[norm, 0.0]])
        probe.step()

    assert probe.summary()["grad/norm"] == pytest.approx(15.0)


def test_a_disabled_probe_records_nothing_but_still_clips_and_steps() -> None:
    """Training code calls the probe unconditionally, so disabling must not skip the step."""
    layer = _linear([[1.0, 1.0]])
    probe = _probe(layer, clip_norm=1.0, enabled=False)
    _set_grad(layer, [[6.0, 8.0]])

    probe.step()

    assert probe.summary() == {}
    assert layer.weight.grad is not None
    assert float(layer.weight.grad.norm()) == pytest.approx(1.0, rel=1e-5)
    # SGD with lr=1 applied the clipped gradient.
    assert float(layer.weight.detach()[0, 0]) == pytest.approx(1.0 - 0.6, rel=1e-5)


# --------------------------------------------------------------------------- #
# Activation probe
# --------------------------------------------------------------------------- #


def test_activation_probe_records_training_passes() -> None:
    model = nn.Sequential(_linear([[1.0], [1.0]]), nn.ReLU())
    model.train()

    with ActivationProbe(model, DiagnosticsConfig()) as probe:
        model(torch.tensor([[-1.0], [-1.0], [1.0], [1.0]]))
        summary = probe.summary()

    assert summary["act/saturated_frac"] == pytest.approx(0.5)
    assert summary["act/dead_frac"] == pytest.approx(0.0)  # both units fire for some row
    assert "act/output_std" in summary


def test_activation_probe_reports_the_worst_site_not_the_average() -> None:
    """Saturation concentrates in one layer; averaging over the stack hides it."""
    healthy = _linear([[1.0]])
    dying = _linear([[-1.0]])
    model = nn.Sequential(healthy, nn.ReLU(), dying, nn.ReLU())
    model.train()

    with ActivationProbe(model, DiagnosticsConfig()) as probe:
        model(torch.tensor([[1.0], [1.0]]))
        summary = probe.summary()

    # First ReLU: nothing saturated. Second: everything. The mean would read 0.5.
    assert summary["act/saturated_frac"] == pytest.approx(1.0)
    assert summary["act/dead_frac"] == pytest.approx(1.0)


def test_activation_probe_skips_the_dead_fraction_for_an_unbatched_output() -> None:
    """The factor network sees one vector per period; no unit can be dead across it."""
    model = nn.Sequential(_linear([[1.0], [1.0]]), nn.ReLU())
    model.train()

    with ActivationProbe(model, DiagnosticsConfig()) as probe:
        model(torch.tensor([-1.0]))
        summary = probe.summary()

    assert summary["act/saturated_frac"] == pytest.approx(1.0)
    assert "act/dead_frac" not in summary


def test_activation_probe_ignores_evaluation_passes() -> None:
    """Evaluation is the model being scored, not the network being diagnosed."""
    model = nn.Sequential(_linear([[1.0]]), nn.ReLU())
    model.eval()

    with ActivationProbe(model, DiagnosticsConfig()) as probe:
        model(torch.tensor([[-1.0], [1.0]]))
        assert probe.summary() == {}


def test_activation_probe_removes_its_hooks_on_exit() -> None:
    activation = nn.ReLU()
    model = nn.Sequential(_linear([[1.0]]), activation)
    model.train()

    with ActivationProbe(model, DiagnosticsConfig()):
        pass
    model(torch.tensor([[-1.0]]))

    assert not activation._forward_hooks


def test_a_disabled_activation_probe_registers_no_hooks() -> None:
    activation = nn.ReLU()
    model = nn.Sequential(_linear([[1.0]]), activation)
    model.train()

    with ActivationProbe(model, DiagnosticsConfig(enabled=False)) as probe:
        model(torch.tensor([[-1.0]]))
        assert not activation._forward_hooks
        assert probe.summary() == {}


def test_an_activation_free_model_reports_nothing() -> None:
    model = nn.Sequential(_linear([[1.0]]), nn.Identity())
    model.train()

    with ActivationProbe(model, DiagnosticsConfig()) as probe:
        model(torch.tensor([[1.0]]))
        assert probe.summary() == {}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("every_n_steps", 0),
        ("zero_weight_tol", -1.0),
        ("saturation_tol", 0.0),
        ("saturation_tol", 1.0),
    ],
)
def test_config_rejects_a_meaningless_setting(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        DiagnosticsConfig(**{field: value})  # type: ignore[arg-type]
