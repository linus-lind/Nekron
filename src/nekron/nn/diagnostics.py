"""Training-time diagnostics: the scalars that make an optimizer run tunable.

A loss curve says that a run went badly; it does not say which knob to turn. The
probes here record the small set of statistics that do, one value per epoch:

* the *pre-clip* gradient norm and how often clipping actually fires, which is
  what a clipping threshold is chosen against;
* the size of one optimizer step's parameter update relative to the parameters
  themselves, which is the scale-free reading of a learning rate;
* the norm and the near-zero fraction of the weight matrices, which is what a
  weight-decay or L1 coefficient actually moves;
* the fraction of each activation that is saturated, and the fraction of its
  units that are dead outright, which is what separates one activation function
  from another.

Measurement points matter more than the formulas. The gradient norm is only
informative before clipping (after it, every clipped step reports the threshold),
and an update is only observable across ``optimizer.step()`` itself — which is
why :class:`OptimizerProbe` owns both operations instead of being called around
them.

Every statistic is accumulated as a device-resident tensor and converted to a
Python float once per epoch in ``summary()``, so instrumenting a step costs no
host-device synchronization. When :attr:`DiagnosticsConfig.enabled` is False the
probes still clip and step but record nothing and return an empty mapping, so
training code can call them unconditionally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import TracebackType

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.hooks import RemovableHandle

from .proximal import ProximalL1

# Denominators are clamped by this before dividing, so a model whose weights are
# all zero reports 0.0 rather than a NaN that MLflow would silently drop.
_EPS = 1e-12

# Which quantile of the observed gradient norms is reported. The established recipe
# for choosing a clipping threshold is to train unclipped and clip at the 90th
# percentile of the norms that run produced, so this is reported rather than
# configured: it is the number that goes into ``grad_clip_norm``.
_CLIP_PERCENTILE = 0.9


@dataclass
class DiagnosticsConfig:
    """What training diagnostics a run records, and how finely.

    Parameters
    ----------
    enabled:
        Whether to record anything at all.
    every_n_steps:
        Record on every ``n``-th measurement opportunity — an optimizer step for
        :class:`OptimizerProbe`, a training forward pass for
        :class:`ActivationProbe`. Measuring costs one copy of the parameters and a
        handful of reductions per step, which is negligible for a small model;
        raise this for a large one.
    zero_weight_tol:
        A weight entry with ``|w|`` at or below this counts as driven to zero,
        which is what an L1 penalty is there to do.
    saturation_tol:
        How close to an activation's flat region an output must be to count as
        saturated. See :func:`saturated_fraction`.
    """

    enabled: bool = True
    every_n_steps: int = 1
    zero_weight_tol: float = 1e-4
    saturation_tol: float = 1e-2

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError(f"every_n_steps must be at least 1; got {self.every_n_steps}.")
        if self.zero_weight_tol < 0.0:
            raise ValueError(f"zero_weight_tol must not be negative; got {self.zero_weight_tol}.")
        if not 0.0 < self.saturation_tol < 1.0:
            raise ValueError(f"saturation_tol must lie in (0, 1); got {self.saturation_tol}.")


def _add(total: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Accumulate on device, so nothing is read back until the epoch ends."""
    return value if total is None else total + value


# --------------------------------------------------------------------------- #
# Activation saturation
# --------------------------------------------------------------------------- #

# A saturation test flags the outputs of one activation that carry (almost) no
# gradient, given the module and its tolerance.
SaturationTest = Callable[[nn.Module, torch.Tensor, float], torch.Tensor]


def _flat_below_zero(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """ReLU and leaky ReLU: the whole negative branch is flat or near-flat."""
    del module, tol
    return out <= 0.0


def _flat_at_lower_asymptote(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """ELU: flat as the output approaches its lower asymptote ``-alpha``."""
    alpha = float(getattr(module, "alpha", 1.0))
    return out <= -alpha + tol


def _flat_near_zero(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """GELU and SiLU: flat in the left tail, where the output decays to zero."""
    del module
    return out.abs() <= tol


def _flat_at_both_ends(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """Tanh: flat as the output approaches either of ``-1`` and ``1``."""
    del module
    return out.abs() >= 1.0 - tol


def _flat_at_unit_interval_ends(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """Sigmoid: flat as the output approaches either of ``0`` and ``1``."""
    del module
    return (out <= tol) | (out >= 1.0 - tol)


# Keyed by exact type, mirroring ``nekron.nn.utils._ACTIVATIONS``: an activation
# absent here (``nn.Identity``) has no flat region and is simply not instrumented.
_SATURATION: dict[type[nn.Module], SaturationTest] = {
    nn.ReLU: _flat_below_zero,
    nn.LeakyReLU: _flat_below_zero,
    nn.ELU: _flat_at_lower_asymptote,
    nn.GELU: _flat_near_zero,
    nn.SiLU: _flat_near_zero,
    nn.Tanh: _flat_at_both_ends,
    nn.Sigmoid: _flat_at_unit_interval_ends,
}


def flat_mask(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """Mask of the outputs of one activation that carry (almost) no gradient.

    "Flat" is defined in output space, per activation type, as the region where
    the activation's own derivative falls below ``tol`` — the negative branch of a
    ReLU, the tails of a tanh, the lower asymptote of an ELU. Testing the output
    rather than differentiating the activation costs one comparison instead of a
    backward pass, and the two agree over more than 95% of the input range for
    every activation this project builds.

    Defining the region by the derivative, rather than per activation by eye, is
    what makes the fractions below comparable across activation functions — which
    is the whole point when the activation is the hyperparameter under test.
    """
    test = _SATURATION.get(type(module))
    if test is None:
        raise KeyError(f"no saturation test registered for {type(module).__name__}.")
    return test(module, out, tol)


def saturated_fraction(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """Share of one activation's outputs that pass (almost) no gradient.

    Read as a *trend*, not as a level. Half of a healthy ReLU layer's outputs sit
    at zero by construction, so the number to react to is a fraction climbing over
    a run — a network progressively switching itself off — rather than any
    particular value. For the saturating activations (tanh, sigmoid) the level does
    mean something directly: outputs pinned in the tails are a layer that has
    stopped learning.
    """
    return flat_mask(module, out, tol).to(out.dtype).mean()


def dead_fraction(module: nn.Module, out: torch.Tensor, tol: float) -> torch.Tensor:
    """Share of units that are flat for *every* row of the batch.

    The sharper half of :func:`saturated_fraction`: a unit that passes no gradient
    for any example in the batch receives no gradient at all, and for a ReLU it can
    never recover. This is the reading with an actual threshold behind it — a large
    share of a ReLU network can end up permanently dead when the learning rate is
    set too high — where the per-output fraction only has a trend.

    Requires a batched output; an activation applied to a single vector has no
    population to be dead across.
    """
    if out.ndim < 2:
        raise ValueError(
            f"dead_fraction needs a batched output; got a {out.ndim}-dimensional tensor."
        )
    return flat_mask(module, out, tol).all(dim=0).to(out.dtype).mean()


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


@torch.no_grad()
def weight_summary(model: nn.Module, cfg: DiagnosticsConfig) -> dict[str, float]:
    """Size and sparsity of a model's weight matrices.

    Taken over the parameters with ``ndim >= 2`` — the same set a LASSO penalty is
    applied to, so biases and normalization scales do not dilute either statistic.

    ``weights/norm`` is the global L2 norm: under weight decay it settles at the
    level where decay balances the gradient, so a norm still climbing at the end of
    a run means the penalty is too weak. ``weights/zero_frac`` is the share of
    entries an L1 penalty has driven to (numerical) zero, which is the direct
    reading of that penalty's strength — near zero means it is inert, near one
    means it has erased the model.
    """
    if not cfg.enabled:
        return {}
    weights = [p for p in model.parameters() if p.ndim >= 2]
    if not weights:
        return {}
    squares = torch.stack([p.pow(2).sum() for p in weights]).sum()
    zeros = torch.stack([(p.abs() <= cfg.zero_weight_tol).sum() for p in weights]).sum()
    count = sum(p.numel() for p in weights)
    return {
        "weights/norm": float(squares.sqrt()),
        "weights/zero_frac": float(zeros) / count,
    }


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


class OptimizerProbe:
    """Clips, steps, applies any proximal penalty, and records what the step did.

    Replaces the ``clip_grad_norm_`` / ``optimizer.step()`` pair in a training
    loop. Each measurement has exactly one valid observation point — the gradient
    norm before clipping rescales it, the update across the whole step — so the
    probe performs the operations rather than being called around them.

    ``proximal`` extends that ownership to a non-smooth penalty. A proximal
    operator has to run after the gradient step and before the next measurement,
    and its shrinkage is part of the update the model actually took, so
    ``opt/update_ratio`` has to span it. Handing it to the probe is the only
    arrangement in which both are true.

    Call :meth:`step` once per optimizer step and :meth:`summary` once per epoch;
    the latter returns the epoch's averages and resets the accumulators.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: DiagnosticsConfig,
        *,
        clip_norm: float,
        proximal: ProximalL1 | None = None,
    ) -> None:
        self.cfg = cfg
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.proximal = proximal
        self._parameters = [p for p in model.parameters() if p.requires_grad]
        # The update ratio is read off the weight matrices only, matching the set a
        # LASSO penalty and :func:`weight_summary` use. Biases carry a small norm
        # and therefore a large relative update, which would drown the reading.
        self._weights = [p for p in self._parameters if p.ndim >= 2]
        # Reused across steps: the update norm needs the weights as they were
        # before the step, and allocating that copy every step would churn the
        # allocator for the whole run.
        self._previous = [torch.empty_like(p) for p in self._weights] if cfg.enabled else []
        self._steps = 0
        self._grad_norms: list[torch.Tensor] = []
        self._update_ratios: list[torch.Tensor] = []

    def step(self) -> None:
        """Clip the gradients if configured, take the optimizer step, and record it."""
        measuring = self.cfg.enabled and self._steps % self.cfg.every_n_steps == 0
        # ``clip_grad_norm_`` returns the norm *before* clipping and, given an
        # infinite threshold, computes it without touching the gradients — so one
        # call covers both the clipped and the unclipped configuration.
        threshold = self.clip_norm if self.clip_norm > 0.0 else float("inf")
        grad_norm = clip_grad_norm_(self._parameters, threshold)
        if measuring:
            # Kept as a device tensor and reduced once per epoch: reading each norm
            # back here would synchronize the device on every step, which is the one
            # cost this instrumentation is meant not to have.
            self._grad_norms.append(grad_norm.detach())
            self._snapshot()
        self.optimizer.step()
        if self.proximal is not None:
            # After the step and inside the measured window: the shrinkage is part
            # of this update. ``lr`` comes from the optimizer rather than a stored
            # copy so the threshold follows the schedule.
            self.proximal(self._weights, self.optimizer.param_groups[0]["lr"])
        if measuring:
            self._update_ratios.append(self._update_ratio())
        self._steps += 1

    @torch.no_grad()
    def _snapshot(self) -> None:
        for buffer, weight in zip(self._previous, self._weights, strict=True):
            buffer.copy_(weight)

    @torch.no_grad()
    def _update_ratio(self) -> torch.Tensor:
        """Mean over weight matrices of ``||W_after - W_before|| / ||W_before||``."""
        if not self._weights:
            return torch.zeros(())
        ratios = [
            (w - before).norm() / before.norm().clamp_min(_EPS)
            for w, before in zip(self._weights, self._previous, strict=True)
        ]
        return torch.stack(ratios).mean()

    def summary(self) -> dict[str, float]:
        """The epoch's optimizer statistics; resets the accumulators.

        ``grad/norm``, ``grad/norm_p90`` and ``grad/norm_max`` are the mean, the
        90th percentile and the largest *pre-clip* gradient norm over the epoch's
        steps. The percentile is the one to act on: the standard recipe for a
        clipping threshold is to train unclipped, read the 90th percentile of the
        norms that run produced and clip there, so this metric is the value to put
        in ``grad_clip_norm``. The maximum says whether clipping is worth having at
        all — a maximum within a small multiple of the percentile means there is no
        tail to clip.

        ``grad/clipped_frac``, logged only when clipping is on, is the share of
        steps the threshold actually bit on. Near zero it is inert; past roughly a
        half it has stopped being clipping and become an obscure way of lowering the
        learning rate, which is better lowered directly.

        ``grad/nonfinite_frac`` appears only when some step produced a NaN or
        infinite gradient. It has to be reported separately because a non-finite
        value would otherwise propagate silently: it is excluded from the statistics
        above, and were it not, they would all read NaN and be dropped by the
        tracker, leaving a chart that simply stops.

        ``opt/update_ratio`` is ``||W_after - W_before|| / ||W_before||``, averaged
        over the weight matrices and taken as the median over the epoch's steps,
        spanning the proximal shrinkage as well as the gradient step where one is
        configured. It
        is the scale-free reading of the learning rate, and the standard rule of
        thumb puts a healthy run at roughly 1e-3: an order of magnitude below and
        the run is barely moving, an order above and it is thrashing. The median
        rather than the mean because the ratio spans decades over a long decayed
        schedule, where an average is decided by its largest steps.
        """
        self._steps = 0
        norms, self._grad_norms = self._grad_norms, []
        ratios, self._update_ratios = self._update_ratios, []
        if not norms:
            return {}
        # One transfer for the whole epoch. The reductions run on the host because
        # ``torch.quantile`` is not implemented on every accelerator backend, and
        # the widening to float64 happens only after the transfer — MPS refuses to
        # produce a float64 tensor at all.
        stacked = torch.stack(norms).cpu().to(torch.float64)
        finite = stacked[stacked.isfinite()]
        summary: dict[str, float] = {}
        if finite.numel() > 0:
            summary["grad/norm"] = float(finite.mean())
            summary["grad/norm_p90"] = float(finite.quantile(_CLIP_PERCENTILE))
            summary["grad/norm_max"] = float(finite.max())
        if finite.numel() < stacked.numel():
            summary["grad/nonfinite_frac"] = 1.0 - finite.numel() / stacked.numel()
        if self.clip_norm > 0.0:
            summary["grad/clipped_frac"] = float((stacked > self.clip_norm).double().mean())
        if ratios:
            summary["opt/update_ratio"] = float(torch.stack(ratios).median())
        return summary


@dataclass
class _SiteTotals:
    """Running totals for one activation site, kept on device until the epoch ends."""

    saturated: torch.Tensor | None = None
    dead: torch.Tensor | None = None
    std: torch.Tensor | None = None
    seen: int = 0
    recorded: int = 0


class ActivationProbe:
    """Records how much of a model's activations is passing gradient.

    A context manager: entering registers a forward hook on every activation
    module :func:`flat_mask` knows, leaving removes them. Hooks fire on every
    forward pass, so each one records only while its module is in training mode —
    evaluation is the model being scored, not the network being diagnosed, and
    mixing the two would blur each statistic across two different populations.

    Statistics are kept per activation site and reported for the *worst* site
    rather than averaged across them. Saturation concentrates in one layer, and an
    average over a stack is exactly the aggregation that hides it — while still
    costing one metric per epoch, whatever the depth.
    """

    def __init__(self, model: nn.Module, cfg: DiagnosticsConfig) -> None:
        self.cfg = cfg
        self._activations = [m for m in model.modules() if type(m) in _SATURATION]
        self._totals = [_SiteTotals() for _ in self._activations]
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> ActivationProbe:
        if self.cfg.enabled:
            self._handles = [
                module.register_forward_hook(partial(self._hook, index))
                for index, module in enumerate(self._activations)
            ]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @torch.no_grad()
    def _hook(
        self,
        index: int,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        del inputs
        if not module.training:
            return
        totals = self._totals[index]
        record = totals.seen % self.cfg.every_n_steps == 0
        totals.seen += 1
        if not record:
            return
        out = out.detach()
        # The mask is taken once and reduced twice; :func:`saturated_fraction` and
        # :func:`dead_fraction` are the same two reductions for a caller holding a
        # single tensor.
        mask = flat_mask(module, out, self.cfg.saturation_tol)
        totals.saturated = _add(totals.saturated, mask.to(out.dtype).mean())
        if out.ndim >= 2:
            # An activation applied to a single vector — the factor network's, which
            # sees one managed-portfolio vector per period — has no population for a
            # unit to be dead across, so only the per-output share is defined.
            totals.dead = _add(totals.dead, mask.all(dim=0).to(out.dtype).mean())
        if out.numel() > 1:
            totals.std = _add(totals.std, out.std())
        totals.recorded += 1

    def summary(self) -> dict[str, float]:
        """The epoch's activation statistics; resets the accumulators.

        ``act/saturated_frac`` is the share of activation outputs passing no
        gradient, at the worst site. Read it as a trend: half of a healthy ReLU
        layer sits at zero by construction, so what matters is a fraction climbing
        over a run. ``act/dead_frac`` is the sharper reading — the share of *units*
        flat across every stock in the cross-section, which for a ReLU is a unit
        that cannot come back. A large and growing share means the learning rate is
        too high for this activation, or that a leakier one (ELU, GELU, leaky ReLU)
        is the answer.

        ``act/output_std`` is the spread of the outputs at the *narrowest* site.
        It catches the failure the two fractions cannot see: a tanh stack collapsed
        into the linear region around zero reports no saturation at all while the
        network has quietly stopped being nonlinear.
        """
        sites = [t for t in self._totals if t.recorded > 0]
        self._totals = [_SiteTotals() for _ in self._activations]
        if not sites:
            return {}
        summary: dict[str, float] = {}
        saturated = [float(t.saturated) / t.recorded for t in sites if t.saturated is not None]
        if saturated:
            summary["act/saturated_frac"] = max(saturated)
        dead = [float(t.dead) / t.recorded for t in sites if t.dead is not None]
        if dead:
            summary["act/dead_frac"] = max(dead)
        spread = [float(t.std) / t.recorded for t in sites if t.std is not None]
        if spread:
            summary["act/output_std"] = min(spread)
        return summary
