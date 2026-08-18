"""The training objective — the one thing this model does not yet have.

The score head emits ``[B, K]`` real numbers per batch of windows and nothing is
yet asked of them. That is a deliberate hole, not an oversight: what the scores
should mean is the modelling decision the rest of this package exists to enable,
and picking a placeholder objective now would mean training against something
that has to be unlearned later.

Everything around the hole is finished. :class:`ScoreObjective` is the contract,
:class:`LossBreakdown` is what an objective returns, and the trainer calls both
unconditionally. Implementing an objective is therefore two edits and no
rewiring:

1. write a class satisfying :class:`ScoreObjective` here;
2. add its name to :data:`~.config.OBJECTIVES` and to :func:`build_objective`.

An objective is handed the model's output, the *whole* batch and the model's
parameters, in that order, because each is needed by objectives this model
plausibly wants and none can be recovered from the others:

``output``
    Both the scores and the pooled representation behind them, so a penalty on
    the representation itself (a contrastive or variance term) does not need a
    second forward pass.
``batch``
    Not just a target vector: a cross-sectional objective — rank the entities
    within each date, say — needs :attr:`~.data.windows.WindowBatch.date_indices`
    and :attr:`~.data.windows.WindowBatch.entity_indices` to know which scores
    belong together, and the shape of a batch of windows is that they do not
    arrive grouped.
``model_parameters``
    So a weight penalty is part of the objective rather than a second thing the
    trainer has to know about, as it is for the conditional autoencoder.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from .config import LossConfig
from .data.windows import WindowBatch
from .model import ScoreOutput


@dataclass
class LossBreakdown:
    """The scalar to backpropagate, plus whatever the objective wants recorded.

    :attr:`total` keeps its graph; :attr:`components` are for the metric row and
    are detached on the way out. They are kept as device tensors until an epoch
    ends, so recording a step costs no host-device synchronization.
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)

    def detached_components(self) -> dict[str, torch.Tensor]:
        """Detached, device-resident component tensors, prefixed for the metric row."""
        detached = {"loss/total": self.total.detach()}
        for name, value in self.components.items():
            detached[f"loss/{name}"] = value.detach()
        return detached

    def to_log_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.detached_components().items()}


class ScoreObjective(Protocol):
    """What the trainer requires of a training objective."""

    def __call__(
        self,
        output: ScoreOutput,
        batch: WindowBatch,
        model_parameters: Iterable[torch.Tensor],
    ) -> LossBreakdown:
        """Score one batch. ``LossBreakdown.total`` must carry a graph."""
        ...


def build_objective(cfg: LossConfig) -> ScoreObjective | None:
    """The objective ``cfg`` names, or ``None`` when none is configured.

    ``None`` is a value the trainer understands, not a failure: a run with
    ``loss.objective="none"`` builds the residuals, the windows, the model and the
    trainer and then declines to fit, which is what makes the whole pipeline
    runnable and inspectable before the objective exists.
    """
    if cfg.objective == "none":
        return None
    # Unreachable while OBJECTIVES has one entry; here so that adding a name to
    # the config without adding it here fails loudly rather than silently
    # training nothing.
    raise NotImplementedError(
        f"loss.objective={cfg.objective!r} is declared in config.OBJECTIVES but not implemented "
        f"in {__name__}.build_objective."
    )
