"""Sequential composition of panel preprocessing transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from nekron.panel import PanelGrain

from .base import PanelTransform, PreprocessingError, transform_grains

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    """Apply an ordered sequence of :class:`PanelTransform` steps to a panel."""

    steps: tuple[PanelTransform, ...]

    def apply(self, panel: pd.DataFrame, *, grain: PanelGrain | None = None) -> pd.DataFrame:
        """Run every step in order, optionally checking each is defined for ``grain``.

        Passing the panel's grain turns a whole class of silent corruption into a
        configuration error: several transforms here read an entity level that a
        factor series or a static reference frame does not have, and rather than
        failing they return a plausible wrong answer. The check is skipped when
        ``grain`` is ``None`` so a caller that already knows what it holds — or is
        working with a bare frame — pays nothing.
        """
        if grain is not None:
            self.require_grain(grain)
        for step in self.steps:
            before = len(panel)
            panel = step.apply(panel)
            logger.debug("%s: %d -> %d rows", type(step).__name__, before, len(panel))
        return panel

    def require_grain(self, grain: PanelGrain) -> None:
        """Raise :class:`PreprocessingError` if any step is undefined for ``grain``."""
        for position, step in enumerate(self.steps):
            supported = transform_grains(step)
            if grain not in supported:
                names = ", ".join(sorted(item.value for item in supported))
                raise PreprocessingError(
                    f"preprocessing step {position} ({type(step).__name__}) is defined for "
                    f"grain {names}, but the panel is {grain.value}. Move the step to the "
                    "merged-panel pipeline, or drop it for this panel."
                )
