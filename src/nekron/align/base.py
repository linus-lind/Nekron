"""Shared abstractions for aligning several panels onto one.

Alignment is deliberately asymmetric. One named panel is the *spine*: it defines
the ``(date, entity)`` index of the result, and every other panel is reindexed
onto it. Nothing is ever outer-joined, so the merged panel has exactly the spine's
rows in exactly the spine's order, and adding a source can only add columns —
never rows, and never a reordering. An N-way symmetric join over financial panels
is where silent row inflation and accidental look-ahead come from; making the
spine explicit removes the whole class.

:class:`Spine` carries the target index together with its level codes, factorized
once. That is the single most important performance decision in this package:
``get_level_values`` materializes an entire level on every call and is not cached,
so aligning K panels the obvious way pays K full passes over the index before any
real work happens.

:class:`PanelAligner` is the switchable contract — one implementation per join
semantic — resolved by name through :mod:`~nekron.align.registry`, exactly like
the ingestion filters and preprocessing transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from nekron.frames import IntpArray, level_keys, require_unique_index
from nekron.panel import Panel, PanelGrain


class AlignmentError(Exception):
    """Base class for alignment errors."""


@dataclass(frozen=True)
class Spine:
    """The target index of a merge, with each key level factorized once.

    Attributes
    ----------
    index:
        The ``(date, entity)`` MultiIndex every aligned block must end up carrying.
    date_name, entity_name:
        Names of the two key levels.
    date_codes, entity_codes:
        Per-row position of that row's key value within the corresponding
        ``uniques``. ``-1`` marks a missing key value.
    date_uniques, entity_uniques:
        The lookup domain for each level. These may contain values no row uses —
        pandas keeps a level's full vocabulary after rows are filtered — so they
        are never a population count.
    """

    index: pd.MultiIndex
    date_name: str
    entity_name: str
    date_codes: IntpArray
    date_uniques: pd.Index
    entity_codes: IntpArray
    entity_uniques: pd.Index

    @classmethod
    def from_panel(cls, panel: Panel) -> Spine:
        """Build a spine from a ``(date, entity)`` panel, reading its level codes."""
        panel.require_grain(PanelGrain.PANEL, what="the alignment spine")
        index = panel.frame.index
        if not isinstance(index, pd.MultiIndex):
            raise AlignmentError(
                f"the alignment spine needs a two-level (date, entity) MultiIndex; got "
                f"{type(index).__name__}."
            )
        assert panel.date_name is not None and panel.entity_name is not None
        # The spine defines the output index, so a duplicate here is not a join
        # hazard but a data error that every downstream cross-section inherits.
        require_unique_index(index, f"the alignment spine (panel keyed {panel.key_names})")
        date_codes, date_uniques = level_keys(index, panel.date_name)
        entity_codes, entity_uniques = level_keys(index, panel.entity_name)
        return cls(
            index=index,
            date_name=panel.date_name,
            entity_name=panel.entity_name,
            date_codes=date_codes,
            date_uniques=date_uniques,
            entity_codes=entity_codes,
            entity_uniques=entity_uniques,
        )

    @property
    def n_rows(self) -> int:
        """Number of rows in the merged panel."""
        return len(self.index)


@runtime_checkable
class ColumnConsumer(Protocol):
    """An aligner that reads source columns it does not itself carry across."""

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Source columns the aligner needs in order to run."""
        ...


def required_columns(aligner: object) -> tuple[str, ...]:
    """Return the source columns ``aligner`` reads, or ``()`` if it declares none.

    An aligner's inputs are not the same set as its outputs. A point-in-time join
    reads the availability date that decides *which* row a spine row may see, but
    that date is bookkeeping — the caller asked for the fundamentals, not for the
    filing date. Declaring it here lets the merge project the source down to just
    what is needed before doing the expensive gather, and then drop the bookkeeping
    afterwards, instead of either carrying every source column through the gather
    or dropping the column the aligner depends on.
    """
    return aligner.required_columns if isinstance(aligner, ColumnConsumer) else ()


@runtime_checkable
class PanelAligner(Protocol):
    """One join semantic: how a source panel's columns reach the spine's rows."""

    @property
    def accepts(self) -> tuple[PanelGrain, ...]:
        """The source-panel grains this aligner can consume."""
        ...

    def align(self, spine: Spine, panel: Panel) -> pd.DataFrame:
        """Return ``panel``'s columns reindexed onto ``spine.index``, row for row.

        The result carries exactly ``spine.index``; keys with no match in the
        source become null rather than raising, so the caller decides whether a
        gap is acceptable.
        """
        ...
