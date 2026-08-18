"""The panel abstraction shared by every stage: a frame plus the keys it is indexed by.

A pipeline that reads one file can leave the index shape implicit — it is always
``(date, entity)``. A pipeline that reads several cannot: a daily price panel, a
quarterly fundamentals panel, a factor series indexed by date alone, a static
sector map indexed by entity alone and a keyless identifier link table all arrive
through the same reader and all have to be told apart. :class:`PanelGrain` names
the four shapes, and :class:`Panel` carries a frame together with its grain so a
stage can check what it was handed instead of guessing from ``nlevels``.

The distinction is not pedantry. Most preprocessing transforms are defined only
over a ``(date, entity)`` panel, and several of them do not fail on the wrong
grain — they return a plausible wrong answer. Forward-filling a static frame fills
nothing (each entity is one row), a cumulative factor over one row per entity is
all ones, and a NaN-fraction filter degenerates into "drop any row with a NaN".
Reference data is exactly that shape, so the grain check is what stands between a
mislabelled source and a silently corrupted panel.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from nekron.constants import DATE_LEVEL, ENTITY_LEVEL


class PanelError(Exception):
    """Base class for panel-shape errors."""


class UnknownPanelError(PanelError, KeyError):
    """Raised when a configuration names a panel that was never loaded.

    Deliberately both a :class:`PanelError` and a :class:`KeyError`. It has to be a
    ``KeyError`` because :class:`~collections.abc.Mapping` implements ``in`` and
    ``get`` by catching one, so a lookup error that is not a ``KeyError`` turns
    ``name in panels`` — the natural way to probe for an optional source — into a
    crash. It stays a ``PanelError`` so a caller can handle every panel problem in
    one place.
    """

    def __str__(self) -> str:
        # KeyError's __str__ reprs its argument, which would double-quote the
        # message and bury it in escapes.
        return str(self.args[0]) if self.args else ""


class PanelGrain(Enum):
    """The index shape of a loaded frame, i.e. which keys identify a row."""

    PANEL = "panel"
    """A ``(date, entity)`` cross-section over time — prices, returns, fundamentals."""

    TIME_SERIES = "time_series"
    """One row per date — market factors, macro series, risk-free rates."""

    CROSS_SECTION = "cross_section"
    """One row per entity — sector codes, security master, any static attribute."""

    TABLE = "table"
    """No key levels at all — identifier link tables and other lookup relations."""

    @property
    def has_date(self) -> bool:
        """Whether rows of this grain are identified by a date."""
        return self in (PanelGrain.PANEL, PanelGrain.TIME_SERIES)

    @property
    def has_entity(self) -> bool:
        """Whether rows of this grain are identified by an entity."""
        return self in (PanelGrain.PANEL, PanelGrain.CROSS_SECTION)

    @classmethod
    def of(cls, *, date: bool, entity: bool) -> PanelGrain:
        """Return the grain implied by which key columns a schema declares."""
        if date and entity:
            return cls.PANEL
        if date:
            return cls.TIME_SERIES
        if entity:
            return cls.CROSS_SECTION
        return cls.TABLE


@dataclass(frozen=True)
class Panel:
    """A frame together with the names of the index levels that key it.

    Parameters
    ----------
    frame:
        The data. Its index carries one level per key the grain declares, in
        ``(date, entity)`` order; a :attr:`PanelGrain.TABLE` frame keeps whatever
        positional index it was read with.
    date_name, entity_name:
        Names of the date and entity index levels, or ``None`` when this grain has
        no such key. They are stored rather than assumed because a schema may
        rename either level.
    """

    frame: pd.DataFrame
    date_name: str | None = DATE_LEVEL
    entity_name: str | None = ENTITY_LEVEL

    @property
    def grain(self) -> PanelGrain:
        """The index shape implied by the key levels this panel declares."""
        return PanelGrain.of(date=self.date_name is not None, entity=self.entity_name is not None)

    @property
    def key_names(self) -> tuple[str, ...]:
        """The index level names that key this panel, in index order."""
        return tuple(name for name in (self.date_name, self.entity_name) if name is not None)

    def with_frame(self, frame: pd.DataFrame) -> Panel:
        """Return a copy carrying ``frame``, keeping this panel's key names."""
        return Panel(frame=frame, date_name=self.date_name, entity_name=self.entity_name)

    def require_grain(self, *allowed: PanelGrain, what: str) -> None:
        """Raise :class:`PanelError` unless this panel's grain is one of ``allowed``."""
        if self.grain in allowed:
            return
        names = ", ".join(sorted(grain.value for grain in allowed))
        raise PanelError(
            f"{what} requires a panel of grain {names}; this panel is "
            f"{self.grain.value} (index levels {list(self.frame.index.names)})."
        )

    def validate(self) -> None:
        """Check that the frame's index actually matches the declared key names.

        The whole index is compared, not just its named levels: an extra unnamed
        level shifts every key one position across, and a stage that reads a level
        by position would then read the wrong one. Two keys sharing a name is
        rejected for the same reason — a lookup by name could not tell them apart.

        Key *uniqueness* is deliberately not checked here, and re-adding it would
        break ingestion. This runs on the panel a source has just been read into
        (:func:`nekron.data.ingest.load_panel`), which is upstream of the
        preprocessing stage whose ``deduplicate`` step exists precisely to merge
        rows sharing a key — a raw file with two rows for one ``(date, entity)``
        is the case that step is configured for, not a fault. Checking here makes
        that step unreachable.

        Uniqueness is enforced where a duplicate would actually corrupt a result,
        which is the merge: the alignment rejects a repeated key in its spine
        (:meth:`nekron.align.base.Spine.from_panel`) and in every join source
        (:mod:`nekron.align.aligners`). The spine is the merged panel's index and
        therefore the population every per-date cross-sectional statistic is taken
        over, so that is the check that protects the numbers.
        """
        expected = self.key_names
        if len(set(expected)) != len(expected):
            raise PanelError(
                f"a panel's key levels must have distinct names; got {list(expected)}."
            )
        index = self.frame.index
        if self.grain is PanelGrain.TABLE:
            if index.nlevels != 1 or index.name is not None:
                raise PanelError(
                    f"a {PanelGrain.TABLE.value} panel must carry a single unnamed index level; "
                    f"found {list(index.names)}."
                )
            return
        if tuple(index.names) != expected:
            raise PanelError(
                f"panel declares key levels {list(expected)} but its index is {list(index.names)}."
            )


class PanelSet(Mapping[str, Panel]):
    """The named panels a run has loaded, addressed by their configuration key.

    A plain mapping would do, except that every lookup here is driven by a name a
    human typed into a config file — a spine name, a join's source, a link table —
    so a miss should say which names *were* loaded rather than raise a bare
    ``KeyError``.
    """

    def __init__(self, panels: Mapping[str, Panel]) -> None:
        self._panels = dict(panels)

    def __getitem__(self, name: str) -> Panel:
        try:
            return self._panels[name]
        except KeyError:
            raise UnknownPanelError(
                f"unknown panel {name!r}; loaded panels are {sorted(self._panels)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __repr__(self) -> str:
        shapes = ", ".join(
            f"{name}: {panel.grain.value}{panel.frame.shape}"
            for name, panel in self._panels.items()
        )
        return f"PanelSet({shapes})"

    def of_grain(self, *grains: PanelGrain) -> dict[str, Panel]:
        """Return the subset of panels whose grain is one of ``grains``."""
        return {name: panel for name, panel in self._panels.items() if panel.grain in grains}
