"""Column selection applied to the panel a feature run outputs.

A feature run builds a working panel that carries both the original input columns
and every produced feature — including scaffolding features that exist only to feed
a later featurizer (the daily return series behind a rolling volatility, the
cumulative on-balance-volume level behind its first difference). What the run
*returns* is a projection of that working panel, and :class:`ColumnSelection`
describes one side of it: which of the available columns to keep.

Selections are strict by default. Naming a column the panel does not carry is a
configuration error rather than a silent no-op, so a renamed or misspelled name
fails the run instead of quietly leaving an unwanted column in the feature matrix —
or dropping a wanted one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnSelection:
    """Which of a set of available columns to keep, as an include/exclude pair.

    Parameters
    ----------
    include:
        Column names to keep. ``None`` imposes no restriction and keeps every
        available column; an explicit list keeps exactly those columns, in the
        order given; ``()`` keeps none.
    exclude:
        Column names to drop from the included set. Exclusion wins over inclusion,
        so the default ``include=None`` paired with an ``exclude`` list keeps
        everything but the named columns.
    strict:
        When ``True`` (default) every name in ``include`` and ``exclude`` must be
        among the available columns, otherwise ``KeyError`` is raised — a stale or
        misspelled name fails the run instead of silently selecting nothing. Set it
        to ``False`` for a selection shared across panels that do not all carry the
        same columns.
    """

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = field(default_factory=tuple)
    strict: bool = True

    def select(self, available: Sequence[str], *, what: str = "columns") -> tuple[str, ...]:
        """Return the selected subset of ``available``, without duplicates.

        Columns come back in the order they are listed in ``include``, or in the
        order of ``available`` when ``include`` is ``None``. ``what`` names the
        selected set in the error message raised by a strict selection.
        """
        self.validate(available, what=what)
        present = set(available)
        candidates = dict.fromkeys(available if self.include is None else self.include)
        dropped = frozenset(self.exclude)
        return tuple(name for name in candidates if name in present and name not in dropped)

    def validate(self, available: Sequence[str], *, what: str = "columns") -> None:
        """Raise ``KeyError`` if a strict selection names a column that is not available.

        Exposed separately from :meth:`select` so a caller that already knows the
        column names — a pipeline that knows what its featurizers declare, say —
        can reject a stale selection before doing any work. A lenient selection
        (``strict=False``) validates nothing.
        """
        if not self.strict:
            return
        present = set(available)
        named = (*(self.include or ()), *self.exclude)
        missing = [name for name in dict.fromkeys(named) if name not in present]
        if missing:
            raise KeyError(
                f"selection refers to unknown {what}: {missing}; available: {sorted(present)}."
            )
