"""The four join semantics that carry a source panel's columns onto the spine.

Every merge in this package is one of four questions, and each has exactly one
correct answer shape:

* :class:`ExactAligner` — the source is keyed the same way the spine is, so a row
  matches at most one source row: a direct ``(date, entity)`` reindex.
* :class:`AsOfEntityAligner` — the source is keyed the same way but its dates are
  *observation* dates, not availability dates. A spine row may only see its
  entity's latest source row that was already knowable on that date. This is the
  only place look-ahead can enter a feature set, so the knowability rule is
  explicit: an availability column, a publication lag, a staleness tolerance and
  whether same-day data counts.
* :class:`BroadcastTimeAligner` — the source is keyed by date alone (factors,
  macro), so one source row reaches every entity trading on that date.
* :class:`BroadcastEntityAligner` — the source is keyed by entity alone (sector,
  security master), so one source row reaches every date of that entity.

All four are frozen dataclasses satisfying :class:`~nekron.align.base.PanelAligner`
and are resolved by the names in :data:`ALIGNERS`, so which semantic a source gets
is a configuration decision rather than a code path.

Three implementation rules are load-bearing and are the reason this module does
not simply call ``merge``:

* The spine's key levels are factorized once, by :class:`~nekron.align.base.Spine`.
  Both broadcasts therefore align the source to the *unique* key values — a few
  thousand dates, a few thousand entities — and only then gather the result out to
  full length. Composing the two indexers into a single gather is slower (it
  allocates a full-length indexer first) and, worse, ``pos[codes]`` reads a ``-1``
  code as "the last element" instead of "no match".
* Nothing is ever gathered with a bare ``take``. A ``MultiIndex`` stores ``-1``
  for a missing level value and every indexer here uses ``-1`` for an unmatched
  key, and ``take(-1)`` silently returns the last row.
* A key dtype mismatch — naive against timezone-aware dates, ``int64`` identifiers
  against strings, ``Period`` against ``Timestamp`` — does not raise anywhere in
  pandas. It matches nothing and fills the column with nulls, which looks exactly
  like sparse reference data. :func:`_require_comparable` compares the *kind* of
  the two key dtypes up front and refuses, rather than inferring a mismatch from
  an empty result: a source that legitimately covers a disjoint period must stay
  legal, and in the as-of joins zero matches is an ordinary outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from nekron.align.base import AlignmentError, Spine
from nekron.align.keys import Int64Array, date_ranks, joint_codes, latest_at_or_before
from nekron.frames import (
    FrameError,
    IntpArray,
    level_keys,
    require_unique_index,
    resolve_level,
    take_filled,
)
from nekron.panel import Panel, PanelError, PanelGrain

__all__ = [
    "ALIGNERS",
    "AsOfEntityAligner",
    "BroadcastEntityAligner",
    "BroadcastTimeAligner",
    "ExactAligner",
]


# ---------------------------------------------------------------------------
# shared checks
# ---------------------------------------------------------------------------


def _require_grain(panel: Panel, aligner: str, accepts: tuple[PanelGrain, ...]) -> None:
    """Reject a source panel whose grain this aligner cannot consume."""
    try:
        panel.require_grain(*accepts, what=f"the {aligner!r} aligner")
    except PanelError as exc:
        raise AlignmentError(str(exc)) from exc


def _require_unique(index: pd.Index, what: str) -> None:
    """Reject a source whose keys repeat, before any lookup is attempted."""
    try:
        require_unique_index(index, what)
    except FrameError as exc:
        raise AlignmentError(
            f"{exc} Aggregate or deduplicate the source panel before aligning it; a "
            "duplicated key has no single right answer, and a merge would silently "
            "fan the spine out to more rows than it has."
        ) from exc


def _key_kind(index: pd.Index) -> str | None:
    """Classify a key index by what it can possibly compare equal to.

    Returns ``None`` when the dtype carries too little information to judge, so an
    unknown key is never rejected on a guess. Categoricals are judged by their
    categories: a categorical of strings matches plain strings, which is the whole
    point of keeping entity identifiers categorical.
    """
    dtype = index.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _key_kind(dtype.categories)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "tz-aware datetime"
    if isinstance(dtype, pd.PeriodDtype):
        return f"period[{dtype.freq.freqstr}]"
    if isinstance(dtype, pd.IntervalDtype):
        return "interval"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "timedelta"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "number"
    if isinstance(dtype, pd.StringDtype) or dtype.kind in "US":
        return "string"
    if dtype.kind == "O":
        # An object index is whatever it holds; ask the values, which is cheap
        # because only the *unique* key values ever reach this function.
        inferred = pd.api.types.infer_dtype(index, skipna=True)
        return {
            "string": "string",
            "bytes": "string",
            "integer": "number",
            "floating": "number",
            "mixed-integer-float": "number",
            "decimal": "number",
            "boolean": "boolean",
            "datetime64": "datetime",
            "datetime": "datetime",
            "date": "datetime",
            "period": "period",
        }.get(inferred)
    return None


def _require_comparable(target: pd.Index, source: pd.Index, *, aligner: str, key: str) -> None:
    """Refuse a lookup whose two sides cannot compare equal by construction.

    Pandas does not raise when join keys are of unrelated types; it matches nothing
    and returns an all-null column, which is indistinguishable from a source that
    simply has no data for these rows. Since the comparison is decided by the
    dtypes alone, it can be decided *before* the lookup — which is what keeps a
    legitimately disjoint source (a factor file covering another decade) legal.
    """
    target_kind = _key_kind(target)
    source_kind = _key_kind(source)
    if target_kind is None or source_kind is None or target_kind == source_kind:
        return
    raise AlignmentError(
        f"the {aligner!r} aligner cannot match on {key}: the spine's key is "
        f"{target.dtype} ({target_kind}) but the source's is {source.dtype} "
        f"({source_kind}). Pandas would not raise here — it would match nothing and "
        f"fill every row with nulls. Cast both sides to one type first (remove or "
        f"add the timezone, convert a period to its timestamp, or read the "
        f"identifier as a single string or integer type)."
    )


def _key_index(index: pd.Index, level: str, *, aligner: str, key: str) -> pd.Index:
    """Return a single-key source index as a flat index of its key values."""
    if isinstance(index, pd.MultiIndex):
        if index.nlevels != 1:
            raise AlignmentError(
                f"the {aligner!r} aligner expects a source keyed by {key} alone; its index "
                f"has levels {list(index.names)}."
            )
        return index.get_level_values(resolve_level(index, level))
    return index


def _require_multi_index(index: pd.Index, aligner: str) -> pd.MultiIndex:
    """Return the source's ``(date, entity)`` MultiIndex, or refuse.

    Exactly two levels: a third one would make the index unique while the
    ``(date, entity)`` pairs that are actually joined on repeat, and a repeated
    join key is the fan-out :func:`_require_unique` exists to prevent.
    """
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        levels = list(index.names) if isinstance(index, pd.MultiIndex) else type(index).__name__
        raise AlignmentError(
            f"the {aligner!r} aligner expects a source with a two-level (date, entity) "
            f"MultiIndex; got {levels}."
        )
    return index


def _timedelta(value: str, what: str) -> pd.Timedelta:
    """Parse an offset alias into a :class:`~pandas.Timedelta`, or explain why not."""
    try:
        delta = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(
            f"{what} must be a pandas offset alias such as '45D'; got {value!r}."
        ) from exc
    if pd.isna(delta):
        raise ValueError(f"{what} must be a pandas offset alias such as '45D'; got {value!r}.")
    return delta


# ---------------------------------------------------------------------------
# shared gathers
# ---------------------------------------------------------------------------


def _broadcast(
    frame: pd.DataFrame, positions: IntpArray, codes: IntpArray, index: pd.Index
) -> pd.DataFrame:
    """Align a source to the unique key values, then gather those out to full length.

    Two gathers, not one: the first is over the handful of distinct keys, and only
    the second is full length. Fusing them into ``positions[codes]`` would allocate
    a full-length indexer for no gain and would read a missing key's ``-1`` code as
    the last unique value.
    """
    aligned = take_filled(frame, positions)
    result = take_filled(aligned, codes)
    result.index = index
    return result


def _gather(frame: pd.DataFrame, positions: IntpArray, index: pd.Index) -> pd.DataFrame:
    """Gather source rows onto the spine, one row per spine row."""
    result = take_filled(frame, positions)
    result.index = index
    return result


def _microseconds(values: pd.Index, *, aligner: str, what: str) -> Int64Array:
    """Return a date index as microsecond integers, refusing non-dates outright.

    Microseconds rather than nanoseconds because the far-future sentinels that
    reference files use (``9999-12-31``) wrap silently into the nineteenth century
    when cast to nanoseconds.
    """
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        raise AlignmentError(
            f"the {aligner!r} aligner cannot rank timezone-aware dates ({what} is "
            f"{values.dtype}); convert both sides to naive dates first."
        )
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise AlignmentError(
            f"the {aligner!r} aligner needs datetime dates; {what} is {values.dtype}."
        )
    return np.asarray(values.to_numpy().astype("datetime64[us]").astype(np.int64), dtype=np.int64)


# ---------------------------------------------------------------------------
# aligners
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactAligner:
    """Carry a ``(date, entity)`` source onto identically keyed spine rows.

    The join is a reindex: each spine row takes the source row with its exact key,
    or nulls. ``MultiIndex.get_indexer`` already does exactly that, and does it on
    the indexes' stored level codes rather than by materializing their values, so
    there is nothing for a hand-rolled packed-key lookup to improve on — measured
    against one, ``get_indexer`` was ~3x faster and used ~5x less peak memory.

    The one thing it does not do is check its inputs, hence the guards below: a
    duplicated source key would fan the spine out, and a key dtype that cannot
    compare equal would return a column of nulls with no complaint at all.
    """

    accepts: ClassVar[tuple[PanelGrain, ...]] = (PanelGrain.PANEL,)

    def align(self, spine: Spine, panel: Panel) -> pd.DataFrame:
        """Return ``panel``'s columns on ``spine.index``, matched key for key."""
        _require_grain(panel, "exact", self.accepts)
        assert panel.date_name is not None and panel.entity_name is not None
        frame = panel.frame
        index = _require_multi_index(frame.index, "exact")
        _require_unique(index, "the source panel of an 'exact' alignment")

        _, source_dates = level_keys(index, panel.date_name)
        _, source_entities = level_keys(index, panel.entity_name)
        _require_comparable(spine.date_uniques, source_dates, aligner="exact", key="date")
        _require_comparable(spine.entity_uniques, source_entities, aligner="exact", key="entity")

        if len(frame) == 0 or spine.n_rows == 0:
            return _gather(frame, np.full(spine.n_rows, -1, dtype=np.intp), spine.index)

        ordered = (
            index
            if _levels_in_order(index, panel)
            else index.reorder_levels([panel.date_name, panel.entity_name])
        )
        positions = np.asarray(ordered.get_indexer(spine.index), dtype=np.intp)
        return _gather(frame, positions, spine.index)


def _levels_in_order(index: pd.MultiIndex, panel: Panel) -> bool:
    """Whether the source's levels already sit in ``(date, entity)`` order.

    ``get_indexer`` matches level by position, not by name, so a source indexed
    ``(entity, date)`` would silently match nothing against a ``(date, entity)``
    spine.
    """
    return tuple(index.names) == (panel.date_name, panel.entity_name)


@dataclass(frozen=True)
class AsOfEntityAligner:
    """Carry each entity's latest *knowable* source row onto every spine date.

    A fundamentals panel is stamped with the period it describes, not the day it
    became public; joining it on that stamp puts a quarter's earnings into the
    features of the weeks before they were announced. This aligner asks the only
    honest question instead: on this date, what was the latest row of this entity
    that a person could already have read?

    Parameters
    ----------
    availability_column:
        A date column of the source frame holding when each row became public (a
        report date such as Compustat's ``RDQ``). When ``None`` the source's own
        date level is used, which is only correct when that level already *is* an
        availability date.
    lag:
        Offset alias added to the availability date before matching — the "assume
        it was public N days later" knob used when no report date exists, e.g.
        ``"45D"`` for quarterly filings.
    tolerance:
        Maximum age of a match, e.g. ``"400D"``. A match older than this becomes
        null rather than carrying a stale value forward forever, which is what
        distinguishes a delisted company's last filing from current data.
    allow_exact:
        Whether a row stamped with the spine's own date counts as knowable that
        day. ``False`` requires strictly earlier availability.

    Notes
    -----
    Rows with a missing availability date can never be matched: an unknown
    publication date is not evidence of early publication. When one entity has
    several rows sharing an availability date the last of them in source order
    wins, deterministically.
    """

    availability_column: str | None = None
    lag: str | None = None
    tolerance: str | None = None
    allow_exact: bool = True

    accepts: ClassVar[tuple[PanelGrain, ...]] = (PanelGrain.PANEL,)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The availability column, which is read but not carried to the spine."""
        return () if self.availability_column is None else (self.availability_column,)

    def __post_init__(self) -> None:
        if self.lag is not None:
            _timedelta(self.lag, "lag")
        if self.tolerance is not None:
            _timedelta(self.tolerance, "tolerance")

    def align(self, spine: Spine, panel: Panel) -> pd.DataFrame:
        """Return ``panel``'s columns on ``spine.index``, as of each row's date."""
        _require_grain(panel, "asof_entity", self.accepts)
        assert panel.date_name is not None and panel.entity_name is not None
        frame = panel.frame
        index = _require_multi_index(frame.index, "asof_entity")
        _require_unique(index, "the source panel of an 'asof_entity' alignment")

        entity_level = resolve_level(index, panel.entity_name)
        _require_comparable(
            spine.entity_uniques, index.levels[entity_level], aligner="asof_entity", key="entity"
        )
        availability = self._availability(panel, index)
        spine_dates = spine.date_uniques
        spine_us = _microseconds(spine_dates, aligner="asof_entity", what="the spine's date level")
        available_us = _microseconds(
            availability, aligner="asof_entity", what="the source's availability date"
        )

        empty = np.full(spine.n_rows, -1, dtype=np.intp)
        if len(frame) == 0 or spine.n_rows == 0:
            return _gather(frame, empty, spine.index)

        # Both key sides are coded against one vocabulary, and only the spine's
        # *unique* entities are coded — the per-row codes are already known.
        unique_entity_codes, source_entity_codes, _ = joint_codes(
            [np.asarray(spine.entity_uniques.to_numpy())],
            [np.asarray(index.get_level_values(entity_level).to_numpy())],
        )
        query_codes = np.where(
            (spine.entity_codes >= 0) & (spine.date_codes >= 0),
            unique_entity_codes.take(np.maximum(spine.entity_codes, 0))
            if unique_entity_codes.size
            else -1,
            -1,
        ).astype(np.int64)
        # A row whose availability date is unknown is not knowable on any date.
        source_codes = np.where(np.asarray(pd.isna(availability)), -1, source_entity_codes).astype(
            np.int64
        )

        (spine_ranks, source_ranks), span = date_ranks(
            spine_dates.to_numpy(), np.asarray(availability.to_numpy())
        )
        # ``span == 0`` covers an empty vocabulary on both sides; ``spine_ranks``
        # is separately empty when every spine date is missing, which no spine row
        # could be knowable as of.
        if span == 0 or spine_ranks.size == 0:
            return _gather(frame, empty, spine.index)
        query_ranks = spine_ranks.take(np.maximum(spine.date_codes, 0))

        positions = latest_at_or_before(
            query_codes,
            query_ranks,
            source_codes,
            source_ranks,
            span=span,
            inclusive=self.allow_exact,
        )
        if self.tolerance is not None:
            positions = self._drop_stale(positions, spine, spine_us, available_us)
        return _gather(frame, positions.astype(np.intp, copy=False), spine.index)

    def _availability(self, panel: Panel, index: pd.MultiIndex) -> pd.Index:
        """The date at which each source row became knowable, lag included."""
        assert panel.date_name is not None
        if self.availability_column is None:
            values = index.get_level_values(resolve_level(index, panel.date_name))
        else:
            if self.availability_column not in panel.frame.columns:
                raise AlignmentError(
                    f"the 'asof_entity' aligner needs its availability column "
                    f"{self.availability_column!r} in the source panel; its columns are "
                    f"{list(panel.frame.columns)}."
                )
            values = pd.Index(panel.frame[self.availability_column])
        if not pd.api.types.is_datetime64_any_dtype(values.dtype):
            raise AlignmentError(
                f"the 'asof_entity' aligner needs a datetime availability date; "
                f"{self.availability_column or panel.date_name!r} is {values.dtype}."
            )
        if self.lag is not None:
            values = values + _timedelta(self.lag, "lag")
        return values

    def _drop_stale(
        self,
        positions: Int64Array,
        spine: Spine,
        spine_us: Int64Array,
        available_us: Int64Array,
    ) -> Int64Array:
        """Null matches older than ``tolerance``, measured on the matched dates.

        The staleness test cannot run on the ranks the lookup used: those are dense
        positions in a shared vocabulary, so the distance between two of them is a
        count of distinct dates, not a number of days.
        """
        assert self.tolerance is not None
        limit = np.int64(
            _timedelta(self.tolerance, "tolerance")
            .to_timedelta64()
            .astype("timedelta64[us]")
            .astype(np.int64)
        )
        matched = positions >= 0
        age = spine_us.take(np.maximum(spine.date_codes, 0)) - available_us.take(
            np.maximum(positions, 0)
        )
        return np.where(matched & (age <= limit), positions, -1).astype(np.int64)


@dataclass(frozen=True)
class BroadcastTimeAligner:
    """Carry a date-keyed source onto every entity trading on that date.

    Market factors, macro series and risk-free rates have one row per date and
    belong to the whole cross-section. The source is aligned to the spine's
    *unique* dates and only then expanded, so a monthly factor file costs one
    lookup per month rather than one per panel row.

    Parameters
    ----------
    as_of:
        When ``True`` a spine date with no source row takes the latest earlier one
        — how a monthly series reaches a daily spine. When ``False`` only exact
        date matches are carried and every other row is null.
    tolerance:
        Maximum age of a backward-filled match, e.g. ``"31D"``. Requires ``as_of``,
        since an exact match has no age.
    """

    as_of: bool = False
    tolerance: str | None = None

    accepts: ClassVar[tuple[PanelGrain, ...]] = (PanelGrain.TIME_SERIES,)

    def __post_init__(self) -> None:
        if self.tolerance is not None:
            _timedelta(self.tolerance, "tolerance")
            if not self.as_of:
                raise ValueError(
                    "tolerance applies to the backward fill only; set as_of=True or drop it."
                )

    def align(self, spine: Spine, panel: Panel) -> pd.DataFrame:
        """Return ``panel``'s columns on ``spine.index``, one source row per date."""
        _require_grain(panel, "broadcast_time", self.accepts)
        assert panel.date_name is not None
        frame = panel.frame
        source = _key_index(frame.index, panel.date_name, aligner="broadcast_time", key="date")
        # Undated rows go first, before the uniqueness check: a spine date level
        # never holds NaT (a missing level value is stored as the code -1), so an
        # undated source row can match nothing on either branch. Counting several
        # of them as duplicate keys would reject a factor file over blank cells
        # that cannot affect the result.
        dated = np.flatnonzero(np.asarray(source.notna()))
        if dated.size != len(source):
            frame = frame.take(dated)
            source = source.take(dated)
        _require_unique(source, "the source panel of a 'broadcast_time' alignment")
        _require_comparable(spine.date_uniques, source, aligner="broadcast_time", key="date")

        if self.as_of:
            # A backward fill over an unsorted index does not raise: it searches a
            # sorted array that is not there and returns later rows for earlier
            # dates, which is look-ahead.
            if not source.is_monotonic_increasing:
                order = np.asarray(source.argsort(), dtype=np.intp)
                frame = frame.take(order)
                source = source.take(order)
            positions = np.asarray(
                source.get_indexer(
                    spine.date_uniques,
                    method="pad",
                    tolerance=(
                        None if self.tolerance is None else _timedelta(self.tolerance, "tolerance")
                    ),
                ),
                dtype=np.intp,
            )
        else:
            positions = np.asarray(source.get_indexer(spine.date_uniques), dtype=np.intp)
        return _broadcast(frame, positions, spine.date_codes, spine.index)


@dataclass(frozen=True)
class BroadcastEntityAligner:
    """Carry an entity-keyed source onto every date of that entity.

    Sector codes, exchange codes and the rest of the security master have one row
    per entity and no time dimension, so each source row reaches every date the
    entity appears on. As with the time broadcast the source is aligned to the
    spine's unique entities first, so the cost is one lookup per entity.
    """

    accepts: ClassVar[tuple[PanelGrain, ...]] = (PanelGrain.CROSS_SECTION,)

    def align(self, spine: Spine, panel: Panel) -> pd.DataFrame:
        """Return ``panel``'s columns on ``spine.index``, one source row per entity."""
        _require_grain(panel, "broadcast_entity", self.accepts)
        assert panel.entity_name is not None
        frame = panel.frame
        source = _key_index(
            frame.index, panel.entity_name, aligner="broadcast_entity", key="entity"
        )
        _require_unique(source, "the source panel of a 'broadcast_entity' alignment")
        _require_comparable(spine.entity_uniques, source, aligner="broadcast_entity", key="entity")
        positions = np.asarray(source.get_indexer(spine.entity_uniques), dtype=np.intp)
        return _broadcast(frame, positions, spine.entity_codes, spine.index)


ALIGNERS: dict[str, type] = {
    "exact": ExactAligner,
    "asof_entity": AsOfEntityAligner,
    "broadcast_time": BroadcastTimeAligner,
    "broadcast_entity": BroadcastEntityAligner,
}
"""Registry names for the join semantics, resolved by :mod:`nekron.align.registry`."""
