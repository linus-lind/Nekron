"""Rewriting a panel's entity key into the spine's identifier space.

Panels arrive keyed by whatever identifier their vendor uses: Compustat by
``gvkey`` plus ``iid``, CRSP by ``permno``, a broker file by ticker, a reference
file by CUSIP. Before any of them can be aligned onto the spine, their entity key
has to be rewritten into the spine's own identifier. That rewrite is the whole
subject of this module, and it has three shapes:

* the panel already carries the target identifier (:meth:`LinkTable.identity`),
* a time-INVARIANT map from key to target — one target per key, forever,
* a time-VARIANT link file, whose rows are valid over ``[valid_from, valid_to]``
  windows because a gvkey maps to different permnos over its life.

The time-variant case is where the silent errors live, and two of them are worth
naming because both produce a plausible panel rather than an exception:

* ``merge_asof(direction="backward")`` followed by a ``date <= valid_to`` filter
  is only correct when a key's windows do not overlap. Given ``A: [2000, 2010]``
  and ``A: [2003, 2005]``, a query for 2007 finds the *later-starting* window,
  fails the ``valid_to`` bound and reports no match — even though the first window
  covers it. Overlap is therefore validated at construction and, when policy
  allows, rewritten into disjoint windows, so :meth:`LinkTable.resolve` is always
  on the exact fast path.
* Rewriting the entity key can map two source keys onto one target on the same
  date (two share classes, one permno). Left alone that duplicates index entries,
  which silently fans out every later join. :func:`relink` detects the collapse
  and makes the caller choose what it means.

Nothing here re-implements the lookup itself: the searchsorted core in
:mod:`nekron.align.keys` answers "the latest row of this key at or before this
date" for both the as-of aligners and this resolver.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from nekron.align.keys import Int64Array, date_ranks, joint_codes, latest_at_or_before, pack
from nekron.constants import ENTITY_LEVEL
from nekron.frames import IntpArray, level_keys, take_filled
from nekron.panel import Panel, PanelGrain

BoolArray = npt.NDArray[np.bool_]
DateArray = npt.NDArray[np.datetime64]

Ambiguity = Literal["error", "first", "last"]
Unmatched = Literal["error", "drop", "keep"]
Collision = Literal["error", "first", "last", "sum", "mean"]

# Substituted for a missing window bound. A link row with no start was always
# valid; one with no end still is. Both are ordinary dates so they rank with
# every other date instead of falling into the NaT slot, which sorts below
# everything and would silently make an open window match nothing.
_OPEN_START = np.datetime64("1678-01-01", "us")
_OPEN_END = np.datetime64("9999-12-31", "us")


class LinkError(Exception):
    """Base class for identifier-linking errors."""


@dataclass(frozen=True)
class KeyNormalization:
    """How one identifier column is cleaned before it is matched.

    Identifiers are compared exactly, so every difference in presentation is a
    lost match. The defaults fix the two differences that are never meaningful —
    surrounding whitespace, and an empty string standing in for "unknown". The
    other two are opt-in *per column* because each destroys information in some
    other column: upper-casing merges Compustat share classes whose case is
    significant, and zero-padding a two-character ``iid`` to a six-character
    ``gvkey`` width turns every key into something that matches nothing.

    Parameters
    ----------
    strip:
        Remove surrounding whitespace.
    upper:
        Upper-case the value.
    zfill:
        Left-pad with zeros to this width.
    empty_as_missing:
        Treat an empty string as missing, so it can never match — including
        against another empty string.
    """

    strip: bool = True
    upper: bool = False
    zfill: int | None = None
    empty_as_missing: bool = True

    @property
    def is_noop(self) -> bool:
        """Whether this normalization would leave every value untouched."""
        return not (self.strip or self.upper or self.empty_as_missing) and self.zfill is None

    def apply(self, values: pd.Series) -> pd.Series:
        """Return ``values`` normalized, preserving dtype wherever possible.

        Non-textual columns pass through untouched: an integer permno has no
        whitespace to strip, and casting it to text to find that out would be
        both slow and a new source of mismatches. A categorical is normalized on
        its *categories* — a few hundred values instead of a few million rows —
        and stays categorical, because materializing it as object costs about
        fifty times the memory.
        """
        if self.is_noop:
            return values
        dtype = values.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return self._apply_to_categories(values)
        if not (pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)):
            return values
        if pd.api.types.is_object_dtype(dtype) and not _looks_textual(values):
            return values
        out = values
        if self.strip:
            out = _string_op(out, lambda accessor: accessor.strip())
        if self.empty_as_missing:
            out = out.mask(out.eq(""))
        if self.upper:
            out = _string_op(out, lambda accessor: accessor.upper())
        if self.zfill is not None:
            width = self.zfill
            out = _string_op(out, lambda accessor: accessor.zfill(width))
        return out

    def _apply_to_categories(self, values: pd.Series) -> pd.Series:
        """Normalize a categorical by rewriting its categories, not its rows."""
        categories = pd.Series(values.cat.categories)
        normalized = self.apply(categories)
        if normalized is categories:
            return values
        remapped, uniques = pd.factorize(normalized, use_na_sentinel=True)
        codes = np.asarray(values.cat.codes, dtype=np.int64)
        # Normalization can collapse two categories into one, or drop one to
        # missing; both are just a different code for the affected rows.
        new_codes = np.where(codes < 0, -1, np.asarray(remapped).take(np.maximum(codes, 0)))
        return pd.Series(
            _categorical_from_codes(new_codes.astype(np.int64), pd.Index(uniques)),
            index=values.index,
        )


def _categorical_from_codes(codes: npt.NDArray[np.int64], categories: pd.Index) -> pd.Categorical:
    """Build a categorical straight from integer codes, without materialising values.

    Passing an array of codes is the whole point here — it is what lets an index
    level be normalized through its vocabulary rather than row by row — and
    ``Categorical.from_codes`` documents ``codes`` as array-like. Older
    ``pandas-stubs`` narrow it to ``Sequence[int]``, so the cast is about the stub
    and not about the runtime; converting to a list to satisfy it would allocate a
    Python object per row.
    """
    return pd.Categorical.from_codes(cast("Sequence[int]", codes), categories=categories)


_DEFAULT_NORMALIZATION = KeyNormalization()


def _looks_textual(values: pd.Series) -> bool:
    """Whether an object column holds strings, so the ``.str`` accessor applies."""
    return pd.api.types.infer_dtype(values, skipna=True) in {
        "string",
        "unicode",
        "bytes",
        "mixed",
        "mixed-integer",
        "empty",
    }


def _string_op(values: pd.Series, operation: Callable[[Any], pd.Series]) -> pd.Series:
    """Apply a ``.str`` operation, leaving non-string entries alone.

    The ``.str`` accessor returns null for every entry that is not a string, so
    an object column holding a stray integer would lose it. Restoring the
    original wherever the operation invented a null keeps normalization from
    deleting data it does not understand.
    """
    result = operation(values.str)
    return result.where(result.notna() | values.isna(), values)


@dataclass(frozen=True)
class LinkReport:
    """What a resolution matched, counted while it was matching.

    The counts come out of the key codes the resolver already computed; deriving
    them afterwards means factorizing the key columns a second time, which
    measured about fourteen times the cost of the arithmetic here.

    Parameters
    ----------
    n_rows:
        Rows offered to the resolver.
    n_unmatched:
        Rows that resolved to nothing, including rows with a null key.
    n_keys:
        Distinct non-null keys among those rows.
    n_keys_unmatched:
        Distinct keys that resolved on *no* row. A time-variant link can leave a
        key matched on some dates and unmatched on others; such a key is not
        counted here, so the two numbers answer different questions — how much
        data is missing, and how much of the identifier universe is unknown.
    n_collapsed:
        Rows removed because they landed on a target another row already held.
    """

    n_rows: int
    n_unmatched: int
    n_keys: int
    n_keys_unmatched: int
    n_collapsed: int = 0

    def __str__(self) -> str:
        share = self.n_unmatched / self.n_rows if self.n_rows else 0.0
        text = (
            f"{self.n_unmatched:,}/{self.n_rows:,} rows ({share:.1%}) and "
            f"{self.n_keys_unmatched:,}/{self.n_keys:,} keys did not resolve"
        )
        if self.n_collapsed:
            text += f"; {self.n_collapsed:,} rows collapsed onto a shared target"
        return text


@dataclass(frozen=True)
class _Resolution:
    """A completed match: which rows failed, how to fetch the targets, what it cost."""

    unmatched: BoolArray
    report: LinkReport
    gather: Callable[[IntpArray], pd.Series]


@dataclass(frozen=True, eq=False, repr=False)
class LinkTable:
    """A mapping from one identifier space to another, optionally time-varying.

    The table is validated and, where necessary, rewritten at construction, so
    that :meth:`resolve` is a single pass with no defensive work in it. Rows
    whose key or target is null are dropped — they can never match anything —
    and overlapping windows are either rejected or canonicalized into disjoint
    ones, depending on ``on_ambiguous``.

    Parameters
    ----------
    table:
        The link rows. A validated, possibly filtered and rewritten copy is
        stored; the frame passed in is never modified.
    source_keys:
        Columns of ``table`` that together identify the source entity. Several
        columns form one composite key (``gvkey`` + ``iid``), matched jointly.
    target:
        Column of ``table`` holding the identifier to map onto.
    valid_from:
        Column holding the inclusive start of each row's validity window. ``None``
        makes the table time-invariant, which is a different and much cheaper
        lookup, not a window from the beginning of time.
    valid_to:
        Column holding the inclusive end. ``None`` — with ``valid_from`` set —
        makes each row valid until the key's next row starts, which is the step
        function a vendor file with only an effective date describes. A null or
        far-future value in the column itself means the window is still open.
    normalize:
        Per-column overrides of :class:`KeyNormalization`. Columns not named here
        get the safe default (strip, empty-as-missing).
    on_ambiguous:
        What to do when one key maps to more than one target at the same time:
        ``"error"`` refuses the table naming the offending keys, ``"first"`` and
        ``"last"`` keep the earliest or latest row of the table.
    passthrough:
        Set by :meth:`identity`: the frame already carries the target, so
        resolution is a rename rather than a lookup.
    """

    table: pd.DataFrame
    source_keys: tuple[str, ...]
    target: str
    valid_from: str | None = None
    valid_to: str | None = None
    normalize: Mapping[str, KeyNormalization] = field(default_factory=dict)
    on_ambiguous: Ambiguity = "error"
    passthrough: bool = False

    _key_columns: tuple[pd.Series, ...] = field(init=False, default=())
    _window_from: DateArray | None = field(init=False, default=None)
    _window_to: DateArray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.source_keys:
            raise LinkError("a link table needs at least one source key column.")
        # The key/target collision check deliberately comes after the identity
        # branch below: an identity link maps a column onto the spine's entity
        # level, and with the project's default level name those are routinely the
        # same name. Checking first rejected the very case the check's own message
        # recommends.
        if not self.passthrough and self.target in self.source_keys:
            raise LinkError(
                f"the target column {self.target!r} cannot also be a source key; a link that "
                f"maps a key onto itself is LinkTable.identity."
            )
        if self.on_ambiguous not in ("error", "first", "last"):
            raise LinkError(
                f"on_ambiguous must be 'error', 'first' or 'last'; got {self.on_ambiguous!r}."
            )
        unknown = sorted(set(self.normalize) - set(self.source_keys))
        if unknown:
            raise LinkError(
                f"normalize names columns that are not source keys: {unknown}; "
                f"source keys are {list(self.source_keys)}."
            )
        if self.passthrough:
            if len(self.source_keys) != 1:
                raise LinkError("an identity link resolves exactly one column.")
            if self.valid_from is not None or self.valid_to is not None:
                raise LinkError("an identity link cannot carry validity windows.")
            return
        if self.valid_to is not None and self.valid_from is None:
            raise LinkError(
                "valid_to needs valid_from; a window with only an end has no start to rank."
            )
        required = [*self.source_keys, self.target]
        required += [name for name in (self.valid_from, self.valid_to) if name is not None]
        missing = [name for name in required if name not in self.table.columns]
        if missing:
            raise LinkError(
                f"link table is missing columns {missing}; it has {list(self.table.columns)}."
            )
        self._prepare()

    def __repr__(self) -> str:
        keys = "+".join(self.source_keys)
        if self.passthrough:
            return f"LinkTable(identity {keys} -> {self.target})"
        kind = "time-variant" if self.is_time_variant else "time-invariant"
        return f"LinkTable({keys} -> {self.target}, {len(self.table)} rows, {kind})"

    @property
    def is_time_variant(self) -> bool:
        """Whether rows of this table are only valid over a date window."""
        return self.valid_from is not None

    @classmethod
    def identity(cls, column: str, target: str) -> LinkTable:
        """A link for a panel that already carries the target identifier.

        This is the common case and it deserves the cheap path: resolution is a
        null check on ``column`` and a rename to ``target``, with none of the
        coding, ranking or searching the real lookup needs.
        """
        empty = pd.DataFrame({column: pd.Series(dtype=object), target: pd.Series(dtype=object)})
        return cls(table=empty, source_keys=(column,), target=target, passthrough=True)

    @classmethod
    def from_spine(
        cls,
        frame: pd.DataFrame,
        source_keys: Sequence[str],
        *,
        date_name: str,
        entity_name: str,
        target: str = ENTITY_LEVEL,
        close_last: bool = False,
        **kwargs: Any,
    ) -> LinkTable:
        """Project a panel's own identifier columns into a time-variant link table.

        A spine that carries, say, a ticker alongside its entity key already is a
        link table — it just has one row per date rather than one row per window.
        Compressing it is where the obvious implementation goes wrong: taking each
        key's first and last date yields one window, and a ticker that was used,
        abandoned and later recycled to a different firm then appears to have been
        valid throughout the gap, shadowing whoever actually held it. Runs are
        therefore broken wherever a key skips a date the spine *does* have, which
        is the only definition of "abandoned" the spine can support.

        Parameters
        ----------
        frame:
            The spine's frame. Keys, dates and the entity may each be a column or
            an index level.
        source_keys:
            Identifier columns to map *from*.
        date_name, entity_name:
            Names of the spine's date and entity keys.
        target:
            Name given to the entity column in the resulting table.
        close_last:
            Whether a run that reaches the spine's final date is closed there. It
            is left open by default, because the mapping did not end — the data
            did, and a later panel row should still resolve.
        kwargs:
            Passed to the constructor (``normalize``, ``on_ambiguous``).
        """
        keys = list(source_keys)
        if target in keys:
            raise LinkError(f"the target column {target!r} cannot also be a source key.")
        normalize: Mapping[str, KeyNormalization] = kwargs.get("normalize", {})
        key_series = [
            normalize.get(name, _DEFAULT_NORMALIZATION).apply(_key_series(frame, name))
            for name in keys
        ]
        entity = _key_series(frame, entity_name)
        dates = _as_microsecond_dates(_raw_date_series(frame, date_name))

        # A key and the entity it points at form one mapping; grouping by the pair
        # is what lets a recycled key produce two windows instead of one.
        pair_codes, _, _ = joint_codes([*key_series, entity], [*key_series, entity])
        vocabulary, ranks = np.unique(dates, return_inverse=True)
        ranks = np.asarray(ranks, dtype=np.int64).reshape(-1)
        usable = (pair_codes >= 0) & ~pd.isna(dates)
        rows = np.flatnonzero(usable)
        if rows.size == 0:
            table = pd.DataFrame(
                {
                    **{
                        name: series.iloc[:0].reset_index(drop=True)
                        for name, series in zip(keys, key_series, strict=True)
                    },
                    target: entity.iloc[:0].reset_index(drop=True),
                    "valid_from": pd.Series(dtype="datetime64[us]"),
                    "valid_to": pd.Series(dtype="datetime64[us]"),
                }
            )
            return cls(
                table=table,
                source_keys=tuple(keys),
                target=target,
                valid_from="valid_from",
                valid_to="valid_to",
                **kwargs,
            )

        order = rows[np.lexsort((ranks[rows], pair_codes[rows]))]
        pairs = pair_codes[order]
        stamps = ranks[order]
        # Several rows may share a (key, date); only the transitions matter.
        keep = np.empty(order.size, dtype=bool)
        keep[0] = True
        keep[1:] = (pairs[1:] != pairs[:-1]) | (stamps[1:] != stamps[:-1])
        order, pairs, stamps = order[keep], pairs[keep], stamps[keep]

        starts = np.empty(order.size, dtype=bool)
        starts[0] = True
        starts[1:] = (pairs[1:] != pairs[:-1]) | (stamps[1:] - stamps[:-1] > 1)
        start_at = np.flatnonzero(starts)
        end_at = np.append(start_at[1:] - 1, order.size - 1)

        window_from = vocabulary[stamps[start_at]]
        window_to = vocabulary[stamps[end_at]]
        if not close_last:
            open_ended = stamps[end_at] == len(vocabulary) - 1
            window_to = np.where(open_ended, np.datetime64("NaT", "us"), window_to)

        representatives = order[start_at]
        table = pd.DataFrame(
            {
                **{
                    name: series.take(representatives).reset_index(drop=True)
                    for name, series in zip(keys, key_series, strict=True)
                },
                target: entity.take(representatives).reset_index(drop=True),
                "valid_from": window_from,
                "valid_to": window_to,
            }
        )
        return cls(
            table=table,
            source_keys=tuple(keys),
            target=target,
            valid_from="valid_from",
            valid_to="valid_to",
            **kwargs,
        )

    def resolve(
        self, frame: pd.DataFrame, *, dates: np.ndarray | None = None
    ) -> tuple[np.ndarray, LinkReport]:
        """Return the target identifier for every row of ``frame``, plus a report.

        Key columns are looked for among ``frame``'s columns first and its index
        level names second. ``dates`` is required for a time-variant table and
        ignored otherwise; a row whose date is missing resolves to nothing rather
        than to whichever window happens to sort first.

        Unmatched rows carry the null of the resulting dtype. Since a link row
        with a null target is dropped at construction, "null" and "unmatched"
        mean the same thing in the returned array.
        """
        resolution = self._resolve(frame, dates)
        values = resolution.gather(np.arange(len(frame), dtype=np.intp))
        return values.to_numpy(), resolution.report

    def _resolve(self, frame: pd.DataFrame, dates: np.ndarray | None) -> _Resolution:
        """Match every row of ``frame``, returning a lazy gatherer for the targets."""
        n_rows = len(frame)
        if self.passthrough:
            return self._resolve_passthrough(frame, n_rows)
        query = [self._normalized(_key_series(frame, name), name) for name in self.source_keys]
        for name, left_column, right_column in zip(
            self.source_keys, query, self._key_columns, strict=True
        ):
            _require_comparable(name, left_column, right_column)
        left, right, n_codes = joint_codes(query, list(self._key_columns))

        if self.is_time_variant:
            positions = self._resolve_windows(left, right, dates)
        else:
            slot = np.full(max(n_codes, 1), -1, dtype=np.int64)
            slot[right] = np.arange(right.size, dtype=np.int64)
            positions = np.where(left >= 0, slot[np.maximum(left, 0)], -1)

        matched = positions >= 0
        report = _count(left, matched, n_codes, n_rows)
        block = self.table[[self.target]]

        def gather(rows: IntpArray) -> pd.Series:
            taken = take_filled(block, np.asarray(positions[rows], dtype=np.intp))
            return taken[self.target]

        return _Resolution(unmatched=~matched, report=report, gather=gather)

    def _resolve_passthrough(self, frame: pd.DataFrame, n_rows: int) -> _Resolution:
        """Resolve an identity link: check the column, then hand it over as-is."""
        column = self.source_keys[0]
        values = self._normalized(_key_series(frame, column), column).reset_index(drop=True)
        unmatched = np.asarray(values.isna().to_numpy(), dtype=bool)
        if n_rows and unmatched.all():
            raise LinkError(
                f"column {column!r} was declared to already carry {self.target!r} but every "
                f"one of its {n_rows} values is missing."
            )
        n_keys = int(pd.Series(values[~unmatched]).nunique()) if n_rows else 0
        report = LinkReport(
            n_rows=n_rows,
            n_unmatched=int(unmatched.sum()),
            n_keys=n_keys,
            n_keys_unmatched=0,
        )
        return _Resolution(
            unmatched=unmatched,
            report=report,
            gather=lambda rows: values.take(rows).reset_index(drop=True).rename(self.target),
        )

    def _resolve_windows(
        self, left: Int64Array, right: Int64Array, dates: np.ndarray | None
    ) -> Int64Array:
        """Per row, the position of the window covering its date; ``-1`` if none."""
        if dates is None:
            raise LinkError(
                f"{self!r} is time-variant, so resolving it needs one date per row; pass dates=."
            )
        assert self._window_from is not None and self._window_to is not None
        if len(dates) != len(left):
            raise LinkError(
                f"dates must have one entry per row; got {len(dates)} for {len(left)} rows."
            )
        query = _as_microsecond_dates(pd.Series(dates))
        # Rank the distinct query dates, not every row. A panel has far fewer dates
        # than rows — a daily panel of 500 names has one date per 500 rows — and
        # ranking is a binary search per element, so factorizing first turns the
        # dominant cost of the resolve into a cheap take.
        date_codes, unique_dates = pd.factorize(query, use_na_sentinel=True)
        (unique_rank, from_rank, to_rank), span = date_ranks(
            np.asarray(unique_dates), self._window_from, self._window_to
        )
        query_rank = unique_rank.take(np.maximum(date_codes, 0))
        positions = latest_at_or_before(
            left, query_rank, right, from_rank, span=span, inclusive=True
        )
        covered = positions >= 0
        if self.valid_to is not None and to_rank.size:
            # The as-of search only guarantees the window had started. Windows are
            # disjoint after construction, so the one it found is the only
            # candidate and this bound check is exact.
            covered &= query_rank <= to_rank[np.where(covered, positions, 0)]
        covered &= ~pd.isna(query)
        return np.where(covered, positions, -1).astype(np.int64)

    def _normalized(self, values: pd.Series, name: str) -> pd.Series:
        """Apply this table's normalization for one key column."""
        return self.normalize.get(name, _DEFAULT_NORMALIZATION).apply(values)

    # -- construction ----------------------------------------------------------

    def _prepare(self) -> None:
        """Validate the table and rewrite it into the shape ``resolve`` assumes."""
        frame = self.table.reset_index(drop=True)
        keys = [self._normalized(frame[name], name) for name in self.source_keys]
        codes, _, _ = joint_codes(keys, keys)
        codes = codes[: len(frame)]

        window_from: DateArray | None = None
        window_to: DateArray | None = None
        if self.valid_from is not None:
            window_from = _fill_missing(_as_microsecond_dates(frame[self.valid_from]), _OPEN_START)
            if self.valid_to is not None:
                window_to = _fill_missing(_as_microsecond_dates(frame[self.valid_to]), _OPEN_END)
            else:
                # Without an end column each row runs until the key's next row
                # starts, so the upper bound is never consulted; a degenerate
                # window is enough to make duplicate starts detectable.
                window_to = window_from

        usable = (codes >= 0) & frame[self.target].notna().to_numpy()
        if not usable.all():
            rows = np.flatnonzero(usable)
            frame = frame.take(rows).reset_index(drop=True)
            keys = [series.take(rows).reset_index(drop=True) for series in keys]
            codes = codes[rows]
            if window_from is not None and window_to is not None:
                window_from, window_to = window_from[rows], window_to[rows]

        targets = np.asarray(pd.factorize(frame[self.target], use_na_sentinel=True)[0], np.int64)
        if self.valid_from is None:
            frame, keys = self._canonicalize_invariant(frame, keys, codes, targets)
        else:
            assert window_from is not None and window_to is not None
            frame, keys, window_from, window_to = self._canonicalize_windows(
                frame, keys, codes, targets, window_from, window_to
            )

        object.__setattr__(self, "table", frame)
        object.__setattr__(self, "_key_columns", tuple(keys))
        object.__setattr__(self, "_window_from", window_from)
        object.__setattr__(self, "_window_to", window_to)

    def _canonicalize_invariant(
        self,
        frame: pd.DataFrame,
        keys: list[pd.Series],
        codes: Int64Array,
        targets: Int64Array,
    ) -> tuple[pd.DataFrame, list[pd.Series]]:
        """Reduce a time-invariant table to exactly one row per key."""
        order = np.lexsort((targets, codes))
        by_key, by_target = codes[order], targets[order]
        if by_key.size > 1:
            same_key = by_key[1:] == by_key[:-1]
            conflicting = np.unique(by_key[1:][same_key & (by_target[1:] != by_target[:-1])])
        else:
            conflicting = np.empty(0, dtype=np.int64)
        if conflicting.size and self.on_ambiguous == "error":
            rows = _example_rows(codes, conflicting)
            raise LinkError(
                f"{conflicting.size} keys map to more than one {self.target!r}, e.g. "
                f"{_describe(frame, self.source_keys, rows)}; set on_ambiguous='first' or "
                f"'last' to pick one, or add validity windows."
            )
        stable = np.lexsort((np.arange(codes.size), codes))
        sorted_codes = codes[stable]
        edge = np.empty(codes.size, dtype=bool)
        if self.on_ambiguous == "last":
            edge[-1:] = True
            edge[:-1] = sorted_codes[1:] != sorted_codes[:-1]
        else:
            edge[:1] = True
            edge[1:] = sorted_codes[1:] != sorted_codes[:-1]
        chosen = np.sort(stable[edge]) if codes.size else np.empty(0, dtype=np.intp)
        if chosen.size == codes.size:
            return frame, keys
        rows = np.asarray(chosen, dtype=np.intp)
        return (
            frame.take(rows).reset_index(drop=True),
            [series.take(rows).reset_index(drop=True) for series in keys],
        )

    def _canonicalize_windows(
        self,
        frame: pd.DataFrame,
        keys: list[pd.Series],
        codes: Int64Array,
        targets: Int64Array,
        window_from: DateArray,
        window_to: DateArray,
    ) -> tuple[pd.DataFrame, list[pd.Series], DateArray, DateArray]:
        """Reject or rewrite overlapping windows so every key's timeline is disjoint."""
        (from_rank, to_rank), span = date_ranks(window_from, window_to)
        backwards = from_rank > to_rank
        if backwards.any():
            rows = np.flatnonzero(backwards)
            raise LinkError(
                f"{rows.size} link windows end before they start, e.g. "
                f"{_describe(frame, self.source_keys, rows[:5])}."
            )

        # Packing key and rank into one int64 makes each key's rows contiguous and
        # ordered, so a single running maximum over the sorted starts finds every
        # overlap: an earlier key's largest packed end is always below the next
        # key's smallest packed start, and no per-key reset is needed.
        packed_lo = pack(codes, from_rank, span)
        packed_hi = pack(codes, to_rank, span)
        order = np.lexsort((packed_hi, packed_lo))
        lo, hi = packed_lo[order], packed_hi[order]
        overlapping = np.zeros(lo.size, dtype=bool)
        if lo.size > 1:
            overlapping[1:] = lo[1:] <= np.maximum.accumulate(hi)[:-1]
        affected = np.unique(codes[order[overlapping]])
        if affected.size == 0:
            return frame, keys, window_from, window_to

        starts_us = window_from.astype("datetime64[us]").astype(np.int64)
        ends_us = window_to.astype("datetime64[us]").astype(np.int64)
        by_key = np.argsort(codes, kind="stable")
        grouped = codes[by_key]
        group_start = np.searchsorted(grouped, affected, side="left")
        group_end = np.searchsorted(grouped, affected, side="right")

        conflicting: list[int] = []
        rewritten: list[tuple[int, int, int]] = []
        for key, begin, end in zip(affected, group_start, group_end, strict=True):
            rows = by_key[begin:end]
            segments, conflicted = _split_timeline(
                rows, starts_us[rows], ends_us[rows], targets[rows], self.on_ambiguous
            )
            rewritten.extend(segments)
            if conflicted:
                conflicting.append(int(key))
        if conflicting and self.on_ambiguous == "error":
            rows = _example_rows(codes, np.asarray(conflicting, dtype=np.int64))
            raise LinkError(
                f"{len(conflicting)} keys have overlapping windows that disagree on "
                f"{self.target!r}, e.g. {_describe(frame, self.source_keys, rows)}; set "
                f"on_ambiguous='first' or 'last' to split the overlap."
            )

        untouched = np.flatnonzero(~np.isin(codes, affected))
        positions = np.concatenate(
            [untouched, np.asarray([row for row, _, _ in rewritten], dtype=np.intp)]
        ).astype(np.intp)
        new_from = np.concatenate(
            [starts_us[untouched], np.asarray([lo for _, lo, _ in rewritten], dtype=np.int64)]
        ).astype("datetime64[us]")
        new_to = np.concatenate(
            [ends_us[untouched], np.asarray([hi for _, _, hi in rewritten], dtype=np.int64)]
        ).astype("datetime64[us]")

        frame = frame.take(positions).reset_index(drop=True)
        keys = [series.take(positions).reset_index(drop=True) for series in keys]
        assert self.valid_from is not None
        frame[self.valid_from] = new_from
        if self.valid_to is not None:
            frame[self.valid_to] = new_to
        return frame, keys, new_from, new_to


def _split_timeline(
    rows: IntpArray,
    starts: Int64Array,
    ends: Int64Array,
    targets: Int64Array,
    policy: Ambiguity,
) -> tuple[list[tuple[int, int, int]], bool]:
    """Cut one key's overlapping windows into disjoint segments.

    Every window boundary becomes a cut point, so each elementary segment is
    covered by a fixed set of rows; the policy picks one of them (``rows`` is in
    table order, so "first" is the earliest row of the file). Segments that end up
    with the same winner and touch are merged again, which keeps a table that was
    merely duplicated from growing. The flag reports whether any segment was
    covered by rows that disagreed — a genuine ambiguity, as opposed to a
    redundant window repeating what another one already said.
    """
    boundaries = np.unique(np.concatenate([starts, ends + 1]))
    segments: list[tuple[int, int, int]] = []
    conflicted = False
    previous = -1
    for lo, next_lo in zip(boundaries[:-1], boundaries[1:], strict=True):
        hi = int(next_lo) - 1
        covering = np.flatnonzero((starts <= lo) & (ends >= lo))
        if covering.size == 0:
            previous = -1
            continue
        if covering.size > 1 and np.unique(targets[covering]).size > 1:
            conflicted = True
        winner = int(rows[covering[-1] if policy == "last" else covering[0]])
        if winner == previous and segments and segments[-1][2] == int(lo) - 1:
            row, start, _ = segments[-1]
            segments[-1] = (row, start, hi)
        else:
            segments.append((winner, int(lo), hi))
        previous = winner
    return segments, conflicted


def relink(
    panel: Panel,
    link: LinkTable,
    *,
    entity_name: str,
    on_unmatched: Unmatched = "drop",
    on_collision: Collision = "error",
) -> tuple[Panel, LinkReport]:
    """Rewrite ``panel``'s entity key into the identifier space ``link`` targets.

    The date level and its name are kept exactly as they were; only the entity
    level is replaced. Two things can go wrong afterwards and both are decisions,
    not defaults: rows whose key does not resolve, and rows that now share a
    ``(date, target)`` with another row because two source identifiers map onto
    one target. The second is the dangerous one — left in place it duplicates
    index entries, and every later join silently fans out.

    Parameters
    ----------
    panel:
        The panel to rewrite. Its grain must carry an entity.
    link:
        The mapping to apply.
    entity_name:
        Name of the new entity level.
    on_unmatched:
        ``"error"`` refuses the panel, ``"drop"`` removes unresolved rows,
        ``"keep"`` retains them with a null entity.
    on_collision:
        What two rows sharing a target on one date mean: ``"error"`` refuses,
        ``"first"``/``"last"`` keep one of them, ``"sum"``/``"mean"`` aggregate
        the columns.
    """
    panel.require_grain(PanelGrain.PANEL, PanelGrain.CROSS_SECTION, what="relinking a panel")
    if on_unmatched not in ("error", "drop", "keep"):
        raise LinkError(f"on_unmatched must be 'error', 'drop' or 'keep'; got {on_unmatched!r}.")
    if on_collision not in ("error", "first", "last", "sum", "mean"):
        raise LinkError(
            f"on_collision must be 'error', 'first', 'last', 'sum' or 'mean'; got {on_collision!r}."
        )
    frame = panel.frame
    # The date level is carried over untouched rather than through the
    # microsecond normalization the resolver uses internally: changing its unit
    # would leave the relinked panel unable to align against anything else.
    dates: np.ndarray | None = None
    if panel.date_name is not None:
        dates = _raw_date_series(frame, panel.date_name).to_numpy()
    elif link.is_time_variant:
        raise LinkError(
            f"{link!r} is time-variant, but a {panel.grain.value} panel has no date to "
            f"resolve it at."
        )

    source_entity = _key_series(frame, panel.entity_name) if panel.entity_name else None
    resolution = link._resolve(frame, dates)
    report = resolution.report
    rows = np.arange(len(frame), dtype=np.intp)
    if on_unmatched == "error" and report.n_unmatched:
        raise LinkError(f"{report}; relinking onto {entity_name!r} refuses to lose them.")
    if on_unmatched == "drop":
        rows = rows[~resolution.unmatched]

    targets = resolution.gather(rows)
    levels: list[Any] = []
    names: list[str] = []
    if panel.date_name is not None:
        assert dates is not None
        levels.append(dates.take(rows))
        names.append(panel.date_name)
    levels.append(targets)
    names.append(entity_name)
    index: pd.Index = (
        pd.MultiIndex.from_arrays(levels, names=names)
        if len(levels) > 1
        else pd.Index(targets, name=entity_name)
    )
    kept = frame if rows.size == len(frame) else frame.take(rows)
    result = kept.set_axis(index, axis=0)

    # The same check as ``duplicated(subset=[date, target], keep=False)``, done on
    # the index that now carries both of them. Rows that did not resolve are
    # exempt: under ``on_unmatched="keep"`` they all carry a null entity, and two
    # nulls are not two identifiers landing on one target — they are two rows that
    # landed on nothing, which ``on_unmatched`` has already ruled on.
    collided = np.asarray(index.duplicated(keep=False), dtype=bool)
    collided &= ~np.asarray(pd.isna(targets), dtype=bool)
    if not collided.any():
        return Panel(frame=result, date_name=panel.date_name, entity_name=entity_name), report

    before = len(result)
    if on_collision == "error":
        examples = index[collided].unique()[:5]
        raise LinkError(
            f"{int(collided.sum())} rows share a ({', '.join(names)}) with another row after "
            f"relinking, e.g. {list(examples)!r}; two source identifiers map onto one "
            f"{entity_name!r}. Set on_collision to 'first', 'last', 'sum' or 'mean'."
        )
    if on_collision in ("first", "last"):
        result = _keep_one(result, levels, names, source_entity, rows, on_collision)
    else:
        numeric = [
            str(name)
            for name in result.columns
            if not pd.api.types.is_numeric_dtype(result[name].dtype)
        ]
        if numeric:
            raise LinkError(
                f"on_collision={on_collision!r} can only aggregate numeric columns; "
                f"{numeric} are not."
            )
        result = result.groupby(
            level=list(range(index.nlevels)), sort=True, observed=True, dropna=False
        ).agg(on_collision)
    report = replace(report, n_collapsed=before - len(result))
    return Panel(frame=result, date_name=panel.date_name, entity_name=entity_name), report


def _keep_one(
    result: pd.DataFrame,
    levels: list[Any],
    names: list[str],
    source_entity: pd.Series | None,
    rows: IntpArray,
    policy: Literal["first", "last"],
) -> pd.DataFrame:
    """Keep one row per collided key, chosen deterministically.

    ``drop_duplicates`` is positional, so on an unsorted frame "first" means
    "whichever row the reader happened to emit first" — reproducible only as long
    as nothing upstream reorders. Sorting by the source identifier before choosing
    makes the winner a property of the data instead. The survivors are then
    restored to the panel's original order.
    """
    keys = {f"__k{position}": level for position, level in enumerate(levels)}
    if source_entity is not None:
        keys["__source"] = source_entity.take(rows).reset_index(drop=True)
    ordering = pd.DataFrame(keys)
    ordering = ordering.sort_values(list(ordering.columns), kind="stable")
    subset = [f"__k{position}" for position in range(len(levels))]
    losers = ordering.duplicated(subset=subset, keep=policy).to_numpy()
    keep = np.ones(len(result), dtype=bool)
    keep[np.asarray(ordering.index)[losers]] = False
    return result.iloc[np.flatnonzero(keep)]


# -- shared helpers ------------------------------------------------------------


def _key_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read one key, from a column if there is one and an index level otherwise.

    Columns win because a panel that carries an identifier as data has usually
    just been given a better one than the level it is indexed by. Index levels are
    read through their codes and handed back as a categorical, so normalizing a
    level touches its vocabulary rather than every row.
    """
    if name in frame.columns:
        return frame[name]
    if name in list(frame.index.names):
        codes, uniques = level_keys(frame.index, name)
        return pd.Series(_categorical_from_codes(np.asarray(codes, dtype=np.int64), uniques))
    raise LinkError(
        f"key {name!r} is neither a column nor an index level; columns are "
        f"{list(frame.columns)} and index levels are {list(frame.index.names)}."
    )


def _raw_date_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read a date key without turning it into a categorical."""
    if name in frame.columns:
        return frame[name]
    if name in list(frame.index.names):
        codes, uniques = level_keys(frame.index, name)
        values = pd.Series(np.asarray(uniques.to_numpy()).take(np.maximum(codes, 0)))
        return values.mask(np.asarray(codes) < 0)
    raise LinkError(
        f"date key {name!r} is neither a column nor an index level; index levels are "
        f"{list(frame.index.names)}."
    )


def _as_microsecond_dates(values: pd.Series | np.ndarray) -> DateArray:
    """Return dates as ``datetime64[us]``, the resolution the windows are kept at.

    :func:`~nekron.align.keys.date_ranks` normalizes the same way before ranking;
    this returns the dates themselves rather than their integer view because a
    link window has to be stored, printed and compared as a date.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise LinkError(
            "timezone-aware dates cannot be matched against naive link windows; "
            "localize or drop the timezone first."
        )
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        textual = pd.api.types.is_object_dtype(series.dtype) or isinstance(
            series.dtype, pd.StringDtype
        )
        # Parsing text straight to microseconds is what lets a "9999-12-31"
        # sentinel survive: pandas 2.2's ``to_datetime`` still defaults to
        # nanoseconds and refuses anything past 2262 outright.
        try:
            series = series.astype("datetime64[us]") if textual else pd.to_datetime(series)
        except (TypeError, ValueError):
            series = pd.to_datetime(series)
    return np.asarray(series.to_numpy().astype("datetime64[us]"))


def _fill_missing(values: DateArray, sentinel: np.datetime64) -> DateArray:
    """Replace missing bounds with an open-ended sentinel."""
    missing = np.isnat(values)
    if not missing.any():
        return values
    return np.asarray(np.where(missing, sentinel, values))


def _require_comparable(name: str, left: pd.Series, right: pd.Series) -> None:
    """Refuse a key column whose two sides could never match.

    Matching a string ticker against an integer one produces no matches and no
    error — the codes simply never coincide — which looks exactly like a panel
    whose identifiers are unknown. Comparing the two dtypes up front turns that
    into a question about the schema.
    """
    left_kind, right_kind = _kind(left), _kind(right)
    if "unknown" in (left_kind, right_kind) or left_kind == right_kind:
        return
    raise LinkError(
        f"key {name!r} is {left.dtype} ({left_kind}) on the panel and {right.dtype} "
        f"({right_kind}) in the link table; they can never match. Cast one side."
    )


def _kind(values: pd.Series) -> str:
    """Classify a key column coarsely enough to catch a hopeless comparison."""
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    if pd.api.types.is_object_dtype(dtype):
        return {
            "string": "string",
            "unicode": "string",
            "integer": "numeric",
            "floating": "numeric",
            "boolean": "bool",
            "datetime64": "datetime",
            "datetime": "datetime",
        }.get(pd.api.types.infer_dtype(values, skipna=True), "unknown")
    return "unknown"


def _count(left: Int64Array, matched: BoolArray, n_codes: int, n_rows: int) -> LinkReport:
    """Turn the codes the resolver already has into the diagnostic counts."""
    known = left >= 0
    present = np.zeros(max(n_codes, 1), dtype=bool)
    present[left[known]] = True
    hit = np.zeros(max(n_codes, 1), dtype=bool)
    hit[left[known & matched]] = True
    return LinkReport(
        n_rows=n_rows,
        n_unmatched=int(n_rows - matched.sum()),
        n_keys=int(present.sum()),
        n_keys_unmatched=int((present & ~hit).sum()),
    )


def _example_rows(codes: Int64Array, wanted: Int64Array, limit: int = 5) -> IntpArray:
    """One representative row per offending key, for an error message."""
    picked = np.unique(wanted)[:limit]
    rows = np.flatnonzero(np.isin(codes, picked))
    if rows.size == 0:
        return rows.astype(np.intp)
    _, first = np.unique(codes[rows], return_index=True)
    return np.asarray(rows[np.sort(first)], dtype=np.intp)


def _describe(frame: pd.DataFrame, source_keys: Sequence[str], rows: npt.NDArray[np.intp]) -> str:
    """Render a few key values the way the user wrote them."""
    shown = []
    for position in rows[:5]:
        values = tuple(frame[name].iloc[int(position)] for name in source_keys)
        shown.append(values[0] if len(values) == 1 else values)
    return repr(shown)
