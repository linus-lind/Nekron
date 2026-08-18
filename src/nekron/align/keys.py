"""Composite-key coding and date ranking, shared by the as-of aligners and the link resolver.

Both problems this package solves — "which quarterly filing was public on this
date" and "which permno did this gvkey map to on this date" — are the same lookup:
given a key and a date, find the latest row of that key at or before that date.
The primitives here reduce that to one :func:`numpy.searchsorted` over an
``int64`` array. The reason to prefer it over :func:`pandas.merge_asof` is
robustness, *not* speed: measured against an equivalent ``merge_asof`` that sorts
both sides, the two are within about 10% of each other at two million rows, and
this path's peak memory can be the higher of the two. What it buys is that
``merge_asof`` requires *both* frames globally sorted on the join date, requires
the ``by`` keys to have byte-identical dtypes, and raises outright on a missing
date. None of those hold reliably here: panels arrive entity-major, a categorical
identifier meets a plain one, and an unfiled quarter has no report date. This path
needs none of them, and sorts only the small side.

Two traps are handled here rather than at each call site, because both fail
silently:

* ``pd.MultiIndex.from_arrays(...).factorize()`` gives a tuple containing NaN a
  perfectly valid code, so null keys match each other. :func:`joint_codes` builds
  the null mask column by column and emits ``-1``, which never matches.
* Casting ``datetime64[us]`` to ``datetime64[ns]`` wraps dates beyond 2262 without
  warning — ``9999-12-31``, the conventional "still valid" sentinel in link
  files, becomes ``1816-03-29``. :func:`date_ranks` normalizes to microseconds
  before any arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

Int64Array = npt.NDArray[np.int64]

# Positions are packed as ``code * span + rank``; re-densify before the running
# span could overflow a signed 64-bit integer.
_MAX_SPAN = 2**62


def joint_codes(
    left: Sequence[npt.ArrayLike], right: Sequence[npt.ArrayLike]
) -> tuple[Int64Array, Int64Array, int]:
    """Factorize an n-column composite key jointly across two sides.

    Both sides are coded against one shared vocabulary, so equal keys get equal
    codes and the codes can be compared directly. A row whose key has a null in
    *any* column is coded ``-1`` and can therefore never match anything —
    including another null key, which is what pandas joins get wrong.

    Returns ``(left_codes, right_codes, n_codes)`` where ``n_codes`` bounds the
    non-negative codes, so ``n_codes`` is a safe out-of-range slot for lookups.
    """
    if len(left) != len(right):
        raise ValueError(
            f"both sides must have the same number of key columns; got {len(left)} and {len(right)}."
        )
    if not left:
        raise ValueError("at least one key column is required.")

    left_columns = [pd.Series(np.asarray(column)) for column in left]
    right_columns = [pd.Series(np.asarray(column)) for column in right]
    n_left = len(left_columns[0])
    n_right = len(right_columns[0])
    total = n_left + n_right

    code = np.zeros(total, dtype=np.int64)
    missing = np.zeros(total, dtype=bool)
    span = 1
    for left_column, right_column in zip(left_columns, right_columns, strict=True):
        joined = pd.concat([left_column, right_column], ignore_index=True)
        column_codes, uniques = pd.factorize(joined, use_na_sentinel=True)
        missing |= column_codes < 0
        width = len(uniques) + 1
        if span > _MAX_SPAN // width:
            code = _densify(code)
            span = int(code.max()) + 1 if total else 1
        code = code * width + (column_codes.astype(np.int64) + 1)
        span *= width

    dense = _densify(code)
    out = np.where(missing, -1, dense).astype(np.int64)
    n_codes = int(out.max()) + 1 if total and bool((out >= 0).any()) else 0
    return out[:n_left], out[n_left:], n_codes


def _densify(code: Int64Array) -> Int64Array:
    """Renumber arbitrary int64 codes into a contiguous ``0..k`` range.

    Hashed, not sorted. Nothing downstream cares what order the dense codes are
    in — they are only compared for equality, used as offsets, and packed with a
    rank — so :func:`numpy.unique`'s full sort is pure overhead; it measured ~19x
    slower than :func:`pandas.factorize` on a two-million-element array.
    """
    if code.size == 0:
        return code
    dense = pd.factorize(code, use_na_sentinel=False)[0]
    return np.asarray(dense, dtype=np.int64).reshape(code.shape)


def date_ranks(*arrays: npt.ArrayLike) -> tuple[list[Int64Array], int]:
    """Rank several date arrays against their shared sorted vocabulary.

    Packing a key code together with a date requires *dense* date positions: raw
    nanosecond timestamps are on the order of 1e18 and overflow the moment they
    are multiplied by anything.

    Every array is normalized to microsecond resolution first, which is what keeps
    far-future sentinels intact. ``NaT`` ranks below every real date; call sites
    that treat a missing date as "unbounded" must substitute the sentinel
    themselves before ranking.
    """
    normalized = [_as_microseconds(array) for array in arrays]
    if not normalized:
        return [], 0
    vocabulary = np.unique(np.concatenate(normalized)) if normalized else np.empty(0, np.int64)
    ranked = [
        np.asarray(np.searchsorted(vocabulary, array), dtype=np.int64) for array in normalized
    ]
    return ranked, int(vocabulary.size)


def _as_microseconds(array: npt.ArrayLike) -> Int64Array:
    """Return a date array's microsecond integer representation."""
    values = np.asarray(array)
    if isinstance(values.dtype, pd.DatetimeTZDtype) or getattr(values.dtype, "tz", None):
        raise ValueError(
            "timezone-aware dates cannot be ranked against naive ones; localize or "
            "remove the timezone before aligning."
        )
    if not np.issubdtype(values.dtype, np.datetime64):
        values = pd.to_datetime(pd.Series(values)).to_numpy()
    return np.asarray(values.astype("datetime64[us]").astype(np.int64), dtype=np.int64)


def pack(codes: Int64Array, ranks: Int64Array, span: int) -> Int64Array:
    """Combine key codes and dense ranks into one monotone ``int64`` sort key."""
    return codes * np.int64(span) + ranks


def latest_at_or_before(
    key_codes: Int64Array,
    ranks: Int64Array,
    source_key_codes: Int64Array,
    source_ranks: Int64Array,
    *,
    span: int,
    inclusive: bool = True,
) -> Int64Array:
    """For each query row, the position of its key's latest source row up to its rank.

    Returns ``-1`` where the key is unknown, has no source row at all, or has none
    early enough. ``inclusive`` decides whether a source row on the query's own
    rank counts — for a point-in-time join that is the question of whether data
    stamped today was knowable today.

    Only the (small) source side is sorted, and only once.
    """
    valid = source_key_codes >= 0
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return np.full(len(key_codes), -1, dtype=np.int64)

    packed_source = pack(source_key_codes[valid], source_ranks[valid], span)
    order = np.argsort(packed_source, kind="stable")
    packed_source = packed_source[order]
    positions = positions[order]

    packed_query = pack(np.maximum(key_codes, 0), ranks, span)
    side: Literal["left", "right"] = "right" if inclusive else "left"
    found = np.asarray(np.searchsorted(packed_source, packed_query, side=side), np.int64) - 1

    matched = (found >= 0) & (key_codes >= 0)
    safe = np.where(matched, found, 0)
    # A hit is only real if it belongs to the *same* key: searchsorted will
    # otherwise fall back to the previous key's last row.
    matched &= packed_source[safe] // span == key_codes
    return np.where(matched, positions[safe], -1).astype(np.int64)
