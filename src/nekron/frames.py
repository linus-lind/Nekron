"""Vectorised primitives for moving data between panels without needless copies.

Every stage that reindexes one frame onto another — alignment, broadcasting, the
link resolver — reduces to the same three operations: read an index level as
integer codes, gather rows by position, and stitch aligned column blocks back
together. Doing each of those the obvious way is what makes a panel pipeline slow,
so they are implemented once, here, and shared.

Three measured facts shape this module:

* :meth:`pandas.MultiIndex.get_level_values` materialises the whole level on every
  call and is not cached (~780 us and ~48 MB per call on a two-million-row panel).
  :func:`level_keys` reads the index's own integer codes instead — measured ~9x
  faster and ~3x smaller, since the stored codes are usually narrower than a
  pointer — and returns them so callers can factorize a level once and reuse it
  for every frame aligned onto it.
* :meth:`pandas.DataFrame.take` treats ``-1`` as "the last row" rather than "no
  match", exactly like :meth:`numpy.ndarray.take`. Since a ``MultiIndex`` stores
  ``-1`` for a missing level value and every indexer here uses ``-1`` for an
  unmatched key, an unguarded ``take`` silently gathers the wrong row.
  :func:`take_filled` is the guard.
* :func:`pandas.concat` along the column axis copies the entire payload on pandas
  2.2 (it is free on 3.0). :func:`assemble` builds the frame from its column
  Series instead, which is zero-copy on both.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

IntpArray = npt.NDArray[np.intp]


class FrameError(Exception):
    """Base class for errors raised by the frame primitives."""


def resolve_level(index: pd.Index, level: int | str) -> int:
    """Resolve an index-level spec (position or name) to a level position."""
    names = list(index.names)
    if isinstance(level, str):
        if level not in names:
            raise KeyError(f"index has no level named {level!r}; names are {names}.")
        return names.index(level)
    nlevels = index.nlevels
    if not -nlevels <= level < nlevels:
        raise IndexError(f"level {level} is out of range for a {nlevels}-level index.")
    return level % nlevels


def level_keys(index: pd.Index, level: int | str) -> tuple[IntpArray, pd.Index]:
    """Return ``(codes, uniques)`` for one index level, without materialising it.

    ``codes[i]`` is the position of row ``i``'s level value in ``uniques``, or
    ``-1`` when that value is missing. Reading a :class:`~pandas.MultiIndex`'s
    stored codes costs nothing, so a caller aligning several frames onto the same
    panel should call this once per level and pass the result around.

    ``uniques`` may contain entries no row uses — pandas keeps a level's full
    vocabulary after rows are filtered out — so it is a lookup domain, never a
    population count.
    """
    if isinstance(index, pd.MultiIndex):
        pos = resolve_level(index, level)
        return np.asarray(index.codes[pos]).astype(np.intp, copy=False), index.levels[pos]
    resolve_level(index, level)
    codes, uniques = pd.factorize(index, sort=False, use_na_sentinel=True)
    return codes.astype(np.intp, copy=False), pd.Index(uniques, name=index.name)


def take_filled(frame: pd.DataFrame, positions: IntpArray) -> pd.DataFrame:
    """Gather rows by position, reading ``-1`` as "no match" rather than "last row".

    The returned frame carries a default positional index: every caller here
    reassigns the index it is aligning onto, and building one is pure waste.

    Columns backed by a plain NumPy integer dtype are promoted to ``float64`` when
    any position is unmatched, because NumPy has no missing integer. Declare such
    a column as a nullable dtype (``Int32``/``Int64``) to keep integer semantics
    through a partial match; ``category``, ``string`` and the nullable dtypes all
    survive unchanged.
    """
    index = pd.RangeIndex(len(positions))
    if len(positions) and bool((positions >= 0).all()):
        return frame.take(positions).set_axis(index, axis=0)
    # Built by position, not by label: ``frame[name]`` hands back a DataFrame when a
    # label repeats, and keying the dict by label would silently merge two columns
    # whose labels merely compare equal. The explicit index is what keeps a frame
    # with no columns from collapsing to zero rows — the fill path has nothing else
    # to infer the row count from, and the fast path above would have kept them.
    columns = {
        position: frame.iloc[:, position].array.take(positions, allow_fill=True)
        for position in range(frame.shape[1])
    }
    return pd.DataFrame(columns, index=index, copy=False).set_axis(frame.columns, axis=1)


def require_unique_index(index: pd.Index, what: str) -> None:
    """Raise :class:`FrameError` unless ``index`` has no duplicate entries.

    ``Index.is_unique`` builds — and caches on the index — the same hash table a
    subsequent ``get_indexer`` needs, so checking costs microseconds and is not
    wasted work. It is worth checking explicitly because the alternative is not a
    clean failure: ``reindex`` raises, but ``join`` and ``merge`` silently fan a
    panel out to more rows than it started with.
    """
    if index.is_unique:
        return
    duplicated = index[index.duplicated()].unique()
    raise FrameError(
        f"{what} must not contain duplicate keys; found {len(duplicated)}, "
        f"e.g. {list(duplicated[:5])!r}."
    )


def assemble(index: pd.Index, blocks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate aligned column blocks onto ``index`` without copying them.

    Every block must already carry ``index`` — this stitches blocks together, it
    does not align them. Passing a block that is merely the right *length* is the
    mistake this guards against, because the frame constructor would silently
    reindex it and fill the result with nulls.
    """
    columns: dict[Hashable, pd.Series] = {}
    for block in blocks:
        if not block.index.equals(index):
            raise FrameError(
                "every block must already be aligned to the target index; got a block "
                f"with {len(block)} rows against {len(index)}."
            )
        for name in block.columns:
            if name in columns:
                raise FrameError(f"duplicate column {name!r} while assembling the panel.")
            columns[name] = block[name]
    return pd.DataFrame(columns, index=index, copy=False)
