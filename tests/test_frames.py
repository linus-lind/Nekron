"""Tests for the vectorised frame primitives in :mod:`nekron.frames`.

Every one of these primitives replaces an obvious pandas call that is either slow
or silently wrong, so the tests are written to pin the *difference*: what the
naive call would have returned, and what the primitive returns instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.align.keys import pack
from nekron.frames import (
    FrameError,
    IntpArray,
    assemble,
    level_keys,
    require_unique_index,
    resolve_level,
    take_filled,
)


def make_index(dates: list[str | None], entities: list[str]) -> pd.MultiIndex:
    """Build a ``(date, entity)`` MultiIndex, where ``None`` means a missing date."""
    return pd.MultiIndex.from_arrays(
        [pd.to_datetime(pd.Series(dates)), entities], names=["date", "entity"]
    )


def positions(*values: int) -> IntpArray:
    """A positional indexer, typed the way the primitives expect it."""
    return np.asarray(values, dtype=np.intp)


def assert_level_reconstructs(index: pd.Index, level: int | str) -> None:
    """Assert ``level_keys`` reproduces the level a naive ``get_level_values`` reads.

    Codes of ``-1`` are checked separately rather than gathered: ``Index.take``
    has the very ``-1``-is-the-last-row behaviour this module exists to guard
    against, so using it here would hide the thing being asserted.
    """
    codes, uniques = level_keys(index, level)
    values = index.get_level_values(level) if isinstance(index, pd.MultiIndex) else index
    present = codes >= 0
    pd.testing.assert_index_equal(uniques.take(codes[present]), values[present])
    assert bool(values[~present].isna().all())


# --------------------------------------------------------------------------- #
# resolve_level
# --------------------------------------------------------------------------- #


def test_resolve_level_accepts_names_and_positions() -> None:
    index = make_index(["2021-01-04", "2021-01-05"], ["A", "B"])
    assert resolve_level(index, "date") == 0
    assert resolve_level(index, "entity") == 1
    assert resolve_level(index, 0) == 0
    assert resolve_level(index, -1) == 1  # negative positions wrap, like numpy


def test_resolve_level_rejects_unknown_name_and_out_of_range_position() -> None:
    index = make_index(["2021-01-04"], ["A"])
    with pytest.raises(KeyError, match="no level named 'sector'"):
        resolve_level(index, "sector")
    with pytest.raises(IndexError, match="out of range"):
        resolve_level(index, 2)


def test_resolve_level_on_a_flat_index() -> None:
    flat = pd.Index(["A", "B"], name="entity")
    assert resolve_level(flat, "entity") == 0
    assert resolve_level(flat, 0) == 0
    with pytest.raises(KeyError):
        resolve_level(flat, "date")


# --------------------------------------------------------------------------- #
# level_keys
# --------------------------------------------------------------------------- #


def test_level_keys_reconstructs_the_level_exactly() -> None:
    # level_keys reads the index's stored codes, so its `uniques` are the level's
    # sorted vocabulary rather than factorize's first-appearance order. The codes
    # are therefore not comparable to a naive factorize position-by-position; what
    # must agree exactly is the value each code resolves to.
    index = make_index(["2021-01-05", "2021-01-04", "2021-01-05"], ["A", "B", "A"])
    codes, uniques = level_keys(index, "date")
    assert codes.tolist() == [1, 0, 1]  # sorted vocabulary, not first-appearance order
    assert pd.factorize(index.get_level_values("date"), sort=False)[0].tolist() == [0, 1, 0]
    assert_level_reconstructs(index, "date")
    assert_level_reconstructs(index, "entity")


def test_level_keys_marks_a_missing_level_value_with_minus_one() -> None:
    index = make_index(["2021-01-05", "2021-01-04", None], ["A", "B", "A"])
    codes, uniques = level_keys(index, "date")

    assert codes.tolist() == [1, 0, -1]  # sorted vocabulary: 01-04 -> 0, 01-05 -> 1
    assert list(uniques) == list(pd.to_datetime(["2021-01-04", "2021-01-05"]))
    # NaT is absent from the vocabulary, exactly as a naive factorize would have it.
    naive_codes, _ = pd.factorize(index.get_level_values("date"), use_na_sentinel=True)
    assert (codes == -1).tolist() == (naive_codes == -1).tolist()
    assert_level_reconstructs(index, "date")


def test_level_keys_uniques_keep_entries_no_row_uses() -> None:
    # pandas keeps a level's full vocabulary after rows are filtered out, so
    # `uniques` is a lookup domain and never a population count.
    index = make_index(["2021-01-04", "2021-01-05", "2021-01-06"], ["A", "B", "C"])
    filtered = index[[1]]
    codes, uniques = level_keys(filtered, "date")

    assert len(uniques) == 3  # all three dates survive as unused vocabulary
    assert codes.tolist() == [1]
    assert len(pd.factorize(filtered.get_level_values("date"))[1]) == 1  # the naive count
    assert_level_reconstructs(filtered, "date")


def test_level_keys_on_a_flat_index_matches_factorize() -> None:
    flat = pd.Index(["b", "a", "b", None], name="entity")
    codes, uniques = level_keys(flat, 0)
    naive_codes, naive_uniques = pd.factorize(flat, sort=False, use_na_sentinel=True)

    assert codes.tolist() == naive_codes.tolist() == [0, 1, 0, -1]
    assert list(uniques) == list(naive_uniques)
    assert uniques.name == "entity"  # the name is carried, factorize drops it
    assert codes.dtype == np.intp

    by_name, _ = level_keys(flat, "entity")
    assert by_name.tolist() == codes.tolist()
    assert_level_reconstructs(flat, 0)


def test_level_keys_returns_intp_codes_for_a_multiindex() -> None:
    index = make_index(["2021-01-04", "2021-01-05"], ["A", "B"])
    codes, _ = level_keys(index, 0)
    assert codes.dtype == np.intp  # MultiIndex stores int8 for a small level


# --------------------------------------------------------------------------- #
# take_filled
# --------------------------------------------------------------------------- #


def sample_frame() -> pd.DataFrame:
    """One row per dtype family the pipeline actually moves through a gather."""
    return pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "b"], categories=["a", "b", "c"]),
            "text": pd.array(["p", "q"], dtype="string"),
            "nullable": pd.array([1, 2], dtype="Int32"),
            "f32": np.array([1.0, 2.0], dtype=np.float32),
            "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "flag": np.array([True, False]),
            "i64": np.array([7, 8], dtype=np.int64),
        }
    )


def test_plain_take_of_minus_one_returns_the_last_row() -> None:
    # The trap take_filled exists to close: -1 is a valid numpy position.
    frame = sample_frame()
    gathered = frame.take([-1])
    assert gathered["i64"].tolist() == [8]
    assert not gathered["i64"].isna().any()


def test_take_filled_reads_minus_one_as_a_missing_row() -> None:
    frame = sample_frame()
    out = take_filled(frame, positions(1, -1))

    assert out.index.tolist() == [0, 1]  # a fresh positional index, as documented
    assert out["cat"].tolist()[0] == "b"
    assert out.iloc[1].isna().all()


def test_take_filled_preserves_dtypes_across_a_fill() -> None:
    frame = sample_frame()
    out = take_filled(frame, positions(1, -1))

    assert isinstance(out["cat"].dtype, pd.CategoricalDtype)
    assert list(out["cat"].dtype.categories) == ["a", "b", "c"]  # unused category kept
    assert out["text"].dtype == frame["text"].dtype
    assert out["nullable"].dtype == frame["nullable"].dtype
    assert out["f32"].dtype == np.float32
    assert out["when"].dtype == frame["when"].dtype
    assert out["when"].isna().tolist() == [False, True]  # NaT, not a wrapped date


def test_take_filled_promotes_numpy_integers_and_bools_on_a_fill() -> None:
    # NumPy has no missing integer and no missing bool, so a partial match
    # degrades both. Declare such a column nullable (Int32/boolean) to keep its
    # semantics; this test pins the degradation so it cannot change unnoticed.
    frame = sample_frame()
    out = take_filled(frame, positions(0, -1))

    assert frame["i64"].dtype == np.int64
    assert out["i64"].dtype == np.float64
    assert out["i64"].tolist() == [7.0, pytest.approx(np.nan, nan_ok=True)]
    assert frame["flag"].dtype == np.bool_
    assert out["flag"].dtype == object


def test_take_filled_without_any_fill_keeps_every_dtype() -> None:
    frame = sample_frame()
    out = take_filled(frame, positions(1, 0))

    assert out.index.tolist() == [0, 1]
    pd.testing.assert_series_equal(
        out["i64"], pd.Series([8, 7], dtype=np.int64, name="i64"), check_exact=True
    )
    assert out["flag"].dtype == np.bool_
    assert out["cat"].tolist() == ["b", "a"]


def test_take_filled_of_no_positions_keeps_the_schema() -> None:
    frame = sample_frame()
    out = take_filled(frame, positions())

    assert len(out) == 0
    assert list(out.columns) == list(frame.columns)
    assert out["i64"].dtype == np.int64  # nothing was filled, so nothing promoted


def test_take_filled_repeats_a_row_when_asked_twice() -> None:
    frame = sample_frame()
    out = take_filled(frame, positions(0, 0, -1, 0))
    assert out["text"].tolist()[:2] == ["p", "p"]
    assert out["text"].isna().tolist() == [False, False, True, False]


# --------------------------------------------------------------------------- #
# require_unique_index
# --------------------------------------------------------------------------- #


def test_require_unique_index_passes_a_unique_index() -> None:
    index = make_index(["2021-01-04", "2021-01-04"], ["A", "B"])
    require_unique_index(index, "the spine")  # must not raise


def test_require_unique_index_names_the_offending_keys() -> None:
    index = make_index(["2021-01-04", "2021-01-04", "2021-01-05"], ["A", "A", "B"])
    with pytest.raises(FrameError) as excinfo:
        require_unique_index(index, "the sector map")

    message = str(excinfo.value)
    assert "the sector map" in message
    assert "found 1" in message
    assert "'A'" in message  # the duplicated key itself is quoted back


def test_require_unique_index_counts_distinct_duplicates() -> None:
    index = pd.Index(["a", "a", "b", "b", "b", "c"], name="entity")
    with pytest.raises(FrameError, match="found 2"):
        require_unique_index(index, "the link table")


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #


def test_assemble_matches_concat_for_aligned_blocks() -> None:
    index = make_index(["2021-01-04", "2021-01-04"], ["A", "B"])
    left = pd.DataFrame({"price": [1.0, 2.0]}, index=index)
    right = pd.DataFrame(
        {"sector": pd.Categorical(["tech", "energy"]), "size": pd.array([3, 4], dtype="Int32")},
        index=index,
    )

    out = assemble(index, [left, right])
    pd.testing.assert_frame_equal(out, pd.concat([left, right], axis=1))
    assert list(out.columns) == ["price", "sector", "size"]
    assert isinstance(out["sector"].dtype, pd.CategoricalDtype)


def test_assemble_rejects_a_block_that_is_merely_the_right_length() -> None:
    # The DataFrame constructor would have reindexed this block and filled the
    # column with nulls, which is the failure mode worth an exception.
    index = make_index(["2021-01-04", "2021-01-04"], ["A", "B"])
    misaligned = pd.DataFrame(
        {"price": [1.0, 2.0]}, index=make_index(["2021-01-05", "2021-01-05"], ["A", "B"])
    )

    with pytest.raises(FrameError, match="already be aligned"):
        assemble(index, [misaligned])


def test_assemble_rejects_a_block_of_the_wrong_length() -> None:
    index = make_index(["2021-01-04", "2021-01-04"], ["A", "B"])
    short = pd.DataFrame({"price": [1.0]}, index=index[:1])
    with pytest.raises(FrameError, match="1 rows against 2"):
        assemble(index, [short])


def test_assemble_rejects_duplicate_column_names() -> None:
    index = make_index(["2021-01-04"], ["A"])
    left = pd.DataFrame({"price": [1.0]}, index=index)
    right = pd.DataFrame({"price": [2.0]}, index=index)

    with pytest.raises(FrameError, match="duplicate column 'price'"):
        assemble(index, [left, right])


def test_assemble_of_no_blocks_keeps_the_index() -> None:
    index = make_index(["2021-01-04", "2021-01-05"], ["A", "B"])
    out = assemble(index, [])
    assert out.shape == (2, 0)
    pd.testing.assert_index_equal(out.index, index)


def test_assemble_does_not_copy_the_column_payload() -> None:
    index = make_index(["2021-01-04", "2021-01-05"], ["A", "B"])
    block = pd.DataFrame({"price": np.array([1.0, 2.0])}, index=index)
    out = assemble(index, [block])
    assert np.shares_memory(out["price"].to_numpy(), block["price"].to_numpy())


# --------------------------------------------------------------------------- #
# pack (nekron.align.keys)
# --------------------------------------------------------------------------- #


def test_pack_orders_by_key_then_rank() -> None:
    codes = np.array([0, 0, 1, 1, 2], dtype=np.intp)
    ranks = np.array([0, 3, 0, 2, 1], dtype=np.int64)
    packed = pack(codes, ranks, span=4)

    assert packed.tolist() == [0, 3, 4, 6, 9]
    assert np.all(np.diff(packed) > 0)  # monotone in (code, rank) order


def test_pack_round_trips_the_pair() -> None:
    span = 7
    codes = np.array([0, 1, 5, 5], dtype=np.intp)
    ranks = np.array([6, 0, 3, 6], dtype=np.int64)
    packed = pack(codes, ranks, span=span)

    assert (packed // span).tolist() == codes.tolist()
    assert (packed % span).tolist() == ranks.tolist()
    assert packed.dtype == np.int64


def test_pack_never_lets_a_rank_leak_into_the_next_key() -> None:
    # The whole point of `span`: the largest rank of key k must stay below the
    # smallest packed value of key k + 1.
    span = 5
    last_of_first = pack(np.array([0], dtype=np.intp), np.array([span - 1], np.int64), span)
    first_of_second = pack(np.array([1], dtype=np.intp), np.array([0], np.int64), span)
    assert last_of_first[0] < first_of_second[0]


def test_take_filled_keeps_the_rows_of_a_frame_with_no_columns() -> None:
    """The fill path must not infer the row count from the columns it does not have.

    Data-dependent, and therefore the worst kind: the fast path returns the right
    number of rows, so a column-less block only collapses once some key fails to
    match.
    """
    empty = pd.DataFrame(index=pd.RangeIndex(3))

    assert len(take_filled(empty, np.array([0, 1, 2], dtype=np.intp))) == 3
    assert len(take_filled(empty, np.array([0, -1, 2], dtype=np.intp))) == 3


def test_take_filled_handles_duplicate_column_labels() -> None:
    """Selecting by label would hand back a frame, or silently merge two columns."""
    frame = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=["x", "x"])

    filled = take_filled(frame, np.array([0, -1], dtype=np.intp))

    assert filled.shape == (2, 2)
    assert list(filled.columns) == ["x", "x"]
    assert filled.iloc[0].tolist() == [0.0, 1.0]
    assert filled.iloc[1].isna().all()


def test_assemble_preserves_non_string_column_labels() -> None:
    """A label is a key, not a display string; stringifying it breaks lookups."""
    index = pd.Index([1, 2, 3], name="entity")
    block = pd.DataFrame({1: [1.0, 2.0, 3.0], "1": [4.0, 5.0, 6.0]}, index=index)

    out = assemble(index, [block])

    assert list(out.columns) == [1, "1"]
    assert out[1].tolist() == [1.0, 2.0, 3.0]
