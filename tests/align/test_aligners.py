"""Tests for the four panel aligners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.align.aligners import (
    ALIGNERS,
    AsOfEntityAligner,
    BroadcastEntityAligner,
    BroadcastTimeAligner,
    ExactAligner,
)
from nekron.align.base import AlignmentError, PanelAligner, Spine
from nekron.panel import Panel

Pair = tuple[str | None, str]


def _panel_index(pairs: list[Pair]) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.to_datetime([date for date, _ in pairs]), [entity for _, entity in pairs]],
        names=["date", "entity"],
    )


def _spine(pairs: list[Pair]) -> Spine:
    frame = pd.DataFrame({"px": np.arange(len(pairs), dtype="float64")}, index=_panel_index(pairs))
    return Spine.from_panel(Panel(frame))


def _source_panel(pairs: list[Pair], data: dict[str, object]) -> Panel:
    return Panel(pd.DataFrame(data, index=_panel_index(pairs)))


def _time_panel(dates: list[str], data: dict[str, object]) -> Panel:
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return Panel(pd.DataFrame(data, index=index), date_name="date", entity_name=None)


def _entity_panel(entities: list[str], data: dict[str, object]) -> Panel:
    index = pd.Index(entities, name="entity")
    return Panel(pd.DataFrame(data, index=index), date_name=None, entity_name="entity")


def _typed_columns(n: int) -> dict[str, object]:
    """One column of each dtype whose survival through a partial match matters."""
    return {
        "cat": pd.Series([f"s{i % 2}" for i in range(n)], dtype="category"),
        "text": pd.Series([f"n{i}" for i in range(n)], dtype="string"),
        "count": pd.array(range(10, 10 + n), dtype="Int32"),
        "value": np.arange(n, dtype="float32"),
    }


# ---------------------------------------------------------------------------
# registry surface
# ---------------------------------------------------------------------------


def test_registry_names_map_to_the_aligner_classes() -> None:
    assert ALIGNERS == {
        "exact": ExactAligner,
        "asof_entity": AsOfEntityAligner,
        "broadcast_time": BroadcastTimeAligner,
        "broadcast_entity": BroadcastEntityAligner,
    }


@pytest.mark.parametrize("aligner_type", list(ALIGNERS.values()))
def test_every_aligner_satisfies_the_protocol(aligner_type: type) -> None:
    assert isinstance(aligner_type(), PanelAligner)


# ---------------------------------------------------------------------------
# ExactAligner
# ---------------------------------------------------------------------------


def test_exact_matches_keys_and_leaves_the_rest_null() -> None:
    spine = _spine([("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-05", "A")])
    source = _source_panel(
        [("2021-01-05", "A"), ("2021-01-04", "A"), ("2021-01-04", "Z")],
        {"ret": [3.0, 1.0, 9.0]},
    )

    result = ExactAligner().align(spine, source)

    assert result.index.equals(spine.index)
    assert list(result.columns) == ["ret"]
    assert result["ret"].tolist()[:1] == [1.0]
    assert pd.isna(result["ret"].iloc[1])
    assert result["ret"].iloc[2] == 3.0


def test_exact_preserves_source_dtypes_through_a_partial_match() -> None:
    spine = _spine([("2021-01-04", "A"), ("2021-01-04", "B")])
    source = _source_panel([("2021-01-04", "A")], _typed_columns(1))

    result = ExactAligner().align(spine, source)

    assert result["cat"].dtype == "category"
    assert result["text"].dtype == source.frame["text"].dtype
    assert result["count"].dtype == "Int32"
    assert result["value"].dtype == "float32"
    assert result["count"].tolist() == [10, pd.NA]
    assert pd.isna(result["value"].iloc[1])


def test_exact_never_gathers_the_last_row_for_a_missing_key() -> None:
    # A NaT date is stored as code -1 in the spine's index; an unguarded take would
    # read that as "the last row" of the source.
    spine = _spine([(None, "A"), ("2021-01-05", "A")])
    source = _source_panel([("2021-01-04", "A"), ("2021-01-05", "A")], {"ret": [1.0, 2.0]})

    result = ExactAligner().align(spine, source)

    assert pd.isna(result["ret"].iloc[0])
    assert result["ret"].iloc[1] == 2.0


def test_exact_rejects_duplicate_source_keys() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _source_panel([("2021-01-04", "A"), ("2021-01-04", "A")], {"ret": [1.0, 2.0]})

    with pytest.raises(AlignmentError, match="deduplicate"):
        ExactAligner().align(spine, source)


def test_exact_rejects_a_mismatched_entity_key_dtype() -> None:
    spine = _spine([("2021-01-04", "1")])
    frame = pd.DataFrame(
        {"ret": [1.0]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2021-01-04"]), [1]], names=["date", "entity"]
        ),
    )

    with pytest.raises(AlignmentError, match="cannot match on entity"):
        ExactAligner().align(spine, Panel(frame))


def test_exact_rejects_a_timezone_aware_source_date_level() -> None:
    spine = _spine([("2021-01-04", "A")])
    frame = pd.DataFrame(
        {"ret": [1.0]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2021-01-04"]).tz_localize("UTC"), ["A"]], names=["date", "entity"]
        ),
    )

    with pytest.raises(AlignmentError, match="cannot match on date"):
        ExactAligner().align(spine, Panel(frame))


def test_exact_rejects_a_source_with_extra_index_levels() -> None:
    # A third level makes the index unique while the joined (date, entity) pairs
    # repeat, so the uniqueness check alone would not catch the fan-out.
    spine = _spine([("2021-01-04", "A")])
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "A"], ["q1", "q2"]],
        names=["date", "entity", "period"],
    )
    source = Panel(pd.DataFrame({"ret": [1.0, 2.0]}, index=index))

    with pytest.raises(AlignmentError, match="two-level"):
        ExactAligner().align(spine, source)


def test_exact_rejects_the_wrong_grain() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _entity_panel(["A"], {"sector": ["tech"]})

    with pytest.raises(AlignmentError, match="'exact'.*cross_section"):
        ExactAligner().align(spine, source)


def test_exact_keeps_categorical_keys_matching_plain_strings() -> None:
    spine = _spine([("2021-01-04", "A"), ("2021-01-04", "B")])
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04"]), pd.Categorical(["B"])], names=["date", "entity"]
    )
    source = Panel(pd.DataFrame({"ret": [7.0]}, index=index))

    result = ExactAligner().align(spine, source)

    assert pd.isna(result["ret"].iloc[0])
    assert result["ret"].iloc[1] == 7.0


def test_exact_handles_an_empty_source() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _source_panel([], {"ret": pd.Series([], dtype="float64")})

    result = ExactAligner().align(spine, source)

    assert result.index.equals(spine.index)
    assert result["ret"].isna().all()


def test_exact_matches_a_dictionary_reference_on_random_data() -> None:
    rng = np.random.default_rng(20240521)
    dates = pd.to_datetime([f"2021-01-{day:02d}" for day in range(1, 13)])
    entities = [f"E{i}" for i in range(7)]
    spine_pairs = sorted(
        {
            (dates[rng.integers(len(dates))], entities[rng.integers(len(entities))])
            for _ in range(60)
        }
    )
    source_pairs = sorted(
        {
            (dates[rng.integers(len(dates))], entities[rng.integers(len(entities))])
            for _ in range(40)
        }
    )
    spine = Spine.from_panel(
        Panel(
            pd.DataFrame(
                {"px": np.arange(len(spine_pairs), dtype="float64")},
                index=pd.MultiIndex.from_tuples(spine_pairs, names=["date", "entity"]),
            )
        )
    )
    values = rng.normal(size=len(source_pairs))
    source = Panel(
        pd.DataFrame(
            {"ret": values},
            index=pd.MultiIndex.from_tuples(source_pairs, names=["date", "entity"]),
        )
    )

    result = ExactAligner().align(spine, source)

    lookup = dict(zip(source_pairs, values, strict=True))
    expected = [lookup.get(key, np.nan) for key in spine_pairs]
    np.testing.assert_allclose(result["ret"].to_numpy(dtype="float64"), expected)


# ---------------------------------------------------------------------------
# BroadcastTimeAligner
# ---------------------------------------------------------------------------


def test_broadcast_time_repeats_one_row_across_the_cross_section() -> None:
    spine = _spine(
        [("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-05", "A"), ("2021-01-05", "B")]
    )
    source = _time_panel(["2021-01-05", "2021-01-04"], {"mkt": [0.5, 0.1]})

    result = BroadcastTimeAligner().align(spine, source)

    assert result.index.equals(spine.index)
    assert result["mkt"].tolist() == [0.1, 0.1, 0.5, 0.5]


def test_broadcast_time_without_as_of_leaves_unmatched_dates_null() -> None:
    spine = _spine([("2021-01-04", "A"), ("2021-01-05", "A"), ("2021-01-06", "A")])
    source = _time_panel(["2021-01-04", "2021-01-06"], {"mkt": [0.1, 0.3]})

    result = BroadcastTimeAligner().align(spine, source)

    assert result["mkt"].iloc[0] == 0.1
    assert pd.isna(result["mkt"].iloc[1])
    assert result["mkt"].iloc[2] == 0.3


def test_broadcast_time_as_of_carries_a_monthly_series_onto_a_daily_spine() -> None:
    spine = _spine(
        [
            ("2021-01-15", "A"),
            ("2021-01-31", "A"),
            ("2021-02-01", "A"),
            ("2021-02-15", "B"),
        ]
    )
    source = _time_panel(["2021-01-31", "2021-02-28"], {"smb": [1.0, 2.0]})

    result = BroadcastTimeAligner(as_of=True).align(spine, source)

    # Nothing before the first monthly stamp, then that stamp until the next one.
    assert pd.isna(result["smb"].iloc[0])
    assert result["smb"].tolist()[1:] == [1.0, 1.0, 1.0]


def test_broadcast_time_as_of_nulls_a_match_older_than_the_tolerance() -> None:
    spine = _spine([("2021-02-05", "A"), ("2021-04-30", "A")])
    source = _time_panel(["2021-01-31"], {"smb": [1.0]})

    result = BroadcastTimeAligner(as_of=True, tolerance="31D").align(spine, source)

    assert result["smb"].iloc[0] == 1.0
    assert pd.isna(result["smb"].iloc[1])


def test_broadcast_time_as_of_sorts_an_unsorted_source_before_filling() -> None:
    # get_indexer(method="pad") over an unsorted index does not raise; it returns
    # later rows for earlier dates, which is look-ahead.
    spine = _spine([("2021-01-15", "A"), ("2021-02-15", "A")])
    source = _time_panel(["2021-02-28", "2021-01-31"], {"smb": [2.0, 1.0]})

    result = BroadcastTimeAligner(as_of=True).align(spine, source)

    assert pd.isna(result["smb"].iloc[0])
    assert result["smb"].iloc[1] == 1.0


def test_broadcast_time_preserves_dtypes_and_skips_missing_spine_dates() -> None:
    spine = _spine([(None, "A"), ("2021-01-04", "A")])
    source = _time_panel(["2021-01-04", "2021-01-05"], _typed_columns(2))

    result = BroadcastTimeAligner().align(spine, source)

    assert result["cat"].dtype == "category"
    assert result["count"].dtype == "Int32"
    assert result["value"].dtype == "float32"
    # The NaT date has code -1: it must not gather the source's last row.
    assert pd.isna(result["value"].iloc[0])
    assert result["count"].tolist() == [pd.NA, 10]


def test_broadcast_time_as_of_ignores_a_source_row_without_a_date() -> None:
    spine = _spine([("2021-02-15", "A")])
    source = _time_panel(["2021-01-31", None], {"smb": [1.0, 99.0]})

    result = BroadcastTimeAligner(as_of=True).align(spine, source)

    assert result["smb"].iloc[0] == 1.0


def test_broadcast_time_rejects_duplicate_dates() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _time_panel(["2021-01-04", "2021-01-04"], {"mkt": [0.1, 0.2]})

    with pytest.raises(AlignmentError, match="deduplicate"):
        BroadcastTimeAligner().align(spine, source)


def test_broadcast_time_rejects_a_mismatched_date_dtype() -> None:
    spine = _spine([("2021-01-04", "A")])
    index = pd.DatetimeIndex(pd.to_datetime(["2021-01-04"]), name="date").tz_localize("UTC")
    source = Panel(pd.DataFrame({"mkt": [0.1]}, index=index), date_name="date", entity_name=None)

    with pytest.raises(AlignmentError, match="cannot match on date"):
        BroadcastTimeAligner().align(spine, source)


def test_broadcast_time_rejects_the_wrong_grain() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _source_panel([("2021-01-04", "A")], {"ret": [1.0]})

    with pytest.raises(AlignmentError, match="'broadcast_time'.*panel"):
        BroadcastTimeAligner().align(spine, source)


def test_broadcast_time_rejects_a_tolerance_without_as_of() -> None:
    with pytest.raises(ValueError, match="as_of"):
        BroadcastTimeAligner(tolerance="31D")


def test_broadcast_time_rejects_an_unparsable_tolerance() -> None:
    with pytest.raises(ValueError, match="offset alias"):
        BroadcastTimeAligner(as_of=True, tolerance="every other tuesday")


# ---------------------------------------------------------------------------
# BroadcastEntityAligner
# ---------------------------------------------------------------------------


def test_broadcast_entity_repeats_a_static_row_across_dates() -> None:
    spine = _spine(
        [("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-05", "A"), ("2021-01-05", "B")]
    )
    source = _entity_panel(["B", "A"], {"sector": ["fin", "tech"]})

    result = BroadcastEntityAligner().align(spine, source)

    assert result.index.equals(spine.index)
    assert result["sector"].tolist() == ["tech", "fin", "tech", "fin"]


def test_broadcast_entity_leaves_unknown_entities_null_and_keeps_dtypes() -> None:
    spine = _spine([("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-05", "B")])
    source = _entity_panel(["A"], _typed_columns(1))

    result = BroadcastEntityAligner().align(spine, source)

    assert result["cat"].dtype == "category"
    assert result["text"].dtype == source.frame["text"].dtype
    assert result["count"].dtype == "Int32"
    assert result["value"].dtype == "float32"
    assert result["count"].tolist() == [10, pd.NA, pd.NA]


def test_broadcast_entity_rejects_duplicate_entities() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _entity_panel(["A", "A"], {"sector": ["tech", "fin"]})

    with pytest.raises(AlignmentError, match="deduplicate"):
        BroadcastEntityAligner().align(spine, source)


def test_broadcast_entity_rejects_a_mismatched_entity_dtype() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = Panel(
        pd.DataFrame({"sector": ["tech"]}, index=pd.Index([1], name="entity")),
        date_name=None,
        entity_name="entity",
    )

    with pytest.raises(AlignmentError, match="cannot match on entity"):
        BroadcastEntityAligner().align(spine, source)


def test_broadcast_entity_rejects_the_wrong_grain() -> None:
    spine = _spine([("2021-01-04", "A")])
    source = _time_panel(["2021-01-04"], {"mkt": [0.1]})

    with pytest.raises(AlignmentError, match="'broadcast_entity'.*time_series"):
        BroadcastEntityAligner().align(spine, source)


# ---------------------------------------------------------------------------
# AsOfEntityAligner
# ---------------------------------------------------------------------------


def _fundamentals() -> Panel:
    """Two quarters for A, one for B, stamped with period end and report date."""
    return _source_panel(
        [("2020-12-31", "A"), ("2021-03-31", "A"), ("2020-12-31", "B")],
        {
            "rdq": pd.to_datetime(["2021-02-15", "2021-05-10", "2021-02-20"]),
            "eps": [1.0, 2.0, 5.0],
        },
    )


def test_asof_entity_never_looks_ahead_of_the_availability_date() -> None:
    spine = _spine(
        [
            ("2021-02-14", "A"),
            ("2021-02-15", "A"),
            ("2021-05-11", "A"),
            ("2021-02-16", "B"),
            ("2021-02-16", "C"),
        ]
    )

    result = AsOfEntityAligner(availability_column="rdq").align(spine, _fundamentals())

    assert result.index.equals(spine.index)
    # Before A's first report date there is nothing knowable.
    assert pd.isna(result["eps"].iloc[0])
    assert result["eps"].iloc[1] == 1.0
    assert result["eps"].iloc[2] == 2.0
    # B's report is two days away, and C is not in the source at all.
    assert pd.isna(result["eps"].iloc[3])
    assert pd.isna(result["eps"].iloc[4])


def test_asof_entity_allow_exact_toggles_the_same_day_boundary() -> None:
    spine = _spine([("2021-02-15", "A")])

    assert (
        AsOfEntityAligner(availability_column="rdq").align(spine, _fundamentals())["eps"].iloc[0]
        == 1.0
    )
    strict = AsOfEntityAligner(availability_column="rdq", allow_exact=False).align(
        spine, _fundamentals()
    )
    assert pd.isna(strict["eps"].iloc[0])


def test_asof_entity_lag_shifts_the_effective_availability_date() -> None:
    spine = _spine([("2021-02-20", "A"), ("2021-03-05", "A")])

    lagged = AsOfEntityAligner(availability_column="rdq", lag="14D").align(spine, _fundamentals())

    # RDQ 2021-02-15 + 14 days == 2021-03-01.
    assert pd.isna(lagged["eps"].iloc[0])
    assert lagged["eps"].iloc[1] == 1.0


def test_asof_entity_without_an_availability_column_uses_the_date_level() -> None:
    spine = _spine([("2021-01-15", "A"), ("2021-04-01", "A")])

    result = AsOfEntityAligner(lag="45D").align(spine, _fundamentals())

    # Period end 2020-12-31 + 45 days == 2021-02-14.
    assert pd.isna(result["eps"].iloc[0])
    assert result["eps"].iloc[1] == 1.0


def test_asof_entity_tolerance_nulls_a_stale_match() -> None:
    spine = _spine([("2021-06-01", "A"), ("2022-06-01", "A")])

    result = AsOfEntityAligner(availability_column="rdq", tolerance="200D").align(
        spine, _fundamentals()
    )

    assert result["eps"].iloc[0] == 2.0
    assert pd.isna(result["eps"].iloc[1])


def test_asof_entity_ignores_rows_whose_availability_date_is_missing() -> None:
    spine = _spine([("2021-06-01", "A")])
    source = _source_panel([("2020-12-31", "A")], {"rdq": pd.to_datetime([None]), "eps": [1.0]})

    result = AsOfEntityAligner(availability_column="rdq").align(spine, source)

    assert pd.isna(result["eps"].iloc[0])


def test_asof_entity_breaks_ties_on_the_last_source_row() -> None:
    spine = _spine([("2021-03-01", "A")])
    source = _source_panel(
        [("2020-09-30", "A"), ("2020-12-31", "A")],
        {"rdq": pd.to_datetime(["2021-02-15", "2021-02-15"]), "eps": [1.0, 2.0]},
    )

    result = AsOfEntityAligner(availability_column="rdq").align(spine, source)
    reversed_source = Panel(source.frame.iloc[::-1])

    assert result["eps"].iloc[0] == 2.0
    assert (
        AsOfEntityAligner(availability_column="rdq").align(spine, reversed_source)["eps"].iloc[0]
        == 1.0
    )


def test_asof_entity_preserves_dtypes_and_skips_missing_spine_dates() -> None:
    spine = _spine([(None, "A"), ("2021-06-01", "A")])
    source = _source_panel(
        [("2020-12-31", "A"), ("2021-03-31", "A")],
        {"rdq": pd.to_datetime(["2021-02-15", "2021-05-10"]), **_typed_columns(2)},
    )

    result = AsOfEntityAligner(availability_column="rdq").align(spine, source)

    assert result["cat"].dtype == "category"
    assert result["count"].dtype == "Int32"
    assert result["value"].dtype == "float32"
    # A NaT spine date is code -1: it must not gather the source's last row.
    assert pd.isna(result["value"].iloc[0])
    assert result["count"].tolist() == [pd.NA, 11]


def test_asof_entity_rejects_a_missing_or_non_date_availability_column() -> None:
    spine = _spine([("2021-06-01", "A")])
    source = _fundamentals()

    with pytest.raises(AlignmentError, match="availability column"):
        AsOfEntityAligner(availability_column="filed").align(spine, source)
    with pytest.raises(AlignmentError, match="datetime availability date"):
        AsOfEntityAligner(availability_column="eps").align(spine, source)


def test_asof_entity_rejects_duplicate_source_keys() -> None:
    spine = _spine([("2021-06-01", "A")])
    source = _source_panel(
        [("2020-12-31", "A"), ("2020-12-31", "A")],
        {"rdq": pd.to_datetime(["2021-02-15", "2021-02-16"]), "eps": [1.0, 2.0]},
    )

    with pytest.raises(AlignmentError, match="deduplicate"):
        AsOfEntityAligner(availability_column="rdq").align(spine, source)


def test_asof_entity_rejects_a_mismatched_entity_dtype() -> None:
    spine = _spine([("2021-06-01", "1")])
    frame = pd.DataFrame(
        {"rdq": pd.to_datetime(["2021-02-15"]), "eps": [1.0]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2020-12-31"]), [1]], names=["date", "entity"]
        ),
    )

    with pytest.raises(AlignmentError, match="cannot match on entity"):
        AsOfEntityAligner(availability_column="rdq").align(spine, Panel(frame))


def test_asof_entity_rejects_the_wrong_grain() -> None:
    spine = _spine([("2021-06-01", "A")])

    with pytest.raises(AlignmentError, match="'asof_entity'.*time_series"):
        AsOfEntityAligner().align(spine, _time_panel(["2021-01-04"], {"mkt": [0.1]}))


def test_asof_entity_rejects_an_unparsable_lag() -> None:
    with pytest.raises(ValueError, match="offset alias"):
        AsOfEntityAligner(lag="a month or so")


def test_asof_entity_handles_an_empty_source() -> None:
    spine = _spine([("2021-06-01", "A")])
    source = Panel(
        pd.DataFrame(
            {"rdq": pd.Series([], dtype="datetime64[ns]"), "eps": pd.Series([], dtype="float64")},
            index=_panel_index([]),
        )
    )

    result = AsOfEntityAligner(availability_column="rdq").align(spine, source)

    assert result.index.equals(spine.index)
    assert result["eps"].isna().all()


def _reference_asof(
    spine: Spine,
    source: pd.DataFrame,
    availability: pd.Series,
    *,
    allow_exact: bool,
    tolerance: pd.Timedelta | None,
) -> list[float]:
    """An obviously correct point-in-time join: for each row, scan every source row."""
    entities = list(source.index.get_level_values("entity"))
    out: list[float] = []
    for date, entity in spine.index:
        best: int | None = None
        for position in range(len(source)):
            available = availability.iloc[position]
            if entities[position] != entity or pd.isna(available) or pd.isna(date):
                continue
            if available > date or (not allow_exact and available == date):
                continue
            if tolerance is not None and date - available > tolerance:
                continue
            if best is None or available >= availability.iloc[best]:
                best = position
        out.append(np.nan if best is None else float(source["eps"].iloc[best]))
    return out


@pytest.mark.parametrize("allow_exact", [True, False])
@pytest.mark.parametrize("tolerance", [None, "90D"])
def test_asof_entity_matches_a_loop_reference_on_random_data(
    allow_exact: bool, tolerance: str | None
) -> None:
    rng = np.random.default_rng(7331)
    entities = [f"E{i}" for i in range(6)]
    spine_dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 400, size=200), unit="D"
    )
    spine_pairs = sorted({(date, entities[rng.integers(len(entities))]) for date in spine_dates})
    spine = Spine.from_panel(
        Panel(
            pd.DataFrame(
                {"px": np.arange(len(spine_pairs), dtype="float64")},
                index=pd.MultiIndex.from_tuples(spine_pairs, names=["date", "entity"]),
            )
        )
    )

    stamps = pd.to_datetime("2020-10-01") + pd.to_timedelta(rng.integers(0, 400, size=50), unit="D")
    source_pairs = sorted({(stamp, entities[rng.integers(len(entities))]) for stamp in stamps})
    rdq = pd.to_datetime([stamp for stamp, _ in source_pairs]) + pd.to_timedelta(
        rng.integers(10, 90, size=len(source_pairs)), unit="D"
    )
    frame = pd.DataFrame(
        {"rdq": rdq, "eps": rng.normal(size=len(source_pairs))},
        index=pd.MultiIndex.from_tuples(source_pairs, names=["date", "entity"]),
    )
    # A handful of rows whose report date is unknown must never be matched.
    frame.iloc[[1, 4], frame.columns.get_loc("rdq")] = pd.NaT

    result = AsOfEntityAligner(
        availability_column="rdq", tolerance=tolerance, allow_exact=allow_exact
    ).align(spine, Panel(frame))

    expected = _reference_asof(
        spine,
        frame,
        frame["rdq"],
        allow_exact=allow_exact,
        tolerance=None if tolerance is None else pd.Timedelta(tolerance),
    )
    np.testing.assert_allclose(
        result["eps"].to_numpy(dtype="float64"), np.asarray(expected, dtype="float64")
    )
    # Both outcomes are exercised: some rows match, some have nothing knowable yet.
    assert 0.0 < float(result["eps"].isna().mean()) < 1.0


def _spine_of(dates: list[str | None], entities: list[str]) -> Spine:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates * len(entities)), sorted(entities * len(dates))],
        names=["date", "entity"],
    )
    return Spine.from_panel(
        Panel(pd.DataFrame({"px": np.arange(len(index), dtype=float)}, index=index))
    )


def test_asof_on_a_spine_whose_dates_are_all_missing_yields_nulls() -> None:
    """A NaT spine date is knowable as of nothing — that is a null, not a crash."""
    spine = _spine_of([None], ["A"])
    source = Panel(
        pd.DataFrame(
            {"v": [7.0]},
            index=pd.MultiIndex.from_arrays(
                [pd.to_datetime(["2020-01-01"]), ["A"]], names=["date", "entity"]
            ),
        )
    )

    block = AsOfEntityAligner().align(spine, source)

    assert block.index.equals(spine.index)
    assert block["v"].isna().all()


def test_exact_with_an_all_null_source_key_yields_nulls() -> None:
    """A categorical level keeps its vocabulary after its rows are nulled."""
    spine = _spine_of(["2021-01-04"], ["A"])
    source = Panel(
        pd.DataFrame(
            {"v": [1.0, 2.0]},
            index=pd.MultiIndex.from_arrays(
                # Distinct dates keep the composite keys unique, so the source
                # passes the duplicate-key guard and reaches the lookup with every
                # entity code at -1.
                [
                    pd.to_datetime(["2021-01-04", "2021-01-05"]),
                    pd.Categorical([None, None], categories=["A", "B"]),
                ],
                names=["date", "entity"],
            ),
        )
    )

    block = ExactAligner().align(spine, source)

    assert block.index.equals(spine.index)
    assert block["v"].isna().all()


def test_broadcast_time_ignores_undated_source_rows() -> None:
    """Blank date cells in a factor file cannot match anything, so they are not duplicates."""
    spine = _spine_of(["2021-01-04"], ["A"])
    source = Panel(
        pd.DataFrame(
            {"v": [1.0, 2.0, 3.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2021-01-01", None, None]), name="date"),
        ),
        date_name="date",
        entity_name=None,
    )

    assert BroadcastTimeAligner(as_of=True).align(spine, source)["v"].tolist() == [1.0]
    assert BroadcastTimeAligner().align(spine, source)["v"].isna().all()
