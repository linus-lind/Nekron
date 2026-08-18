"""Tests for identifier linking: :class:`LinkTable`, :func:`relink`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nekron.align.linking import (
    KeyNormalization,
    LinkError,
    LinkReport,
    LinkTable,
    relink,
)
from nekron.panel import Panel


def dates(*values: str) -> np.ndarray:
    """The given days as a plain datetime64 array."""
    return as_dates(values).to_numpy()


def as_dates(values: Sequence[str | None]) -> pd.Series:
    """Parse dates at microsecond resolution.

    ``pd.to_datetime`` defaults to nanoseconds on pandas 2.2 and refuses the
    far-future sentinels link files use, so the tests parse the way the module
    does.
    """
    return pd.Series(list(values), dtype=object).astype("datetime64[us]")


def make_panel(
    rows: Sequence[tuple[str, Any]],
    columns: dict[str, Sequence[Any]] | None = None,
    *,
    entity_name: str = "gvkey",
) -> Panel:
    """A ``(date, entity)`` panel from explicit ``(date, entity)`` pairs."""
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime([date for date, _ in rows]), [entity for _, entity in rows]],
        names=["date", entity_name],
    )
    payload = columns or {"x": np.arange(len(rows), dtype=float)}
    return Panel(frame=pd.DataFrame(payload, index=index), entity_name=entity_name)


def windowed_link(
    keys: Sequence[str],
    targets: Sequence[Any],
    starts: Sequence[str | None],
    ends: Sequence[str | None],
    **kwargs: Any,
) -> LinkTable:
    """A single-column, time-variant link table."""
    table = pd.DataFrame(
        {
            "gvkey": list(keys),
            "permno": list(targets),
            "valid_from": as_dates(starts),
            "valid_to": as_dates(ends),
        }
    )
    return LinkTable(
        table=table,
        source_keys=("gvkey",),
        target="permno",
        valid_from="valid_from",
        valid_to="valid_to",
        **kwargs,
    )


def resolve_frame(link: LinkTable, keys: Sequence[Any], when: np.ndarray | None) -> np.ndarray:
    """Resolve a bare list of keys, one row each."""
    values, _ = link.resolve(pd.DataFrame({"gvkey": list(keys)}), dates=when)
    return values


# -- 1. time-variant windows ---------------------------------------------------


def test_disjoint_windows_resolve_per_date() -> None:
    link = windowed_link(
        ["A", "A", "B"],
        [10, 11, 20],
        ["2000-01-01", "2005-01-01", "2000-01-01"],
        ["2004-12-31", "2009-12-31", "2009-12-31"],
    )
    when = dates("1999-06-01", "2003-06-01", "2007-06-01", "2011-06-01")
    resolved = resolve_frame(link, ["A"] * 4, when)
    assert [None if pd.isna(v) else v for v in resolved] == [None, 10, 11, None]


def test_window_bounds_are_inclusive_on_both_ends() -> None:
    link = windowed_link(["A"], [10], ["2000-01-01"], ["2000-12-31"])
    when = dates("1999-12-31", "2000-01-01", "2000-12-31", "2001-01-01")
    resolved = resolve_frame(link, ["A"] * 4, when)
    assert [None if pd.isna(v) else v for v in resolved] == [None, 10, 10, None]


def test_open_ended_window_stays_valid_forever() -> None:
    link = windowed_link(["A", "A"], [10, 11], ["2000-01-01", "2005-01-01"], ["2004-12-31", None])
    resolved = resolve_frame(link, ["A", "A"], dates("2006-01-01", "2200-01-01"))
    assert list(resolved) == [11, 11]


def test_far_future_sentinel_is_not_wrapped_by_a_nanosecond_cast() -> None:
    # 9999-12-31 is the conventional "still valid" marker; casting it to ns
    # silently wraps it to 1816, which would close the window in the past.
    link = windowed_link(["A"], [10], ["2000-01-01"], ["9999-12-31"])
    assert list(resolve_frame(link, ["A"], dates("2500-01-01"))) == [10]


def test_missing_query_date_resolves_to_nothing() -> None:
    link = windowed_link(["A"], [10], ["2000-01-01"], [None])
    resolved = resolve_frame(link, ["A", "A"], dates("2001-01-01", "NaT"))
    assert resolved[0] == 10 and pd.isna(resolved[1])


def test_time_variant_table_without_dates_is_refused() -> None:
    link = windowed_link(["A"], [10], ["2000-01-01"], [None])
    with pytest.raises(LinkError, match="time-variant"):
        link.resolve(pd.DataFrame({"gvkey": ["A"]}))


def test_effective_date_only_table_is_a_step_function() -> None:
    table = pd.DataFrame(
        {
            "gvkey": ["A", "A"],
            "permno": [10, 11],
            "start": pd.to_datetime(["2000-01-01", "2005-01-01"]),
        }
    )
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno", valid_from="start")
    resolved = resolve_frame(link, ["A"] * 3, dates("1999-01-01", "2001-01-01", "2030-01-01"))
    assert [None if pd.isna(v) else v for v in resolved] == [None, 10, 11]


# -- 2. overlap validation -----------------------------------------------------


def test_overlapping_windows_agreeing_on_target_still_resolve() -> None:
    # The merge_asof trap: the short window starts later, so an as-of search finds
    # it and its end then excludes 2007 -- even though the long window covers it.
    link = windowed_link(
        ["A", "A"], [10, 10], ["2000-01-01", "2003-01-01"], ["2010-01-01", "2005-01-01"]
    )
    assert list(resolve_frame(link, ["A"], dates("2007-01-01"))) == [10]
    assert len(link.table) == 1  # redundant window absorbed, not rejected


def test_conflicting_overlap_is_refused_and_names_the_key() -> None:
    with pytest.raises(LinkError, match=r"overlapping windows.*'B'"):
        windowed_link(
            ["A", "B", "B"],
            [10, 20, 21],
            ["2000-01-01", "2000-01-01", "2003-01-01"],
            ["2010-01-01", "2010-01-01", "2005-01-01"],
        )


def test_overlap_policies_split_the_timeline() -> None:
    keys, targets = ["A", "A"], [10, 11]
    starts, ends = ["2000-01-01", "2003-01-01"], ["2010-01-01", "2005-01-01"]
    when = dates("2001-01-01", "2004-01-01", "2007-01-01")

    first = windowed_link(keys, targets, starts, ends, on_ambiguous="first")
    assert list(resolve_frame(first, ["A"] * 3, when)) == [10, 10, 10]

    last = windowed_link(keys, targets, starts, ends, on_ambiguous="last")
    assert list(resolve_frame(last, ["A"] * 3, when)) == [10, 11, 10]
    # The middle segment became its own row; the flanking ones stay with row 0.
    assert len(last.table) == 3
    assert last.table["valid_from"].is_monotonic_increasing


def test_canonicalized_windows_are_disjoint() -> None:
    link = windowed_link(
        ["A", "A", "A"],
        [10, 11, 12],
        ["2000-01-01", "2002-01-01", "2004-01-01"],
        ["2006-01-01", "2008-01-01", "2010-01-01"],
        on_ambiguous="last",
    )
    table = link.table.sort_values("valid_from")
    assert (table["valid_from"].to_numpy()[1:] > table["valid_to"].to_numpy()[:-1]).all()


def test_unrelated_keys_are_not_flagged_by_the_running_maximum() -> None:
    # B's window sits inside A's; packing keeps the groups apart, so only a real
    # per-key overlap is detected.
    link = windowed_link(
        ["A", "B"], [10, 20], ["2000-01-01", "2003-01-01"], ["2010-01-01", "2005-01-01"]
    )
    assert len(link.table) == 2


def test_backwards_window_is_refused() -> None:
    with pytest.raises(LinkError, match="end before they start"):
        windowed_link(["A"], [10], ["2005-01-01"], ["2000-01-01"])


# -- 3. null keys --------------------------------------------------------------


def test_null_key_never_matches_even_another_null() -> None:
    table = pd.DataFrame({"gvkey": ["A", None], "permno": [10, 99]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    assert len(link.table) == 1  # the null-keyed link row is dropped outright
    resolved = resolve_frame(link, ["A", None], None)
    assert resolved[0] == 10 and pd.isna(resolved[1])


def test_null_target_row_is_dropped() -> None:
    table = pd.DataFrame({"gvkey": ["A", "B"], "permno": [10, None]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    resolved = resolve_frame(link, ["A", "B"], None)
    assert resolved[0] == 10 and pd.isna(resolved[1])


def test_partially_null_composite_key_never_matches() -> None:
    table = pd.DataFrame({"gvkey": ["A"], "iid": ["01"], "permno": [10]})
    link = LinkTable(table=table, source_keys=("gvkey", "iid"), target="permno")
    query = pd.DataFrame({"gvkey": ["A", "A"], "iid": ["01", None]})
    values, report = link.resolve(query)
    assert values[0] == 10 and pd.isna(values[1])
    assert report.n_rows == 2 and report.n_unmatched == 1 and report.n_keys == 1


# -- 4. composite keys ---------------------------------------------------------


def test_composite_key_of_mixed_dtypes() -> None:
    table = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "exchange": pd.Categorical(["N", "Q", "N"]),
            "series": [1, 2, 1],
            "permno": [10, 11, 20],
        }
    )
    link = LinkTable(table=table, source_keys=("ticker", "exchange", "series"), target="permno")
    query = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "exchange": pd.Categorical(["N", "Q", "N", "N"]),
            "series": [1, 2, 1, 2],
        }
    )
    values, _ = link.resolve(query)
    assert [None if pd.isna(v) else v for v in values] == [10, 11, 20, None]


def test_key_dtypes_that_could_never_match_are_refused() -> None:
    table = pd.DataFrame({"gvkey": [1001, 1002], "permno": [10, 20]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    with pytest.raises(LinkError, match="can never match"):
        link.resolve(pd.DataFrame({"gvkey": ["1001", "1002"]}))


def test_column_wins_over_index_level_of_the_same_name() -> None:
    table = pd.DataFrame({"gvkey": ["A"], "permno": [10]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    frame = pd.DataFrame(
        {"gvkey": ["A"]},
        index=pd.Index(["Z"], name="gvkey"),
    )
    assert list(link.resolve(frame)[0]) == [10]


def test_index_level_keys_are_resolved() -> None:
    table = pd.DataFrame({"gvkey": ["A", "B"], "permno": [10, 20]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    panel = make_panel([("2020-01-01", "A"), ("2020-01-01", "B")])
    assert list(link.resolve(panel.frame)[0]) == [10, 20]


# -- 5. time-invariant fast path ----------------------------------------------


def test_time_invariant_needs_no_dates_and_matches_the_windowed_answer() -> None:
    keys = [f"K{n:03d}" for n in range(50)]
    targets = list(range(50))
    invariant = LinkTable(
        table=pd.DataFrame({"gvkey": keys, "permno": targets}),
        source_keys=("gvkey",),
        target="permno",
    )
    assert not invariant.is_time_variant
    windowed = windowed_link(keys, targets, ["1900-01-01"] * 50, [None] * 50)
    query = [keys[n % 50] for n in range(500)] + ["missing"]
    when = pd.to_datetime(["2020-01-01"] * 501).to_numpy()
    assert np.array_equal(
        resolve_frame(invariant, query, None).astype(float),
        resolve_frame(windowed, query, when).astype(float),
        equal_nan=True,
    )


def test_time_invariant_path_is_faster_than_an_infinite_window() -> None:
    keys = [f"K{n:04d}" for n in range(2000)]
    targets = list(range(2000))
    invariant = LinkTable(
        table=pd.DataFrame({"gvkey": keys, "permno": targets}),
        source_keys=("gvkey",),
        target="permno",
    )
    windowed = windowed_link(keys, targets, ["1900-01-01"] * 2000, [None] * 2000)
    rng = np.random.default_rng(7)
    query = pd.DataFrame({"gvkey": rng.choice(keys, size=200_000)})
    when = pd.to_datetime(
        rng.choice(pd.date_range("2000-01-01", periods=500, freq="D"), size=200_000)
    ).to_numpy()

    def timed(link: LinkTable, with_dates: np.ndarray | None) -> float:
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            link.resolve(query, dates=with_dates)
            best = min(best, time.perf_counter() - start)
        return best

    assert timed(invariant, None) < timed(windowed, when)


def test_time_invariant_conflicting_targets_are_refused() -> None:
    table = pd.DataFrame({"gvkey": ["A", "A", "B"], "permno": [10, 11, 20]})
    with pytest.raises(LinkError, match=r"more than one 'permno'.*'A'"):
        LinkTable(table=table, source_keys=("gvkey",), target="permno")


@pytest.mark.parametrize(("policy", "expected"), [("first", 10), ("last", 11)])
def test_time_invariant_policies_pick_one_target(policy: str, expected: int) -> None:
    table = pd.DataFrame({"gvkey": ["A", "A", "B"], "permno": [10, 11, 20]})
    link = LinkTable(
        table=table,
        source_keys=("gvkey",),
        target="permno",
        on_ambiguous=policy,  # type: ignore[arg-type]
    )
    assert len(link.table) == 2
    assert list(resolve_frame(link, ["A", "B"], None)) == [expected, 20]


def test_time_invariant_duplicate_rows_collapse_to_one() -> None:
    table = pd.DataFrame({"gvkey": ["A", "A"], "permno": [10, 10]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    assert len(link.table) == 1


# -- 6. many-to-one collapse ---------------------------------------------------


def collapsing_panel() -> tuple[Panel, LinkTable]:
    """Two share classes of one firm, mapping onto a single permno."""
    panel = make_panel(
        [("2020-01-01", "A"), ("2020-01-01", "B"), ("2020-01-02", "A")],
        {"x": [1.0, 3.0, 5.0]},
    )
    link = LinkTable(
        table=pd.DataFrame({"gvkey": ["A", "B"], "permno": [10, 10]}),
        source_keys=("gvkey",),
        target="permno",
    )
    return panel, link


def test_collision_error_names_the_shared_key() -> None:
    panel, link = collapsing_panel()
    with pytest.raises(LinkError, match="share a"):
        relink(panel, link, entity_name="permno")


@pytest.mark.parametrize(("policy", "expected"), [("first", 1.0), ("last", 3.0)])
def test_collision_first_and_last_keep_one_row(policy: str, expected: float) -> None:
    panel, link = collapsing_panel()
    out, report = relink(
        panel,
        link,
        entity_name="permno",
        on_collision=policy,  # type: ignore[arg-type]
    )
    assert len(out.frame) == 2
    assert out.frame["x"].to_numpy()[0] == expected
    assert report.n_collapsed == 1
    assert "collapsed" in str(report)


def test_collision_choice_does_not_depend_on_row_order() -> None:
    panel, link = collapsing_panel()
    shuffled = Panel(
        frame=panel.frame.iloc[[2, 1, 0]],
        entity_name=panel.entity_name,
    )
    kept = relink(panel, link, entity_name="permno", on_collision="first")[0]
    kept_shuffled = relink(shuffled, link, entity_name="permno", on_collision="first")[0]
    assert sorted(kept.frame["x"]) == sorted(kept_shuffled.frame["x"]) == [1.0, 5.0]


@pytest.mark.parametrize(("policy", "expected"), [("sum", 4.0), ("mean", 2.0)])
def test_collision_aggregates(policy: str, expected: float) -> None:
    panel, link = collapsing_panel()
    out, report = relink(
        panel,
        link,
        entity_name="permno",
        on_collision=policy,  # type: ignore[arg-type]
    )
    assert out.frame.loc[(pd.Timestamp("2020-01-01"), 10), "x"] == expected
    assert report.n_collapsed == 1


def test_collision_aggregate_refuses_non_numeric_columns() -> None:
    panel, link = collapsing_panel()
    panel = Panel(frame=panel.frame.assign(name=["a", "b", "c"]), entity_name=panel.entity_name)
    with pytest.raises(LinkError, match="numeric"):
        relink(panel, link, entity_name="permno", on_collision="sum")


def test_no_collision_leaves_the_panel_untouched() -> None:
    panel = make_panel([("2020-01-01", "A"), ("2020-01-01", "B")])
    link = LinkTable(
        table=pd.DataFrame({"gvkey": ["A", "B"], "permno": [10, 20]}),
        source_keys=("gvkey",),
        target="permno",
    )
    out, report = relink(panel, link, entity_name="permno")
    assert report.n_collapsed == 0
    assert list(out.frame.index.names) == ["date", "permno"]
    assert list(out.frame["x"]) == list(panel.frame["x"])


# -- 7. unmatched policies -----------------------------------------------------


def unmatched_setup() -> tuple[Panel, LinkTable]:
    panel = make_panel(
        [("2020-01-01", "A"), ("2020-01-01", "B"), ("2020-01-02", "B")],
        {"x": [1.0, 2.0, 3.0]},
    )
    link = LinkTable(
        table=pd.DataFrame({"gvkey": ["A"], "permno": [10]}),
        source_keys=("gvkey",),
        target="permno",
    )
    return panel, link


def test_unmatched_drop_removes_rows_and_keeps_the_integer_dtype() -> None:
    panel, link = unmatched_setup()
    out, report = relink(panel, link, entity_name="permno", on_unmatched="drop")
    assert list(out.frame["x"]) == [1.0]
    assert out.frame.index.get_level_values("permno").dtype == np.int64
    assert (report.n_rows, report.n_unmatched) == (3, 2)
    assert (report.n_keys, report.n_keys_unmatched) == (2, 1)


def test_unmatched_keep_retains_rows_with_a_null_entity() -> None:
    panel, link = unmatched_setup()
    out, _ = relink(panel, link, entity_name="permno", on_unmatched="keep")
    entities = out.frame.index.get_level_values("permno")
    assert len(out.frame) == 3 and int(entities.isna().sum()) == 2


def test_unmatched_error_refuses_and_reports() -> None:
    panel, link = unmatched_setup()
    with pytest.raises(LinkError, match=r"2/3 rows \(66.7%\) and 1/2 keys did not resolve"):
        relink(panel, link, entity_name="permno", on_unmatched="error")


def test_report_counts_a_partially_matched_key_as_matched() -> None:
    link = windowed_link(["A"], [10], ["2000-01-01"], ["2005-01-01"])
    query = pd.DataFrame({"gvkey": ["A", "A", "B"]})
    _, report = link.resolve(query, dates=dates("2001-01-01", "2009-01-01", "2001-01-01"))
    assert (report.n_rows, report.n_unmatched) == (3, 2)
    # A resolved somewhere, B never did.
    assert (report.n_keys, report.n_keys_unmatched) == (2, 1)


def test_report_string_is_readable() -> None:
    report = LinkReport(n_rows=36_000, n_unmatched=1_011, n_keys=1_200, n_keys_unmatched=600)
    assert str(report) == "1,011/36,000 rows (2.8%) and 600/1,200 keys did not resolve"


def test_relink_keeps_the_date_level_dtype() -> None:
    panel, link = unmatched_setup()
    out, _ = relink(panel, link, entity_name="permno")
    before = panel.frame.index.get_level_values("date").dtype
    assert out.frame.index.get_level_values("date").dtype == before


def test_relink_of_a_cross_section() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["A", "B"], name="gvkey"))
    panel = Panel(frame=frame, date_name=None, entity_name="gvkey")
    link = LinkTable(
        table=pd.DataFrame({"gvkey": ["A", "B"], "permno": [10, 20]}),
        source_keys=("gvkey",),
        target="permno",
    )
    out, _ = relink(panel, link, entity_name="permno")
    assert list(out.frame.index) == [10, 20] and out.frame.index.name == "permno"


def test_time_variant_link_needs_a_dated_panel() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=pd.Index(["A"], name="gvkey"))
    panel = Panel(frame=frame, date_name=None, entity_name="gvkey")
    with pytest.raises(LinkError, match="no date"):
        relink(panel, windowed_link(["A"], [10], ["2000-01-01"], [None]), entity_name="permno")


# -- 8. from_spine -------------------------------------------------------------


def recycled_ticker_spine() -> pd.DataFrame:
    """A ticker used by one firm, abandoned for a day, then taken over by another."""
    rows = [
        ("2020-01-01", "E1", "AAA"),
        ("2020-01-02", "E1", "AAA"),
        ("2020-01-03", "E2", "BBB"),
        ("2020-01-04", "E2", "AAA"),
        ("2020-01-05", "E2", "AAA"),
    ]
    return pd.DataFrame(
        {"ticker": [ticker for _, _, ticker in rows]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime([date for date, _, _ in rows]), [e for _, e, _ in rows]],
            names=["date", "entity"],
        ),
    )


def test_from_spine_breaks_a_run_where_a_key_is_abandoned() -> None:
    link = LinkTable.from_spine(
        recycled_ticker_spine(), ["ticker"], date_name="date", entity_name="entity"
    )
    windows = link.table[link.table["ticker"] == "AAA"].sort_values("valid_from")
    assert len(windows) == 2  # not one window spanning the gap
    assert list(windows["entity"]) == ["E1", "E2"]
    assert windows["valid_to"].iloc[0] == pd.Timestamp("2020-01-02")


def test_from_spine_leaves_the_gap_unresolvable() -> None:
    link = LinkTable.from_spine(
        recycled_ticker_spine(), ["ticker"], date_name="date", entity_name="entity"
    )
    resolved, _ = link.resolve(
        pd.DataFrame({"ticker": ["AAA"] * 5}),
        dates=pd.date_range("2020-01-01", periods=5, freq="D").to_numpy(),
    )
    assert [None if pd.isna(v) else v for v in resolved] == ["E1", "E1", None, "E2", "E2"]


def test_from_spine_last_window_is_open_unless_closed() -> None:
    spine = recycled_ticker_spine()
    open_ended = LinkTable.from_spine(spine, ["ticker"], date_name="date", entity_name="entity")
    later = np.array(["2030-01-01"], dtype="datetime64[ns]")
    assert list(open_ended.resolve(pd.DataFrame({"ticker": ["AAA"]}), dates=later)[0]) == ["E2"]

    closed = LinkTable.from_spine(
        spine, ["ticker"], date_name="date", entity_name="entity", close_last=True
    )
    assert pd.isna(closed.resolve(pd.DataFrame({"ticker": ["AAA"]}), dates=later)[0][0])


def test_from_spine_on_an_empty_frame() -> None:
    spine = recycled_ticker_spine().iloc[:0]
    link = LinkTable.from_spine(spine, ["ticker"], date_name="date", entity_name="entity")
    assert len(link.table) == 0
    values, report = link.resolve(pd.DataFrame({"ticker": ["AAA"]}), dates=dates("2020-01-01"))
    assert pd.isna(values[0]) and report.n_unmatched == 1


def test_from_spine_round_trips_through_relink() -> None:
    spine = recycled_ticker_spine()
    link = LinkTable.from_spine(spine, ["ticker"], date_name="date", entity_name="entity")
    panel = Panel(
        frame=spine.reset_index("entity", drop=True).set_index("ticker", append=True),
        entity_name="ticker",
    )
    out, report = relink(panel, link, entity_name="entity")
    assert report.n_unmatched == 0
    assert list(out.frame.index.get_level_values("entity")) == ["E1", "E1", "E2", "E2", "E2"]


# -- 9. identity ---------------------------------------------------------------


def test_identity_is_a_rename_not_a_lookup() -> None:
    link = LinkTable.identity("permno", "entity")
    assert link.passthrough and len(link.table) == 0
    frame = pd.DataFrame({"permno": [10, 20, 30]})
    values, report = link.resolve(frame)
    assert list(values) == [10, 20, 30]
    assert (report.n_rows, report.n_unmatched, report.n_keys) == (3, 0, 3)


def test_identity_relink_renames_the_level_and_keeps_the_dtype() -> None:
    link = LinkTable.identity("permno", "entity")
    panel = make_panel([("2020-01-01", 10), ("2020-01-01", 20)], entity_name="permno")
    out, _ = relink(panel, link, entity_name="entity")
    assert list(out.frame.index.names) == ["date", "entity"]
    assert list(out.frame.index.get_level_values("entity")) == [10, 20]


def test_identity_reports_missing_identifiers() -> None:
    link = LinkTable.identity("permno", "entity")
    values, report = link.resolve(pd.DataFrame({"permno": [10.0, np.nan]}))
    assert report.n_unmatched == 1 and pd.isna(values[1])


def test_identity_refuses_a_column_that_carries_nothing() -> None:
    link = LinkTable.identity("permno", "entity")
    with pytest.raises(LinkError, match="every one"):
        link.resolve(pd.DataFrame({"permno": [np.nan, np.nan]}))


def test_identity_cannot_be_time_variant() -> None:
    with pytest.raises(LinkError, match="identity"):
        LinkTable(
            table=pd.DataFrame({"a": [], "b": []}),
            source_keys=("a",),
            target="b",
            valid_from="c",
            passthrough=True,
        )


# -- 10. normalization ---------------------------------------------------------


def test_whitespace_and_empty_strings_are_handled_by_default() -> None:
    table = pd.DataFrame({"gvkey": [" A ", ""], "permno": [10, 20]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    assert len(link.table) == 1  # the empty key became missing and was dropped
    resolved = resolve_frame(link, ["A", " A", ""], None)
    assert list(resolved[:2]) == [10, 10] and pd.isna(resolved[2])


def test_case_is_significant_unless_upper_is_requested() -> None:
    table = pd.DataFrame({"iid": ["01c"], "permno": [10]})
    strict = LinkTable(table=table, source_keys=("iid",), target="permno")
    assert pd.isna(strict.resolve(pd.DataFrame({"iid": ["01C"]}))[0][0])

    folded = LinkTable(
        table=table,
        source_keys=("iid",),
        target="permno",
        normalize={"iid": KeyNormalization(upper=True)},
    )
    assert folded.resolve(pd.DataFrame({"iid": ["01C"]}))[0][0] == 10


def test_zfill_is_per_column_because_a_blanket_width_is_destructive() -> None:
    table = pd.DataFrame({"gvkey": ["001004"], "iid": ["01"], "permno": [10]})
    query = pd.DataFrame({"gvkey": ["1004"], "iid": ["01"]})
    padded = LinkTable(
        table=table,
        source_keys=("gvkey", "iid"),
        target="permno",
        normalize={"gvkey": KeyNormalization(zfill=6)},
    )
    assert padded.resolve(query)[0][0] == 10

    # The same width applied to every column pads the two-character iid to
    # "000001" on both sides -- consistent, but no longer the vendor's key, and
    # it collides with any other iid whose padded form matches.
    blanket = LinkTable(
        table=table,
        source_keys=("gvkey", "iid"),
        target="permno",
        normalize={
            "gvkey": KeyNormalization(zfill=6),
            "iid": KeyNormalization(zfill=6),
        },
    )
    assert blanket._key_columns[1].iloc[0] == "000001"


def test_normalization_leaves_a_categorical_categorical() -> None:
    table = pd.DataFrame(
        {"gvkey": pd.Categorical([" A ", "B "]), "permno": [10, 20]},
    )
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    assert isinstance(link._key_columns[0].dtype, pd.CategoricalDtype)
    assert list(link._key_columns[0].cat.categories) == ["A", "B"]


def test_normalization_leaves_numeric_and_stray_values_alone() -> None:
    table = pd.DataFrame({"gvkey": [1001, 1002], "permno": [10, 20]})
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    assert list(resolve_frame(link, [1001, 1002], None)) == [10, 20]

    mixed = pd.DataFrame({"gvkey": pd.Series([" A ", 7], dtype=object), "permno": [10, 20]})
    link = LinkTable(table=mixed, source_keys=("gvkey",), target="permno")
    values, _ = link.resolve(pd.DataFrame({"gvkey": pd.Series(["A", 7], dtype=object)}))
    assert list(values) == [10, 20]


def test_normalize_of_an_unknown_column_is_refused() -> None:
    with pytest.raises(LinkError, match="not source keys"):
        LinkTable(
            table=pd.DataFrame({"gvkey": ["A"], "permno": [10]}),
            source_keys=("gvkey",),
            target="permno",
            normalize={"iid": KeyNormalization()},
        )


# -- construction guards -------------------------------------------------------


def test_missing_columns_are_named() -> None:
    with pytest.raises(LinkError, match=r"missing columns \['iid'\]"):
        LinkTable(
            table=pd.DataFrame({"gvkey": ["A"], "permno": [10]}),
            source_keys=("gvkey", "iid"),
            target="permno",
        )


def test_target_cannot_be_a_source_key() -> None:
    with pytest.raises(LinkError, match="cannot also be a source key"):
        LinkTable(table=pd.DataFrame({"gvkey": ["A"]}), source_keys=("gvkey",), target="gvkey")


def test_valid_to_without_valid_from_is_refused() -> None:
    with pytest.raises(LinkError, match="needs valid_from"):
        LinkTable(
            table=pd.DataFrame({"gvkey": ["A"], "permno": [10], "to": [pd.Timestamp("2020")]}),
            source_keys=("gvkey",),
            target="permno",
            valid_to="to",
        )


def test_timezone_aware_dates_are_refused() -> None:
    table = pd.DataFrame(
        {
            "gvkey": ["A"],
            "permno": [10],
            "valid_from": pd.to_datetime(["2020-01-01"]).tz_localize("UTC"),
            "valid_to": pd.to_datetime(["2021-01-01"]).tz_localize("UTC"),
        }
    )
    with pytest.raises(LinkError, match="timezone"):
        LinkTable(
            table=table,
            source_keys=("gvkey",),
            target="permno",
            valid_from="valid_from",
            valid_to="valid_to",
        )


def test_the_table_passed_in_is_never_modified() -> None:
    table = pd.DataFrame({"gvkey": [" A ", "A"], "permno": [10, 10]})
    original = table.copy()
    LinkTable(table=table, source_keys=("gvkey",), target="permno")
    pd.testing.assert_frame_equal(table, original)


def test_unknown_policies_are_refused() -> None:
    table = pd.DataFrame({"gvkey": ["A"], "permno": [10]})
    with pytest.raises(LinkError, match="on_ambiguous"):
        LinkTable(
            table=table,
            source_keys=("gvkey",),
            target="permno",
            on_ambiguous="whatever",  # type: ignore[arg-type]
        )
    link = LinkTable(table=table, source_keys=("gvkey",), target="permno")
    panel = make_panel([("2020-01-01", "A")])
    with pytest.raises(LinkError, match="on_unmatched"):
        relink(panel, link, entity_name="permno", on_unmatched="maybe")  # type: ignore[arg-type]
    with pytest.raises(LinkError, match="on_collision"):
        relink(panel, link, entity_name="permno", on_collision="median")  # type: ignore[arg-type]


# -- the oracle ----------------------------------------------------------------


def random_link_table(seed: int, *, overlapping: bool) -> pd.DataFrame:
    """A link file with multi-window keys, gaps and open ends."""
    rng = np.random.default_rng(seed)
    calendar = pd.date_range("2000-01-01", periods=400, freq="7D")
    keys, targets, starts, ends = [], [], [], []
    for number in range(40):
        key = f"K{number:03d}"
        cursor = int(rng.integers(0, 50))
        for _ in range(int(rng.integers(1, 5))):
            if cursor >= len(calendar) - 2:
                break
            length = int(rng.integers(5, 60))
            stop = min(cursor + length, len(calendar) - 1)
            keys.append(key)
            targets.append(int(rng.integers(1, 500)))
            starts.append(calendar[cursor])
            open_ended = rng.random() < 0.15
            ends.append(pd.NaT if open_ended else calendar[stop])
            if open_ended:
                break  # an open end overlaps whatever would come after it
            if overlapping and rng.random() < 0.3:
                cursor = max(cursor, stop - int(rng.integers(1, 10)))
            else:
                cursor = stop + int(rng.integers(1, 8))
    return pd.DataFrame({"gvkey": keys, "permno": targets, "valid_from": starts, "valid_to": ends})


def oracle(table: pd.DataFrame, keys: Sequence[Any], when: np.ndarray) -> list[Any]:
    """The answer by brute force: scan every link row for every query row."""
    out: list[Any] = []
    rows = list(table.itertuples(index=False))
    for key, date in zip(keys, pd.to_datetime(when), strict=True):
        found = None
        for row in rows:
            if row.gvkey != key or pd.isna(date):
                continue
            end = pd.Timestamp("9999-12-31") if pd.isna(row.valid_to) else row.valid_to
            if row.valid_from <= date <= end:
                assert found is None, "the oracle needs a table with disjoint windows"
                found = row.permno
        out.append(found)
    return out


def query_rows(seed: int, size: int) -> tuple[list[Any], np.ndarray]:
    """Random keys (some unknown, some null) and dates (some missing)."""
    rng = np.random.default_rng(seed)
    calendar = pd.date_range("1999-01-01", periods=3000, freq="D")
    keys: list[Any] = []
    for _ in range(size):
        draw = rng.random()
        if draw < 0.05:
            keys.append(None)
        elif draw < 0.15:
            keys.append(f"U{rng.integers(0, 999):03d}")
        else:
            keys.append(f"K{rng.integers(0, 40):03d}")
    when = np.array(calendar[rng.integers(0, len(calendar), size)].to_numpy(), copy=True)
    when[rng.random(size) < 0.02] = np.datetime64("NaT")
    return keys, when


def test_fast_path_matches_a_brute_force_scan() -> None:
    table = random_link_table(20240521, overlapping=False)
    link = LinkTable(
        table=table,
        source_keys=("gvkey",),
        target="permno",
        valid_from="valid_from",
        valid_to="valid_to",
    )
    assert len(link.table) == len(table)  # nothing was rewritten
    keys, when = query_rows(11, 3000)
    values, report = link.resolve(pd.DataFrame({"gvkey": keys}), dates=when)
    expected = oracle(table, keys, when)
    assert [None if pd.isna(v) else int(v) for v in values] == expected
    assert report.n_unmatched == sum(value is None for value in expected)


def test_canonicalized_overlaps_match_a_brute_force_scan() -> None:
    table = random_link_table(99, overlapping=True)
    link = LinkTable(
        table=table,
        source_keys=("gvkey",),
        target="permno",
        valid_from="valid_from",
        valid_to="valid_to",
        on_ambiguous="last",
    )
    keys, when = query_rows(12, 3000)
    values, _ = link.resolve(pd.DataFrame({"gvkey": keys}), dates=when)
    # The oracle asserts the rewritten table is disjoint, then answers from it.
    expected = oracle(link.table, keys, when)
    assert [None if pd.isna(v) else int(v) for v in values] == expected


def test_unresolved_rows_are_not_reported_as_colliding_with_each_other() -> None:
    """``keep`` leaves several rows with a null entity; nulls are not a collision.

    Two rows that resolved to nothing are not two identifiers landing on one
    target, so the collision policy must not fire on them — the unmatched policy
    has already ruled on those rows.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]).as_unit("us")
    frame = pd.DataFrame(
        {"v": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_arrays(
            [dates, pd.array(["MISS_A", "MISS_B", "MISS_C"], dtype="string")],
            names=["date", "entity"],
        ),
    )
    table = pd.DataFrame({"entity": pd.array(["KNOWN"], dtype="string"), "permno": [10001]})
    link = LinkTable(table=table, source_keys=("entity",), target="permno")

    relinked, report = relink(
        Panel(frame), link, entity_name="entity", on_unmatched="keep", on_collision="error"
    )

    assert report.n_unmatched == 3
    assert report.n_collapsed == 0
    assert len(relinked.frame) == 3
    assert relinked.frame.index.get_level_values("entity").isna().all()
