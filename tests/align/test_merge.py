"""Tests for :func:`nekron.align.merge.merge_panels`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.align import AlignmentError, merge_panels
from nekron.align.config import AlignerSpec, AlignmentConfig, JoinSpec, LinkSpec
from nekron.panel import Panel, PanelSet

DATES = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]).as_unit("us")
ENTITIES = np.array([10001, 10002], dtype="int32")


def _spine() -> Panel:
    index = pd.MultiIndex.from_product([DATES, ENTITIES], names=["date", "entity"])
    frame = pd.DataFrame(
        {"close": np.arange(6, dtype="float64"), "volume": np.arange(6, dtype="float64") * 10},
        index=index,
    )
    return Panel(frame)


def _factors() -> Panel:
    frame = pd.DataFrame({"mktrf": [0.01, -0.02, 0.03]}, index=pd.Index(DATES, name="date"))
    return Panel(frame, date_name="date", entity_name=None)


def _sectors() -> Panel:
    frame = pd.DataFrame(
        {"sector": pd.Categorical(["Tech", "Energy"])},
        index=pd.Index(ENTITIES, name="entity"),
    )
    return Panel(frame, date_name=None, entity_name="entity")


def _panels(**extra: Panel) -> PanelSet:
    return PanelSet({"prices": _spine(), **extra})


def _join(panel: str, aligner: str, **kwargs: object) -> JoinSpec:
    params = kwargs.pop("params", {})
    return JoinSpec(panel=panel, aligner=AlignerSpec(type=aligner, params=dict(params)), **kwargs)  # type: ignore[arg-type]


def test_spine_only_merge_returns_the_spine() -> None:
    merged = merge_panels(_panels(), AlignmentConfig(spine="prices"))

    assert list(merged.frame.columns) == ["close", "volume"]
    assert merged.frame.index.equals(_spine().frame.index)
    assert merged.grain.value == "panel"


def test_spine_columns_can_be_projected() -> None:
    merged = merge_panels(_panels(), AlignmentConfig(spine="prices", spine_columns=["close"]))

    assert list(merged.frame.columns) == ["close"]


def test_broadcasts_preserve_row_count_order_and_dtypes() -> None:
    cfg = AlignmentConfig(
        spine="prices",
        joins=[_join("factors", "broadcast_time"), _join("sectors", "broadcast_entity")],
    )
    merged = merge_panels(_panels(factors=_factors(), sectors=_sectors()), cfg)

    # A merge adds columns and nothing else: same rows, same order.
    assert merged.frame.index.equals(_spine().frame.index)
    assert list(merged.frame.columns) == ["close", "volume", "mktrf", "sector"]
    # Every entity on 2020-01-03 sees that date's factor.
    assert merged.frame.loc[(pd.Timestamp("2020-01-03"),), "mktrf"].tolist() == [-0.02, -0.02]
    # Every date of entity 10001 sees its sector, still categorical.
    assert isinstance(merged.frame["sector"].dtype, pd.CategoricalDtype)
    assert set(merged.frame.xs(10001, level="entity")["sector"]) == {"Tech"}


def test_column_collision_is_refused_rather_than_suffixed() -> None:
    clashing = Panel(
        pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.Index(DATES, name="date")),
        date_name="date",
        entity_name=None,
    )
    cfg = AlignmentConfig(spine="prices", joins=[_join("other", "broadcast_time")])

    with pytest.raises(AlignmentError, match="would overwrite existing columns"):
        merge_panels(_panels(other=clashing), cfg)


def test_prefix_resolves_a_collision() -> None:
    clashing = Panel(
        pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.Index(DATES, name="date")),
        date_name="date",
        entity_name=None,
    )
    cfg = AlignmentConfig(spine="prices", joins=[_join("other", "broadcast_time", prefix="ff_")])
    merged = merge_panels(_panels(other=clashing), cfg)

    assert list(merged.frame.columns) == ["close", "volume", "ff_close"]


def test_rename_map_applies_after_the_prefix() -> None:
    cfg = AlignmentConfig(
        spine="prices",
        joins=[_join("factors", "broadcast_time", prefix="ff_", rename={"ff_mktrf": "market"})],
    )
    merged = merge_panels(_panels(factors=_factors()), cfg)

    assert list(merged.frame.columns) == ["close", "volume", "market"]


def test_a_grain_mismatch_names_the_aligner_that_would_fit() -> None:
    cfg = AlignmentConfig(spine="prices", joins=[_join("sectors", "broadcast_time")])

    with pytest.raises(AlignmentError, match="broadcast_entity"):
        merge_panels(_panels(sectors=_sectors()), cfg)


def test_required_join_that_matches_nothing_is_an_error() -> None:
    disjoint = Panel(
        pd.DataFrame({"other": [1.0]}, index=pd.Index(pd.to_datetime(["1999-01-01"]), name="date")),
        date_name="date",
        entity_name=None,
    )
    cfg = AlignmentConfig(spine="prices", joins=[_join("far", "broadcast_time", required=True)])

    with pytest.raises(AlignmentError, match="matched no spine row"):
        merge_panels(_panels(far=disjoint), cfg)


def test_min_coverage_reports_the_shortfall() -> None:
    partial = Panel(
        pd.DataFrame({"sector": ["Tech"]}, index=pd.Index(ENTITIES[:1], name="entity")),
        date_name=None,
        entity_name="entity",
    )
    cfg = AlignmentConfig(
        spine="prices", joins=[_join("half", "broadcast_entity", min_coverage=0.9)]
    )

    with pytest.raises(AlignmentError, match="covered 50.0% of spine rows"):
        merge_panels(_panels(half=partial), cfg)


def test_unknown_panel_name_lists_the_available_ones() -> None:
    cfg = AlignmentConfig(spine="prices", joins=[_join("typo", "broadcast_time")])

    with pytest.raises(Exception, match="unknown panel 'typo'"):
        merge_panels(_panels(factors=_factors()), cfg)


def test_stale_column_name_in_a_join_is_refused() -> None:
    cfg = AlignmentConfig(
        spine="prices", joins=[_join("factors", "broadcast_time", columns=["nope"])]
    )

    with pytest.raises(KeyError, match="nope"):
        merge_panels(_panels(factors=_factors()), cfg)


def test_link_translates_a_foreign_identifier_before_aligning() -> None:
    """A source keyed by gvkey reaches a permno spine through a link table."""
    foreign_index = pd.MultiIndex.from_product(
        [DATES, pd.Index(["G1", "G2"], dtype="string")], names=["date", "entity"]
    )
    foreign = Panel(pd.DataFrame({"assets": np.arange(6, dtype="float64")}, index=foreign_index))
    link = Panel(
        pd.DataFrame({"gvkey": pd.array(["G1", "G2"], dtype="string"), "permno": ENTITIES}),
        date_name=None,
        entity_name=None,
    )
    cfg = AlignmentConfig(
        spine="prices",
        joins=[
            JoinSpec(
                panel="fundamentals",
                aligner=AlignerSpec(type="exact", params={}),
                link=LinkSpec(
                    mode="table",
                    table="link",
                    source_keys=["entity"],
                    table_keys=["gvkey"],
                    target="permno",
                ),
                prefix="cs_",
            )
        ],
    )

    merged = merge_panels(_panels(fundamentals=foreign, link=link), cfg)

    assert list(merged.frame.columns) == ["close", "volume", "cs_assets"]
    assert merged.frame.index.equals(_spine().frame.index)
    # G1 -> 10001 and G2 -> 10002, so the values land on the matching permno rows.
    assert merged.frame["cs_assets"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
