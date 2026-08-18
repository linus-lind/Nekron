"""Tests for FeaturePipeline orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nekron.features import (
    ColumnSelection,
    CrossSectionalBucket,
    FeaturePipeline,
    ForwardReturns,
    GroupNeutralize,
    MovingAverage,
    RealizedVolatility,
    SimpleReturns,
)
from nekron.features.base import FeatureError, PanelContext

from .conftest import random_ohlcv


def test_order_invariance_and_index_preserved() -> None:
    panel = random_ohlcv(30, ["A", "B", "C"])
    pipe = FeaturePipeline(featurizers=(SimpleReturns("close", (1, 5), ("r1", "r5")),))
    date_major = panel.sort_index()
    entity_major = panel.swaplevel().sort_index().swaplevel()
    out_dm = pipe.apply(date_major)
    out_em = pipe.apply(entity_major)
    assert out_dm.index.equals(date_major.index)
    assert out_em.index.equals(entity_major.index)
    pd.testing.assert_frame_equal(out_dm.sort_index(), out_em.sort_index())


def test_drop_warmup_false_by_default_keeps_all_dates() -> None:
    panel = random_ohlcv(6, ["A", "B"]).sort_index()
    out = FeaturePipeline(featurizers=(SimpleReturns("close", (3,), ("r3",)),)).apply(panel)
    assert out.index.equals(panel.index)  # pipeline default drop_warmup=False -> no drop


def test_drop_warmup_discards_leading_burnin_dates() -> None:
    panel = random_ohlcv(6, ["A", "B"]).sort_index()
    dates = panel.index.get_level_values("date").unique()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (3,), ("r3",)),), drop_warmup=True
    ).apply(panel)
    # r3 (horizon 3) is all-NaN across the cross-section on the first 3 dates -> dropped
    kept = out.index.get_level_values("date").unique()
    assert list(kept) == list(dates[3:])
    assert out["r3"].notna().all()  # no burn-in NaN remains on kept dates


def test_drop_warmup_uses_longest_horizon() -> None:
    panel = random_ohlcv(12, ["A", "B", "C"]).sort_index()
    dates = panel.index.get_level_values("date").unique()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (2, 7), ("r2", "r7")),), drop_warmup=True
    ).apply(panel)
    kept = out.index.get_level_values("date").unique()
    assert list(kept) == list(dates[7:])  # the longest horizon (7) drives the cut


def test_drop_warmup_ignores_forward_target_trailing_nan() -> None:
    panel = random_ohlcv(10, ["A", "B"]).sort_index()
    dates = panel.index.get_level_values("date").unique()
    out = FeaturePipeline(
        featurizers=(
            SimpleReturns("close", (2,), ("r2",)),  # leading burn-in of 2
            ForwardReturns("close", (4,), ("fwd4",), method="simple"),  # trailing NaN only
        ),
        drop_warmup=True,
    ).apply(panel)
    kept = out.index.get_level_values("date").unique()
    assert list(kept) == list(dates[2:])  # forward target's trailing NaN does not extend warm-up


def test_dependency_threading() -> None:
    """A featurizer can consume a column produced by an earlier one."""
    panel = random_ohlcv(40, ["A", "B"]).sort_index()
    pipe = FeaturePipeline(
        featurizers=(
            SimpleReturns("close", (1,), ("r1",)),
            RealizedVolatility(
                "r1", (10,), ("rv10",), min_periods=10, ddof=1, annualization_factor=1.0
            ),
        )
    )
    out = pipe.apply(panel)
    assert "rv10" in out.columns
    assert out["rv10"].notna().any()


def test_keep_inputs_defaults_to_dropping_every_raw_column() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(featurizers=(SimpleReturns("close", (1,), ("r1",)),)).apply(panel)
    assert list(out.columns) == ["r1"]


def test_keep_inputs_retains_every_raw_column() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),),
        keep_inputs=ColumnSelection(),  # include=None -> no restriction
    ).apply(panel)
    assert list(out.columns) == [*panel.columns, "r1"]  # inputs first, features after


def test_keep_inputs_retains_a_subset_in_selection_order() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),),
        keep_inputs=ColumnSelection(include=("shares", "close")),
    ).apply(panel)
    assert list(out.columns) == ["shares", "close", "r1"]
    pd.testing.assert_series_equal(out["close"], panel["close"])  # values carried through as-is


def test_keep_inputs_excludes_from_the_full_panel() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),),
        keep_inputs=ColumnSelection(exclude=("ret",)),  # everything but 'ret'
    ).apply(panel)
    assert list(out.columns) == [c for c in panel.columns if c != "ret"] + ["r1"]


def test_keep_inputs_unknown_column_raises() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    pipe = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),),
        keep_inputs=ColumnSelection(include=("not_a_column",)),
    )
    with pytest.raises(KeyError, match="panel columns"):
        pipe.apply(panel)


def test_excluded_feature_is_computed_then_dropped() -> None:
    """A scaffolding feature still feeds the featurizers that follow it."""
    panel = random_ohlcv(40, ["A", "B"]).sort_index()
    feats = (
        SimpleReturns("close", (1,), ("r1",)),
        RealizedVolatility(
            "r1", (10,), ("rv10",), min_periods=10, ddof=1, annualization_factor=1.0
        ),
    )
    full = FeaturePipeline(featurizers=feats).apply(panel)
    pruned = FeaturePipeline(
        featurizers=feats, keep_features=ColumnSelection(exclude=("r1",))
    ).apply(panel)
    assert list(pruned.columns) == ["rv10"]  # r1 was computed, consumed, then dropped
    pd.testing.assert_series_equal(pruned["rv10"], full["rv10"])  # unchanged by the pruning


def test_keep_features_include_selects_a_subset() -> None:
    panel = random_ohlcv(20, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1, 5), ("r1", "r5")),),
        keep_features=ColumnSelection(include=("r5",)),
    ).apply(panel)
    assert list(out.columns) == ["r5"]


def test_keep_features_unknown_column_raises_on_construction() -> None:
    # the featurizers declare their outputs, so a stale name fails before any data is read
    with pytest.raises(KeyError, match="feature columns"):
        FeaturePipeline(
            featurizers=(SimpleReturns("close", (1,), ("r1",)),),
            keep_features=ColumnSelection(exclude=("r2",)),  # r2 is not produced
        )


def test_dropped_feature_does_not_extend_the_warmup() -> None:
    panel = random_ohlcv(12, ["A", "B"]).sort_index()
    dates = panel.index.get_level_values("date").unique()
    feats = (SimpleReturns("close", (2, 7), ("r2", "r7")),)
    both = FeaturePipeline(featurizers=feats, drop_warmup=True).apply(panel)
    without_r7 = FeaturePipeline(
        featurizers=feats, keep_features=ColumnSelection(exclude=("r7",)), drop_warmup=True
    ).apply(panel)
    assert list(both.index.get_level_values("date").unique()) == list(dates[7:])
    # r7 no longer reaches the output, so its 7-date burn-in no longer costs any dates
    assert list(without_r7.index.get_level_values("date").unique()) == list(dates[2:])


def test_output_dtype_applies_only_to_retained_features() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),),
        keep_inputs=ColumnSelection(include=("close",)),
        output_dtype="float32",
    ).apply(panel)
    assert out["r1"].dtype == np.float32
    assert out["close"].dtype == panel["close"].dtype  # a retained input keeps its own dtype


def test_output_dtype_cast() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    out = FeaturePipeline(
        featurizers=(SimpleReturns("close", (1,), ("r1",)),), output_dtype="float32"
    ).apply(panel)
    assert out["r1"].dtype == np.float32


def test_duplicate_output_name_raises() -> None:
    with pytest.raises(FeatureError, match="more than one"):  # caught at construction
        FeaturePipeline(
            featurizers=(
                SimpleReturns("close", (1,), ("dup",)),
                MovingAverage("close", (3,), ("dup",), kind="sma", min_periods=3, adjust=False),
            )
        )


def test_feature_consumed_before_it_is_produced_raises() -> None:
    with pytest.raises(FeatureError, match="produced later"):
        FeaturePipeline(
            featurizers=(  # the volatility reads r1 before the returns step writes it
                RealizedVolatility(
                    "r1", (5,), ("rv5",), min_periods=5, ddof=1, annualization_factor=1.0
                ),
                SimpleReturns("close", (1,), ("r1",)),
            )
        )


def test_grouping_key_produced_later_raises() -> None:
    # the group key is read through key_inputs, not inputs, but it still orders the steps
    with pytest.raises(FeatureError, match="produced later"):
        FeaturePipeline(
            featurizers=(
                SimpleReturns("close", (1,), ("r1",)),
                GroupNeutralize(("r1",), ("r1n",), group_column="bucket", mode="demean", ddof=1),
                CrossSectionalBucket(("shares",), ("bucket",), n_buckets=2),
            )
        )


def test_grouping_key_on_the_index_constrains_nothing() -> None:
    panel = random_ohlcv(8, ["A", "B"]).sort_index()
    out = FeaturePipeline(
        featurizers=(
            SimpleReturns("close", (1,), ("r1",)),
            GroupNeutralize(("r1",), ("r1n",), group_column="entity", mode="demean", ddof=1),
        )
    ).apply(panel)  # 'entity' is an index level, produced by no featurizer
    assert list(out.columns) == ["r1", "r1n"]


def test_feature_colliding_with_an_input_column_raises() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    pipe = FeaturePipeline(featurizers=(SimpleReturns("close", (1,), ("volume",)),))
    with pytest.raises(FeatureError, match="collides"):
        pipe.apply(panel)


class _Misdeclared:
    """A featurizer returning a column it does not declare as an output."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("declared",)

    def transform(self, panel: pd.DataFrame, ctx: PanelContext) -> dict[str, np.ndarray]:
        return {"actual": np.zeros(ctx.n_rows)}


def test_undeclared_output_raises() -> None:
    panel = random_ohlcv(5, ["A"]).sort_index()
    with pytest.raises(FeatureError, match="declares outputs"):
        FeaturePipeline(featurizers=(_Misdeclared(),)).apply(panel)


def test_missing_input_raises() -> None:
    panel = random_ohlcv(10, ["A"]).sort_index()
    pipe = FeaturePipeline(featurizers=(SimpleReturns("nonexistent", (1,), ("r1",)),))
    with pytest.raises(KeyError):
        pipe.apply(panel)


def test_single_level_index_rejected() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.Index([1, 2], name="date"))
    with pytest.raises(FeatureError, match="two-level"):
        FeaturePipeline(featurizers=(SimpleReturns("close", (1,), ("r1",)),)).apply(frame)


def test_empty_featurizers_returns_empty_frame() -> None:
    panel = random_ohlcv(5, ["A"]).sort_index()
    out = FeaturePipeline(featurizers=()).apply(panel)
    assert out.shape == (5, 0)
    assert out.index.equals(panel.index)


def test_unsorted_input_reordered_correctly() -> None:
    panel = random_ohlcv(20, ["A", "B"])
    shuffled = panel.sample(frac=1.0, random_state=3)
    pipe = FeaturePipeline(featurizers=(SimpleReturns("close", (1,), ("r1",)),))
    out = pipe.apply(shuffled)
    # compare against the sorted computation reindexed to the shuffled order
    ref = pipe.apply(panel.sort_index()).reindex(shuffled.index)
    pd.testing.assert_frame_equal(out, ref)
