"""Tests for the featurizer registry and Hydra configuration path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from nekron.features import (
    ColumnSelection,
    ColumnSelectionConfig,
    FeatureConfig,
    FeaturizerSpec,
    SimpleReturns,
    build_featurizer,
    build_pipeline,
    register_configs,
    registered_featurizers,
    to_config,
    to_selection,
)
from nekron.features.base import FeatureError

from .conftest import random_ohlcv

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "features"

_OBV_PARAMS = {"close_column": "close", "volume_column": "volume", "output_name": "obv"}
_OBV_DIFF_PARAMS = {"input_column": "obv", "periods": [1], "output_names": ["obv_change"]}


def test_registry_has_full_catalog() -> None:
    names = registered_featurizers()
    assert len(names) > 50
    assert "simple_returns" in names and "yang_zhang_volatility" in names


def test_build_featurizer_coerces_lists_to_tuples() -> None:
    feat = build_featurizer(
        "simple_returns",
        {"input_column": "close", "horizons": [1, 5], "output_names": ["r1", "r5"]},
    )
    assert isinstance(feat, SimpleReturns)
    assert feat.horizons == (1, 5) and feat.output_names == ("r1", "r5")


def test_build_featurizer_unknown_type_raises() -> None:
    with pytest.raises(FeatureError, match="unknown featurizer type"):
        build_featurizer("does_not_exist", {})


def test_build_featurizer_bad_params_raises() -> None:
    with pytest.raises(FeatureError, match="cannot build"):
        build_featurizer("simple_returns", {"input_column": "close"})  # missing required fields


def test_selection_defaults_drop_inputs_and_keep_features() -> None:
    cfg = FeatureConfig()
    assert to_selection(cfg.keep_inputs) == ColumnSelection(include=())  # no raw columns
    assert to_selection(cfg.keep_features) == ColumnSelection()  # every produced feature


def test_selection_config_is_coerced_to_tuples() -> None:
    pipeline = build_pipeline(
        FeatureConfig(
            featurizers=[
                FeaturizerSpec(
                    type="simple_returns",
                    params={"input_column": "close", "horizons": [1], "output_names": ["r1"]},
                )
            ],
            keep_inputs=ColumnSelectionConfig(include=["close"], exclude=[], strict=False),
        )
    )
    assert pipeline.keep_inputs == ColumnSelection(include=("close",), strict=False)


def test_intermediate_output_is_computed_then_dropped() -> None:
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obv"]),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ]
    )
    pipeline = build_pipeline(cfg)
    assert pipeline.keep_features.exclude == ("obv",)  # marking compiles into an exclusion
    features = pipeline.apply(random_ohlcv(10, ["A", "B"]).sort_index())
    assert list(features.columns) == ["obv_change"]  # the level fed the difference, then went


def test_intermediate_marking_is_merged_with_an_explicit_exclusion() -> None:
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obv"]),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ],
        keep_features=ColumnSelectionConfig(exclude=["obv_change"]),
    )
    assert build_pipeline(cfg).keep_features.exclude == ("obv_change", "obv")


def test_intermediate_may_be_consumed_as_a_grouping_key() -> None:
    """A bucket column is scaffolding even though it is read as a key, not an input."""
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(
                type="simple_returns",
                params={"input_column": "close", "horizons": [1], "output_names": ["r1"]},
            ),
            FeaturizerSpec(
                type="cross_sectional_bucket",
                params={
                    "input_columns": ["shares"],
                    "output_names": ["size_bucket"],
                    "n_buckets": 2,
                },
                intermediate=["size_bucket"],
            ),
            FeaturizerSpec(
                type="group_neutralize",
                params={
                    "input_columns": ["r1"],
                    "output_names": ["r1_neutral"],
                    "group_column": "size_bucket",  # a key_inputs dependency, not an input
                    "mode": "demean",
                    "ddof": 1,
                },
            ),
        ]
    )
    features = build_pipeline(cfg).apply(random_ohlcv(10, ["A", "B", "C", "D"]).sort_index())
    assert list(features.columns) == ["r1", "r1_neutral"]  # the buckets did their job and went


def test_intermediate_also_requested_as_an_output_raises() -> None:
    # asking for obv while marking it scaffolding is self-contradictory, not a silent drop
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obv"]),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ],
        keep_features=ColumnSelectionConfig(include=["obv", "obv_change"]),
    )
    with pytest.raises(FeatureError, match="cannot be both"):
        build_pipeline(cfg)


def test_intermediate_repeated_in_an_exclusion_is_listed_once() -> None:
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obv"]),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ],
        keep_features=ColumnSelectionConfig(exclude=["obv"]),
    )
    assert build_pipeline(cfg).keep_features.exclude == ("obv",)


def test_intermediate_must_name_an_output_of_its_own_featurizer() -> None:
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obvv"]),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ]
    )
    with pytest.raises(FeatureError, match="its outputs are"):
        build_pipeline(cfg)


def test_intermediate_without_a_consumer_raises() -> None:
    # nothing downstream reads obv, so computing it only to discard it is a config mistake
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS, intermediate=["obv"])
        ]
    )
    with pytest.raises(FeatureError, match="no later featurizer consumes"):
        build_pipeline(cfg)


def test_unconsumed_output_is_dropped_through_keep_features() -> None:
    # the counterpart to the test above: an unwanted output nothing consumes is excluded
    cfg = FeatureConfig(
        featurizers=[
            FeaturizerSpec(type="on_balance_volume", params=_OBV_PARAMS),
            FeaturizerSpec(type="difference", params=_OBV_DIFF_PARAMS),
        ],
        keep_features=ColumnSelectionConfig(exclude=["obv"]),
    )
    features = build_pipeline(cfg).apply(random_ohlcv(10, ["A", "B"]).sort_index())
    assert list(features.columns) == ["obv_change"]


def test_crsp_config_composes_and_runs() -> None:
    register_configs()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="crsp")
    pipeline = build_pipeline(to_config(cfg))
    assert len(pipeline.featurizers) > 40

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=280)  # > 252 so annual features define
    entities = list(range(10000, 10010))
    idx = pd.MultiIndex.from_product([dates, entities], names=["date", "entity"])
    n = len(idx)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    panel = pd.DataFrame(
        {
            "DlyOpen": close * 1.001,
            "DlyHigh": close * 1.01,
            "DlyLow": close * 0.99,
            "DlyClose": close,
            "DlyVol": rng.integers(10_000, 1_000_000, n).astype(float),
            "DlyPrcVol": close * rng.integers(10_000, 1_000_000, n),
            "ShrOut": rng.integers(1_000_000, 100_000_000, n).astype(float),
            "DlyRet": rng.normal(0, 0.01, n),
        },
        index=idx,
    )
    features = pipeline.apply(panel)
    # drop_warmup=True (config default) trims the leading burn-in dates, so the
    # feature index is a nonempty date-suffix of the panel.
    all_dates = panel.index.get_level_values("date").unique()
    kept_dates = features.index.get_level_values("date").unique()
    assert 0 < len(kept_dates) < len(all_dates)
    assert list(kept_dates) == list(all_dates[len(all_dates) - len(kept_dates) :])
    assert features.index.isin(panel.index).all()
    assert features.shape[1] > 60
    assert not np.isinf(features.to_numpy(dtype=np.float64)).any()
    # keep_inputs.include is [] in the config, so no raw panel column survives
    assert not set(panel.columns) & set(features.columns)
    # the cumulative OBV level is marked intermediate: differenced, then dropped
    assert "obv_change" in features.columns and "obv" not in features.columns
