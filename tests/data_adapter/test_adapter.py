"""End-to-end tests for the fan-in data adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from nekron.align.config import AlignerSpec, AlignmentConfig, JoinSpec
from nekron.cache import CacheConfig, CacheError
from nekron.data.config import IngestionConfig
from nekron.data_adapter import (
    AdapterError,
    DataAdapterConfig,
    FeatureStages,
    PreprocessingStages,
    SplitConfig,
    assemble,
    build_datasets,
)
from nekron.features.config import FeatureConfig, FeaturizerSpec
from nekron.panel import PanelError
from nekron.preprocessing.base import PreprocessingError
from nekron.preprocessing.config import PreprocessingConfig, TransformSpec

Ingestion = Callable[..., IngestionConfig]


def _adapter(
    ingestion: IngestionConfig,
    *,
    cache_dir: Path,
    joins: list[JoinSpec] | None = None,
    preprocessing: PreprocessingStages | None = None,
    features: FeatureStages | None = None,
    cache_enabled: bool = True,
) -> DataAdapterConfig:
    return DataAdapterConfig(
        ingestion=ingestion,
        preprocessing=preprocessing or PreprocessingStages(),
        features=features or FeatureStages(),
        alignment=AlignmentConfig(spine="prices", joins=joins or []),
        split=SplitConfig(train_fraction=0.25, val_fraction=0.5),
        cache=CacheConfig(enabled=cache_enabled, directory=str(cache_dir)),
    )


def _join(panel: str, aligner: str, **params: object) -> JoinSpec:
    return JoinSpec(panel=panel, aligner=AlignerSpec(type=aligner, params=dict(params)))


def test_single_panel_pipeline_matches_the_source(ingestion: Ingestion, tmp_path: Path) -> None:
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    panel = assemble(cfg)

    assert list(panel.frame.index.names) == ["date", "entity"]
    assert list(panel.frame.columns) == ["DlyClose", "DlyVol"]
    assert len(panel.frame) == 8


def test_three_grains_merge_onto_the_spine(ingestion: Ingestion, tmp_path: Path) -> None:
    cfg = _adapter(
        ingestion("prices", "factors", "sectors"),
        cache_dir=tmp_path / "cache",
        joins=[_join("factors", "broadcast_time"), _join("sectors", "broadcast_entity")],
    )
    panel = assemble(cfg)

    assert list(panel.frame.columns) == ["DlyClose", "DlyVol", "mktrf", "rf", "sector"]
    assert len(panel.frame) == 8
    # The factor for a date reaches both entities; the sector reaches every date.
    assert panel.frame.loc[(pd.Timestamp("2020-01-03"),), "mktrf"].tolist() == [-0.02, -0.02]
    assert set(panel.frame.xs(10001, level="entity")["sector"]) == {"Tech"}
    assert isinstance(panel.frame["sector"].dtype, pd.CategoricalDtype)


def test_per_panel_preprocessing_runs_at_native_grain(ingestion: Ingestion, tmp_path: Path) -> None:
    """A constant fill on the factor series happens before it is broadcast."""
    stages = PreprocessingStages(
        panels={
            "factors": PreprocessingConfig(
                steps=[
                    TransformSpec(type="constant_fill", params={"columns": ["rf"], "value": 0.0})
                ]
            )
        }
    )
    cfg = _adapter(
        ingestion("prices", "factors"),
        cache_dir=tmp_path / "cache",
        joins=[_join("factors", "broadcast_time")],
        preprocessing=stages,
    )
    panel = assemble(cfg)

    assert panel.frame["rf"].notna().all()


def test_a_panel_only_transform_is_refused_on_a_cross_section(
    ingestion: Ingestion, tmp_path: Path
) -> None:
    """The grain guard turns a silent wrong answer into a configuration error."""
    stages = PreprocessingStages(
        panels={
            "sectors": PreprocessingConfig(
                steps=[TransformSpec(type="forward_fill", params={"columns": ["sector"]})]
            )
        }
    )
    cfg = _adapter(
        ingestion("prices", "sectors"),
        cache_dir=tmp_path / "cache",
        joins=[_join("sectors", "broadcast_entity")],
        preprocessing=stages,
    )

    with pytest.raises(PreprocessingError, match="is defined for grain panel, but the panel is"):
        assemble(cfg)


def test_features_are_refused_on_a_non_panel_grain(ingestion: Ingestion, tmp_path: Path) -> None:
    stages = FeatureStages(
        panels={
            "factors": FeatureConfig(
                featurizers=[
                    FeaturizerSpec(
                        type="temporal_shift",
                        params={"input_column": "mktrf", "periods": [1], "output_names": ["lag"]},
                    )
                ]
            )
        }
    )
    cfg = _adapter(
        ingestion("prices", "factors"),
        cache_dir=tmp_path / "cache",
        joins=[_join("factors", "broadcast_time")],
        features=stages,
    )

    with pytest.raises(PanelError, match="feature stage for panel 'factors'"):
        assemble(cfg)


def test_merged_features_see_every_source(ingestion: Ingestion, tmp_path: Path) -> None:
    """A feature over a broadcast factor column proves the merged stage runs last."""
    stages = FeatureStages(
        merged=FeatureConfig(
            featurizers=[
                FeaturizerSpec(
                    type="temporal_shift",
                    params={"input_column": "mktrf", "periods": [1], "output_names": ["mktrf_lag"]},
                )
            ],
            drop_warmup=False,
        )
    )
    cfg = _adapter(
        ingestion("prices", "factors"),
        cache_dir=tmp_path / "cache",
        joins=[_join("factors", "broadcast_time")],
        features=stages,
    )
    panel = assemble(cfg)

    assert list(panel.frame.columns) == ["mktrf_lag"]
    assert len(panel.frame) == 8


def _corrupt_keeping_identity(path: Path) -> None:
    """Replace a file's contents with unparseable bytes of the same size and mtime."""
    stat = path.stat()
    path.write_bytes(b"?" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_a_cached_run_never_reparses_the_source(ingestion: Ingestion, tmp_path: Path) -> None:
    """The whole key chain is known before any read, so a hit skips ingestion entirely.

    The source is replaced with garbage that a parser could not survive, while its
    size and modification time are restored so the ``stat`` fingerprint is
    unchanged. A run that still returns the right answer cannot have opened it.
    """
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    first = assemble(cfg)

    _corrupt_keeping_identity(Path(cfg.ingestion.panels["prices"].source.csv.path))

    second = assemble(cfg)
    pd.testing.assert_frame_equal(second.frame, first.frame, check_exact=True)


def test_content_fingerprinting_catches_what_stat_cannot(
    ingestion: Ingestion, tmp_path: Path
) -> None:
    """The documented blind spot of ``stat`` mode, and the escape hatch that closes it."""
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    cfg.cache.fingerprint = "content"
    assemble(cfg)

    _corrupt_keeping_identity(Path(cfg.ingestion.panels["prices"].source.csv.path))

    # Same size, same mtime, different bytes: the digest changes, so the cached
    # artifact is not reused and the unreadable file is actually opened.
    with pytest.raises(Exception, match="missing schema columns|Error tokenizing|is empty"):
        assemble(cfg)


def test_a_missing_source_is_reported_even_with_a_warm_cache(
    ingestion: Ingestion, tmp_path: Path
) -> None:
    """A key cannot be computed without the file, so a hit cannot mask its absence."""
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    assemble(cfg)
    Path(cfg.ingestion.panels["prices"].source.csv.path).unlink()

    with pytest.raises(CacheError, match="cannot fingerprint source file"):
        assemble(cfg)


def test_changing_a_spec_invalidates_the_cached_stage(ingestion: Ingestion, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    assemble(_adapter(ingestion("prices"), cache_dir=cache_dir))

    narrowed = ingestion("prices")
    narrowed.panels["prices"].selection.columns = ["DlyClose"]
    panel = assemble(_adapter(narrowed, cache_dir=cache_dir))

    assert list(panel.frame.columns) == ["DlyClose"]


def test_editing_the_source_file_invalidates_the_cache(
    ingestion: Ingestion, tmp_path: Path
) -> None:
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    assert len(assemble(cfg).frame) == 8

    source = Path(cfg.ingestion.panels["prices"].source.csv.path)
    source.write_text(
        source.read_text(encoding="utf-8") + "10001,08/01/2020,14.0,140\n", encoding="utf-8"
    )
    assert len(assemble(cfg).frame) == 9


def test_disabled_cache_writes_nothing(ingestion: Ingestion, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cfg = _adapter(ingestion("prices"), cache_dir=cache_dir, cache_enabled=False)
    assemble(cfg)

    assert not cache_dir.exists()


def test_build_datasets_splits_by_date(ingestion: Ingestion, tmp_path: Path) -> None:
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    splits = build_datasets(cfg)

    assert splits.feature_columns == ("DlyClose", "DlyVol")
    assert len(splits.train) + len(splits.val) + len(splits.test) == 8
    assert splits.train.index.get_level_values("date").max() <= splits.train_end
    assert splits.test.index.get_level_values("date").min() > splits.val_end


def test_splitting_on_a_missing_level_names_the_index(ingestion: Ingestion, tmp_path: Path) -> None:
    cfg = _adapter(ingestion("prices"), cache_dir=tmp_path / "cache")
    cfg.split.date_level = "nope"

    with pytest.raises(AdapterError, match="cannot split on level 'nope'"):
        build_datasets(cfg)
