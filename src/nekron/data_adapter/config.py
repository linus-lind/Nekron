"""Typed Hydra configuration for the end-to-end data adapter.

The adapter composes the stage schemas — panel ingestion
(:class:`~nekron.data.config.IngestionConfig`), preprocessing
(:class:`~nekron.preprocessing.config.PreprocessingConfig`), alignment
(:class:`~nekron.align.config.AlignmentConfig`) and feature creation
(:class:`~nekron.features.config.FeatureConfig`) — plus a temporal
:class:`SplitConfig` and the stage :class:`~nekron.cache.CacheConfig`.

The pipeline is no longer a chain but a fan-in: each named panel runs its own
preprocessing and feature stages *at its native grain*, the panels are then merged
onto one spine, and the merged panel runs a second preprocessing and feature
stage. That shape is what makes per-source work correct rather than merely
possible — a quarterly growth rate has to be differenced on quarterly rows, and
computing it after the values have been forward-filled onto a daily spine gives a
different, wrong answer.

Both preprocessing and feature creation therefore appear twice, as
:class:`PreprocessingStages` and :class:`FeatureStages`: a ``panels`` mapping
keyed by panel name, and a ``merged`` stage that runs after the join. Each entry
is an ordinary stage config, so the existing ``configs/preprocessing/*.yaml`` and
``configs/features/*.yaml`` group files mount into them unchanged::

    defaults:
      - preprocessing@data_pipeline.preprocessing.panels.crsp: crsp
      - features@data_pipeline.features.merged: crsp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from nekron.align.config import AlignmentConfig
from nekron.align.config import register_configs as register_alignment_configs
from nekron.cache import CacheConfig
from nekron.constants import DATE_LEVEL
from nekron.data.config import IngestionConfig
from nekron.data.config import register_configs as register_ingestion_configs
from nekron.features.config import FeatureConfig
from nekron.features.config import register_configs as register_feature_configs
from nekron.preprocessing.config import PreprocessingConfig
from nekron.preprocessing.config import register_configs as register_preprocessing_configs


@dataclass
class PreprocessingStages:
    """Cleaning applied per panel at its native grain, then once after the merge.

    Parameters
    ----------
    panels:
        Per-panel pipelines, keyed by the panel's ingestion name. A panel with no
        entry is passed through untouched.
    merged:
        The pipeline applied to the merged panel, for anything that can only be
        expressed once every source is present — imputing a fundamental from a
        market value, say.
    """

    panels: dict[str, PreprocessingConfig] = field(default_factory=dict)
    merged: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass
class FeatureStages:
    """Features computed per panel at its native grain, then once after the merge.

    Parameters
    ----------
    panels:
        Per-panel feature pipelines, keyed by the panel's ingestion name. These run
        before the join, which is the point: a horizon expressed in *rows* means
        something different on a quarterly panel than on a daily one. Only
        ``(date, entity)`` panels can carry a feature pipeline.
    merged:
        The feature pipeline applied to the merged panel — where the bulk of the
        feature set normally lives, since it is the only stage that sees every
        source at once.
    """

    panels: dict[str, FeatureConfig] = field(default_factory=dict)
    merged: FeatureConfig = field(default_factory=FeatureConfig)


SplitScheme = Literal["single", "walk_forward"]
"""Whether the dates are cut once, or swept by a train window that moves forward."""

WindowMode = Literal["rolling", "expanding"]
"""Whether a walk-forward train window slides, or stays anchored at the sample start."""

SPLIT_SCHEMES: tuple[str, ...] = ("single", "walk_forward")
WINDOW_MODES: tuple[str, ...] = ("rolling", "expanding")


@dataclass
class WalkForwardConfig:
    """A train/validation/test window swept forward across the panel's dates.

    Every size is a count of *periods* — positions in the sequence of dates the
    model can use — not calendar days and not rows of the panel. On a daily panel
    one period is one trading date, so the defaults below describe a five-year
    train window with one-year validation and test windows.

    Parameters
    ----------
    mode:
        ``"rolling"`` slides the train window forward with the rest of the fold, so
        every fold trains on the same amount of history and the model is never
        shown the distant past. ``"expanding"`` anchors the window at the start of
        the sample instead, so fold ``k`` trains on everything before its
        validation window. Rolling is the default: it is the scheme that tests
        whether a model still works when the regime it was fitted in has passed.
    train_size, val_size, test_size:
        Lengths of the three segments, in periods. All three must be positive —
        the validation window is what early stopping selects on, and a fold with
        no test window would be scored on nothing.
    step:
        How far the whole fold advances between folds. ``None`` uses
        :attr:`test_size`, which tiles the test windows edge to edge: every scored
        date is scored exactly once and the folds together form one continuous
        out-of-sample series. A smaller step overlaps them, which buys more folds
        at the cost of reusing dates.
    purge:
        Periods dropped at each internal boundary — after the train window and
        after the validation window. They belong to no split. A forward-return
        target at date ``t`` is realized over ``(t, t + H]``, so the last ``H - 1``
        dates of a segment carry labels that reach into the segment after it; the
        rule is therefore ``purge >= H - 1``. That is ``0`` for a one-period-ahead
        target and ``20`` for a monthly target on daily data. Nothing derives it
        automatically — the horizon lives in the featurizer that produced the
        target column, not here.
    max_folds:
        Keep only the first ``n`` folds. For smoke runs; ``None`` uses every fold
        the sample admits.
    """

    mode: str = "rolling"
    train_size: int = 1260
    val_size: int = 252
    test_size: int = 252
    step: int | None = None
    purge: int = 0
    max_folds: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in WINDOW_MODES:
            raise ValueError(
                f"split.walk_forward.mode must be one of {list(WINDOW_MODES)}; got {self.mode!r}."
            )
        for name in ("train_size", "val_size", "test_size"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"split.walk_forward.{name} must be positive; got {getattr(self, name)}."
                )
        if self.purge < 0:
            raise ValueError(f"split.walk_forward.purge must not be negative; got {self.purge}.")
        if self.step is not None and self.step <= 0:
            raise ValueError(
                "split.walk_forward.step must be positive; leave it null to advance by "
                f"test_size. Got {self.step}."
            )
        if self.max_folds is not None and self.max_folds <= 0:
            raise ValueError(
                "split.walk_forward.max_folds must be positive; leave it null to use every "
                f"fold the sample admits. Got {self.max_folds}."
            )


@dataclass
class SplitConfig:
    """How the merged panel is cut into train/validation/test.

    Two schemes share one representation. ``"single"`` cuts the dates once at two
    timestamps: ``date <= train_end`` -> train, ``train_end < date <= val_end`` ->
    val, ``date > val_end`` -> test, with no overlap. ``"walk_forward"`` ignores
    those two dates and sweeps the window described by :attr:`walk_forward`
    instead, producing one fold per position of that window. A single split is the
    degenerate one-fold schedule, so both schemes hand every consumer the same
    thing and nothing downstream has to branch.

    Parameters
    ----------
    scheme:
        ``"single"`` (the default, and the historical behaviour) or
        ``"walk_forward"``.
    train_end, val_end:
        ISO date strings bounding the single split; when omitted they are derived
        from the ``train_fraction`` / ``val_fraction`` quantiles of the available
        dates. Ignored under ``"walk_forward"``.
    burn_in:
        Leading dates discarded before any fold is cut, under either scheme. It
        exists because features are computed over the whole panel before it is
        split, so the first rows of the sample sit inside the longest trailing
        window a featurizer uses (252 rows, for a twelve-month statistic) and
        carry a column that is entirely missing — which the model's
        standardization then maps to a constant. Under a single split those dates
        are diluted across a long train set; under walk-forward they can be a
        large fraction of the first fold, which is where model selection happens
        first. ``0`` keeps every date, which is what the single split has always
        done.
    """

    scheme: str = "single"
    train_end: str | None = None
    val_end: str | None = None
    date_level: str = DATE_LEVEL
    train_fraction: float = 0.6
    val_fraction: float = 0.8
    burn_in: int = 0
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    def __post_init__(self) -> None:
        if self.scheme not in SPLIT_SCHEMES:
            raise ValueError(
                f"split.scheme must be one of {list(SPLIT_SCHEMES)}; got {self.scheme!r}."
            )
        if not 0.0 < self.train_fraction <= self.val_fraction < 1.0:
            raise ValueError(
                "require 0 < train_fraction <= val_fraction < 1; got "
                f"train_fraction={self.train_fraction}, val_fraction={self.val_fraction}."
            )
        if self.burn_in < 0:
            raise ValueError(f"split.burn_in must not be negative; got {self.burn_in}.")


@dataclass
class DataAdapterConfig:
    """Full raw-to-model-ready pipeline: ingest -> per-panel -> align -> merged -> split."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    preprocessing: PreprocessingStages = field(default_factory=PreprocessingStages)
    features: FeatureStages = field(default_factory=FeatureStages)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


CONFIG_SCHEMA_NAME = "base_data_adapter"


def register_configs() -> None:
    """Register the adapter schema and every stage schema with Hydra's ConfigStore.

    Registers the stage schemas first so the group config files a model composes
    into its ``data_pipeline`` section resolve, then the adapter schema itself.
    Safe to call more than once.
    """
    register_ingestion_configs()
    register_preprocessing_configs()
    register_feature_configs()
    register_alignment_configs()
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=DataAdapterConfig)


def to_config(cfg: DictConfig) -> DataAdapterConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed ``DataAdapterConfig``."""
    return cast(DataAdapterConfig, OmegaConf.to_object(cfg))
