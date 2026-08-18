"""End-to-end data adapter: ingest -> per-panel -> align -> merged -> temporal split.

One configuration-driven entry, :func:`build_datasets`, runs the whole pipeline so
a model never re-implements this wiring.

The pipeline is a fan-in, not a chain. Every named source is ingested, cleaned and
optionally featurized independently, *at its own grain*; the panels are then
merged onto one spine, and the merged panel goes through a second cleaning and
feature stage before being split by date. Per-source work happens before the join
because a horizon expressed in rows means something different on a quarterly panel
than on a daily one — a year-on-year growth rate has to be differenced on
quarterly rows, not on the forward-filled daily copies of them.

Features are computed on the full panel before the split: every featurizer is
backward-looking (rolling and cross-sectional operations use only same-or-prior
dates), so no test information leaks into earlier splits.

Splitting
---------
The assembled panel is cut by :mod:`nekron.data_adapter.splitting`, which produces
a *schedule* of folds rather than a single split — one fold under the default
``single`` scheme, and as many as the sample admits under ``walk_forward``.
:func:`build_folds` returns that schedule as a :class:`FoldPlan`, which
materializes a fold's frames only when asked: an expanding walk-forward sweep
overlaps heavily, and holding every fold's slices at once would multiply the
largest object in the process by the number of folds. :func:`build_datasets` is
the unchanged single-split entry point and still returns one :class:`DataSplits`.

Caching
-------
Each stage is content-addressed by a digest of its own configuration chained onto
the digest of the stage upstream, so editing one panel's feature list recomputes
that panel and everything downstream while every other panel is read back from
disk.

Crucially the *whole key chain is computed before any data is read*: a key depends
only on configuration and on the identity of the source files, never on their
contents. The adapter can therefore look for the deepest cached stage first and
start from there — a run whose merged panel is already cached never opens a CSV at
all, and one that changed only the merged feature list re-reads a single artifact
instead of every source.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from nekron.align import merge_panels
from nekron.cache import PanelCache, build_cache, canonical_json, stage_key
from nekron.data import load_panel, source_paths
from nekron.features import build_pipeline as build_feature_pipeline
from nekron.features.config import FeatureConfig
from nekron.panel import Panel, PanelGrain, PanelSet
from nekron.preprocessing import build_pipeline as build_preprocessing_pipeline
from nekron.preprocessing.config import PreprocessingConfig

from .base import AdapterError
from .config import DataAdapterConfig, SplitConfig
from .splitting import FoldBounds, FoldWindow, plan_folds, resolve_cut_dates

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterError",
    "DataSplits",
    "FoldPlan",
    "assemble",
    "assemble_panel",
    "build_datasets",
    "build_folds",
    "plan_panel",
    "split_panel",
]


@dataclass(frozen=True)
class DataSplits:
    """Temporal train/val/test feature panels sharing a ``(date, entity)`` MultiIndex."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_end: pd.Timestamp
    val_end: pd.Timestamp


@dataclass(frozen=True)
class FoldPlan:
    """A merged panel plus the fold schedule cut over its dates.

    The frames of a fold are produced on demand by :meth:`splits`, never held for
    every fold at once. That is not premature economy: a walk-forward sweep with an
    expanding window has fold ``k`` contain fold ``k-1`` almost entirely, so
    materializing the schedule eagerly would hold several near-copies of the
    largest object in the process.

    Parameters
    ----------
    panel:
        The merged, featurized panel, indexed by ``(date, entity)``.
    dates:
        Its sorted unique dates — the sequence the fold positions index into.
    bounds:
        One :class:`~nekron.data_adapter.splitting.FoldBounds` per fold.
    cut_dates:
        The inclusive upper date bound of each fold's train and validation
        segments, in fold order. Stored rather than derived because the two
        schemes disagree on what it is: a single split is defined *by* its two cut
        dates, which need not be dates the panel carries, while a walk-forward
        fold is defined by positions and its bound is simply its last date.
    """

    panel: pd.DataFrame
    dates: pd.Index
    bounds: tuple[FoldBounds, ...]
    cut_dates: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    feature_columns: tuple[str, ...]
    date_level: str

    def __len__(self) -> int:
        return len(self.bounds)

    def __iter__(self) -> Iterator[FoldBounds]:
        return iter(self.bounds)

    def window(self, index: int) -> FoldWindow:
        """Fold ``index`` resolved against real dates, for logging or reporting."""
        return self.bounds[index].window(self.dates)

    def windows(self) -> tuple[FoldWindow, ...]:
        """Every fold resolved against real dates, in fold order."""
        return tuple(bounds.window(self.dates) for bounds in self.bounds)

    def splits(self, index: int) -> DataSplits:
        """Materialize fold ``index`` as train/validation/test frames."""
        bounds = self.bounds[index]
        train_end, val_end = self.cut_dates[index]
        # One pass over the index, reused for all three segments: on a panel of
        # millions of rows this is the whole cost of the call.
        date_values = self.panel.index.get_level_values(self.date_level)
        return DataSplits(
            train=self._segment(date_values, bounds.train),
            val=self._segment(date_values, bounds.val),
            test=self._segment(date_values, bounds.test),
            feature_columns=self.feature_columns,
            train_end=train_end,
            val_end=val_end,
        )

    def _segment(self, date_values: pd.Index, span: range) -> pd.DataFrame:
        """The rows whose date falls in ``span``, as a mask over the whole panel.

        A mask rather than a sort-and-slice: the panel is not guaranteed to be
        sorted by date, and re-sorting it per fold would cost more than the scan.
        """
        if len(span) == 0:
            return self.panel.iloc[:0]
        low = self.dates[span.start]
        high = self.dates[span.stop - 1]
        segment: pd.DataFrame = self.panel[(date_values >= low) & (date_values <= high)]
        return segment


@dataclass(frozen=True)
class _Step:
    """One cacheable stage: what it is called, what it hashes to, and how to run it.

    ``run`` takes the previous stage's panel — ``None`` for a stage that produces
    one from scratch — so a chain can be resumed from whichever step was cached.
    """

    label: str
    key: str
    spec: Any
    run: Callable[[Panel | None], Panel] = field(compare=False)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def assemble_panel(cfg: DataAdapterConfig) -> pd.DataFrame:
    """Run ingestion, the per-panel stages, alignment and the merged stages."""
    return assemble(cfg).frame


def assemble(cfg: DataAdapterConfig) -> Panel:
    """Run the full fan-in and return the merged, featurized panel."""
    cache = build_cache(cfg.cache)
    referenced = _referenced_panels(cfg)
    chains = {name: chain for name, chain in _panel_chains(cfg).items() if name in referenced}
    _warn_unreferenced(cfg, referenced)
    panel_keys = {name: chain[-1].key for name, chain in chains.items()}

    loaded: dict[str, PanelSet] = {}

    def prepared() -> PanelSet:
        """Materialize every panel, but only if the merged stages actually need it."""
        if "panels" not in loaded:
            loaded["panels"] = PanelSet(
                {name: _resolve(cache, chain) for name, chain in chains.items()}
            )
        return loaded["panels"]

    return _resolve(cache, _merged_chain(cfg, panel_keys, prepared))


def dataset_fingerprint(cfg: DataAdapterConfig) -> str:
    """The digest of the terminal stage: which panel a run would actually see.

    The same key the cache stores the finished panel under, which makes it the one
    string that answers "was this the same data" across two runs. It covers every
    stage configuration in the chain and the identity of every source file, so it
    moves when the feature list moves, when a preprocessing step is inserted, and
    when the CSV underneath is replaced — none of which the configuration
    parameters recorded alongside a run would necessarily reveal, since a file can
    change without its path doing so.

    Costs a ``stat`` per source file under the default fingerprint mode. Under
    ``cache.fingerprint="content"`` it digests the source files instead, which is
    seconds per gigabyte and is paid again when the panel is assembled.
    """
    referenced = _referenced_panels(cfg)
    chains = {name: chain for name, chain in _panel_chains(cfg).items() if name in referenced}
    panel_keys = {name: chain[-1].key for name, chain in chains.items()}
    return _merged_chain(cfg, panel_keys, _unassembled)[-1].key


def _unassembled() -> PanelSet:
    """Stand in for the panels a key computation never needs to materialize."""
    raise AdapterError("the stage keys were requested without assembling any panel.")


def build_datasets(cfg: DataAdapterConfig) -> DataSplits:
    """Ingest, prepare, align, featurize and temporally split into train/val/test.

    The single-split entry point. Under ``split.scheme="walk_forward"`` a run has
    more than one fold and there is no single answer to return, so this raises and
    points at :func:`build_folds`.
    """
    return split_panel(assemble_panel(cfg), cfg.split)


def build_folds(cfg: DataAdapterConfig) -> FoldPlan:
    """Ingest, prepare, align, featurize, and cut the configured fold schedule.

    The general entry point: one fold under ``split.scheme="single"``, and one per
    position of the sweep under ``"walk_forward"``.
    """
    return plan_panel(assemble_panel(cfg), cfg.split)


# --------------------------------------------------------------------------- #
# Stage chains
# --------------------------------------------------------------------------- #


def _referenced_panels(cfg: DataAdapterConfig) -> set[str]:
    """The panels the alignment actually reads: the spine, its joins, their link tables.

    Ingesting a source nothing joins would cost a full load on every merged-stage
    miss, and — because the merged key is chained onto every panel key — would also
    invalidate the merged artifact whenever that unused source changed. Both are
    avoidable: the reachable set is right there in the alignment config.
    """
    names = {cfg.alignment.spine}
    for join in cfg.alignment.joins:
        names.add(join.panel)
        if join.link is not None and join.link.table is not None:
            names.add(join.link.table)
    return names


def _warn_unreferenced(cfg: DataAdapterConfig, referenced: set[str]) -> None:
    """Say so when a configured source is never used, rather than silently skipping it."""
    unused = sorted(set(cfg.ingestion.panels) - referenced)
    if unused:
        logger.warning(
            "ingested panel(s) %s are not referenced by the alignment spine or any join, "
            "so they are not loaded; remove them or add a join.",
            unused,
        )


def _panel_chains(cfg: DataAdapterConfig) -> dict[str, list[_Step]]:
    """Build the per-panel stage chains, keys and all, without reading any data."""
    chains: dict[str, list[_Step]] = {}
    for name, spec in cfg.ingestion.panels.items():
        key = stage_key(
            asdict(spec),
            upstream=None,
            sources={path: path for path in source_paths(spec)},
            fingerprint=cfg.cache.fingerprint,  # type: ignore[arg-type]
        )
        chain = [
            _Step(f"ingest[{name}]", key, asdict(spec), _ingest_step(spec)),
        ]
        preprocessing = cfg.preprocessing.panels.get(name)
        if preprocessing is not None:
            key = stage_key(asdict(preprocessing), upstream=key)
            chain.append(
                _Step(
                    f"preprocess[{name}]",
                    key,
                    asdict(preprocessing),
                    _preprocess_step(preprocessing),
                )
            )
        features = cfg.features.panels.get(name)
        if features is not None:
            key = stage_key(asdict(features), upstream=key)
            chain.append(
                _Step(f"features[{name}]", key, asdict(features), _feature_step(features, name))
            )
        chains[name] = chain
    return chains


def _merged_chain(
    cfg: DataAdapterConfig, panel_keys: dict[str, str], prepared: Callable[[], PanelSet]
) -> list[_Step]:
    """Build the alignment and merged-panel stage chain."""
    key = stage_key(asdict(cfg.alignment), upstream=_combined_key(panel_keys))
    chain = [
        _Step(
            f"align[{cfg.alignment.spine}]",
            key,
            asdict(cfg.alignment),
            lambda _previous: merge_panels(prepared(), cfg.alignment),
        )
    ]
    preprocessing = cfg.preprocessing.merged
    if preprocessing.steps:
        key = stage_key(asdict(preprocessing), upstream=key)
        chain.append(
            _Step("preprocess[merged]", key, asdict(preprocessing), _preprocess_step(preprocessing))
        )
    features = cfg.features.merged
    if features.featurizers:
        key = stage_key(asdict(features), upstream=key)
        chain.append(
            _Step("features[merged]", key, asdict(features), _feature_step(features, "merged"))
        )
    return chain


def _ingest_step(spec: Any) -> Callable[[Panel | None], Panel]:
    def run(_previous: Panel | None) -> Panel:
        return load_panel(spec)

    return run


def _preprocess_step(cfg: PreprocessingConfig) -> Callable[[Panel | None], Panel]:
    def run(previous: Panel | None) -> Panel:
        panel = _require_previous(previous)
        pipeline = build_preprocessing_pipeline(cfg)
        return panel.with_frame(pipeline.apply(panel.frame, grain=panel.grain))

    return run


def _feature_step(cfg: FeatureConfig, name: str) -> Callable[[Panel | None], Panel]:
    def run(previous: Panel | None) -> Panel:
        panel = _require_previous(previous)
        panel.require_grain(PanelGrain.PANEL, what=f"the feature stage for panel {name!r}")
        return panel.with_frame(build_feature_pipeline(cfg).apply(panel.frame))

    return run


# --------------------------------------------------------------------------- #
# Cache-aware execution
# --------------------------------------------------------------------------- #


def _resolve(cache: PanelCache, chain: list[_Step]) -> Panel:
    """Run ``chain`` from its deepest cached stage onward.

    Scanning backwards is what makes the cache worth having: the last stage is the
    one most likely to be asked for and the most expensive to rebuild, and a hit
    there means none of the work in front of it — including opening the source
    files — is done at all.
    """
    panel: Panel | None = None
    start = 0
    for position in range(len(chain) - 1, -1, -1):
        cached = cache.load(chain[position].key)
        if cached is not None:
            logger.info("cache hit  %-22s %s", chain[position].label, chain[position].key[:12])
            panel = cached
            start = position + 1
            break

    for step in chain[start:]:
        logger.info("cache miss %-22s %s", step.label, step.key[:12])
        panel = step.run(panel)
        _require_nonempty(panel, step.label)
        cache.store(step.key, panel, spec=step.spec)

    if panel is None:
        raise AdapterError("an empty stage chain produces no panel; configure at least one source.")
    return panel


def _combined_key(keys: dict[str, str]) -> str:
    """One digest standing for the whole set of prepared panels."""
    return hashlib.sha256(canonical_json(dict(sorted(keys.items()))).encode("utf-8")).hexdigest()


def _require_previous(previous: Panel | None) -> Panel:
    if previous is None:
        raise AdapterError("this stage needs the output of the stage before it.")
    return previous


def _require_nonempty(panel: Panel, stage: str) -> Panel:
    """Raise a stage-attributed error if a stage yields nothing."""
    if len(panel.frame) == 0:
        raise AdapterError(f"{stage} produced an empty panel.")
    return panel


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def plan_panel(panel: pd.DataFrame, cfg: SplitConfig) -> FoldPlan:
    """Cut the configured fold schedule over a ``(date, entity)`` panel's dates."""
    if cfg.date_level not in list(panel.index.names):
        raise AdapterError(
            f"cannot split on level {cfg.date_level!r}; the panel index is "
            f"{list(panel.index.names)}."
        )
    dates = pd.Index(sorted(panel.index.get_level_values(cfg.date_level).unique()))
    if len(dates) == 0:
        raise AdapterError("cannot split an empty panel.")
    bounds = plan_folds(dates, cfg)
    return FoldPlan(
        panel=panel,
        dates=dates,
        bounds=bounds,
        cut_dates=_cut_dates(dates, cfg, bounds),
        feature_columns=tuple(str(column) for column in panel.columns),
        date_level=cfg.date_level,
    )


def split_panel(panel: pd.DataFrame, cfg: SplitConfig) -> DataSplits:
    """Split a ``(date, entity)`` panel into train/val/test by date (no overlap).

    Raises when the configured scheme produces more than one fold — see
    :func:`plan_panel` for the schedule, and :class:`FoldPlan` for its frames.
    """
    plan = plan_panel(panel, cfg)
    if len(plan) != 1:
        raise AdapterError(
            f"split.scheme={cfg.scheme!r} produced {len(plan)} folds, and a single "
            "train/val/test split is only defined for one. Use build_folds() / "
            "plan_panel() and iterate the schedule instead."
        )
    return plan.splits(0)


def _cut_dates(
    dates: pd.Index, cfg: SplitConfig, bounds: tuple[FoldBounds, ...]
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """The inclusive train and validation upper bounds reported for each fold.

    A single split reports the two dates it was cut at, which is what it has
    always reported and need not be a date the panel carries. A walk-forward fold
    has no such dates — it is cut at positions — so it reports the last date of
    each segment, which bounds it just as tightly.
    """
    if cfg.scheme != "walk_forward":
        return (resolve_cut_dates(dates[cfg.burn_in :] if cfg.burn_in else dates, cfg),)
    return tuple((_last(dates, fold.train), _last(dates, fold.val)) for fold in bounds)


def _last(dates: pd.Index, span: range) -> pd.Timestamp:
    """The final date a span covers; the one before it starts, when it is empty."""
    position = span.stop - 1 if len(span) else max(span.start - 1, 0)
    return pd.Timestamp(dates[position])
