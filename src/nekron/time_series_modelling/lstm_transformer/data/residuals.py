"""The frozen conditional autoencoder, and the residual series it leaves behind.

A fitted conditional autoencoder reconstructs each period's cross-section as
``r_hat = beta(z) . f``. What it does *not* explain is the residual

    u_{i,t} = r_{i,t} - beta(z_{i,t})' f_t,

one number per stock per period, and it is those series this package models. This
module is the bridge: :func:`load_pretrained_cae` recovers a fitted autoencoder
from its MLflow run, and :func:`build_residual_panel` replays it over every period
of the panel it was fitted on and lays the residuals out as a dense
``[periods, entities]`` matrix.

Units
-----
The autoencoder is fitted on cross-sectionally z-scored returns, so ``r``,
``r_hat`` and therefore ``u`` are all in *standardized* units: a residual of 0.5
is half a cross-sectional standard deviation of that period's returns. Nothing is
rescaled here. That keeps the series comparable across periods of very different
volatility, which is what a fixed-length sequence model needs; recovering raw
return units is a multiplication by the period's cross-sectional return standard
deviation, which the panel would have to be re-read to obtain.

The residuals are *not* mean-zero within a period. The model carries no
intercept, so nothing forces the cross-sectional mean of ``u`` to vanish, and in
practice it does not.

Reproducibility
---------------
Everything needed to rebuild the residuals — the weights, the column order they
were fitted against, and the entire data-pipeline configuration that produced
that panel — travels inside the run's checkpoint artifact. Nothing about the
autoencoder is restated in this package's own configuration, so there is no
second copy to drift. The one thing that is *checked* rather than assumed is the
column order: a feature configuration that has moved on since the fit would
produce a panel whose columns no longer mean what the network's input dimensions
mean, and the model would answer confidently and wrongly. That comparison is the
gate at the top of :func:`build_residual_panel`.

Cost
----
Replaying the autoencoder is seconds; rebuilding the per-period cross-sections it
consumes is minutes, because every feature column is ranked within every date.
The result is therefore cached like any other pipeline stage, keyed on the
weights themselves as well as on the configuration — two runs of the same
architecture produce different residuals, and a key that ignored the weights
would hand the second run the first one's.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import OmegaConf

from nekron.asset_pricing.conditional_autoencoder.config import ConditionalAutoencoderConfig
from nekron.asset_pricing.conditional_autoencoder.data.dataset import build_cross_section_panel
from nekron.asset_pricing.conditional_autoencoder.model import ConditionalAutoencoder
from nekron.cache import PanelCache, build_cache, read_manifest, stage_key
from nekron.data import source_paths
from nekron.data_adapter import assemble_panel
from nekron.data_adapter.config import SplitConfig
from nekron.data_adapter.splitting import rebase_split
from nekron.nn import resolve_device
from nekron.panel import Panel

from ..config import CaeConfig

logger = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float32]

CACHE_STAGE = "lstm_transformer.residuals"
"""Label folded into the cache key, so this artifact cannot collide with a stage."""


class ResidualError(Exception):
    """Raised when a fitted autoencoder cannot be recovered or replayed."""


# --------------------------------------------------------------------------- #
# Loading the frozen model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PretrainedCae:
    """A fitted conditional autoencoder, with everything needed to interpret it.

    :attr:`beta_columns` and :attr:`portfolio_columns` are the column orders the
    networks were fitted against — input dimension ``p`` means "the p-th of these
    names" and nothing else — and :attr:`config` is the configuration that
    produced them, so the panel can be rebuilt exactly.
    """

    model: ConditionalAutoencoder
    config: ConditionalAutoencoderConfig
    beta_columns: tuple[str, ...]
    portfolio_columns: tuple[str, ...]
    run_id: str
    weights_digest: str
    best_epoch: int
    best_metric: float

    def describe(self) -> str:
        """One line naming the run and the fit it recovered."""
        return (
            f"cae run {self.run_id[:8]} -> factors={self.config.model.num_factors} "
            f"beta_columns={len(self.beta_columns)} portfolios={len(self.portfolio_columns)} "
            f"best_epoch={self.best_epoch} best_metric={self.best_metric:.4f}"
        )


def load_pretrained_cae(cfg: CaeConfig) -> PretrainedCae:
    """Recover the fitted autoencoder logged under ``cfg.run_id``.

    The checkpoint artifact is preferred over the run's logged *model* on purpose.
    MLflow pickles the whole ``nn.Module``, which ties the artifact to the module
    path it was defined at and to the device it was pickled on, and which carries
    neither the column order nor the data configuration. The checkpoint carries
    all three, and rebuilding the architecture from the recorded config and
    loading a state dict into it is both version-robust and device-clean.
    """
    import mlflow

    mlflow.set_tracking_uri(cfg.tracking_uri)
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=cfg.run_id, artifact_path=cfg.checkpoint_artifact
        )
    except Exception as exc:
        raise ResidualError(
            f"could not download {cfg.checkpoint_artifact!r} from MLflow run {cfg.run_id!r} at "
            f"{cfg.tracking_uri!r}: {exc}. The run must be a conditional-autoencoder fit whose "
            "trainer logged its checkpoint."
        ) from exc

    payload = _load_checkpoint(path)
    for key in ("model_state", "config", "beta_columns"):
        if key not in payload:
            raise ResidualError(
                f"checkpoint {cfg.checkpoint_artifact!r} of run {cfg.run_id!r} has no "
                f"{key!r}; it was not written by this project's conditional-autoencoder "
                f"trainer (keys present: {sorted(payload)})."
            )

    ca_cfg = _typed_cae_config(payload["config"], cfg.run_id)
    beta_columns = tuple(str(name) for name in payload["beta_columns"])
    if not beta_columns:
        raise ResidualError(f"run {cfg.run_id!r} recorded no beta_columns; it cannot be replayed.")
    # Older checkpoints predate the portfolio_columns field. It is exactly
    # recoverable: the resolver returns the configured list verbatim when it is
    # non-empty, and falls back to the beta columns when it is not.
    recorded = payload.get("portfolio_columns")
    portfolio_columns = tuple(
        str(name) for name in (recorded or ca_cfg.data.portfolio_columns or beta_columns)
    )

    state = cast(Mapping[str, torch.Tensor], payload["model_state"])
    model = ConditionalAutoencoder.from_config(
        ca_cfg,
        num_beta_columns=len(beta_columns),
        num_portfolios=len(portfolio_columns),
    )
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise ResidualError(
            f"the checkpoint of run {cfg.run_id!r} does not fit the architecture its own "
            f"config describes: {exc}"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return PretrainedCae(
        model=model,
        config=ca_cfg,
        beta_columns=beta_columns,
        portfolio_columns=portfolio_columns,
        run_id=cfg.run_id,
        weights_digest=_weights_digest(state),
        best_epoch=int(payload.get("best_epoch", -1)),
        best_metric=float(payload.get("best_metric", float("nan"))),
    )


def _load_checkpoint(path: str) -> dict[str, Any]:
    """Read a checkpoint, preferring torch's restricted unpickler.

    ``weights_only=True`` refuses to execute arbitrary code while unpickling and
    covers this payload, whose non-tensor entries are plain containers. It is
    tried first and fallen back on rather than skipped, so the day someone stores
    a richer object in the checkpoint the run keeps working and says why.
    """
    try:
        return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    except Exception as exc:  # noqa: BLE001 - any restriction failure falls back below
        logger.warning(
            "checkpoint %s could not be read under weights_only=True (%s); falling back to the "
            "unrestricted unpickler. Only do this for artifacts you produced.",
            path,
            exc,
        )
        return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def _typed_cae_config(saved: Any, run_id: str) -> ConditionalAutoencoderConfig:
    """Merge a checkpoint's config dict back onto the live schema.

    Merging rather than reconstructing is what absorbs schema drift in the useful
    direction: a field added since the fit takes its default, while a field the
    checkpoint carries overrides it, and every ``__post_init__`` still runs.
    """
    try:
        merged = OmegaConf.merge(OmegaConf.structured(ConditionalAutoencoderConfig), saved)
        return cast(ConditionalAutoencoderConfig, OmegaConf.to_object(merged))
    except Exception as exc:
        raise ResidualError(
            f"the config recorded by run {run_id!r} no longer validates against "
            f"ConditionalAutoencoderConfig: {exc}"
        ) from exc


def _weights_digest(state: Mapping[str, torch.Tensor]) -> str:
    """A digest of the fitted parameters, in a fixed key order.

    Part of the residual cache key. Two runs of an identical architecture on an
    identical panel differ only in their weights, and a key that ignored them
    would serve the first run's residuals to the second.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# The residual panel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidualPanel:
    """Every residual the frozen autoencoder produces, as a dense matrix.

    Parameters
    ----------
    dates:
        The periods that produced a cross-section, ascending. Not the panel's date
        index: a period too thin to price is dropped when the cross-sections are
        built, so this is shorter, and any schedule cut over these residuals must
        be cut over *this* sequence.
    entities:
        The union of entities that appear in any period, lexicographically sorted.
    values:
        ``[len(dates), len(entities)]`` residuals, ``NaN`` where the entity has no
        cross-section entry for that date. The gaps are left as ``NaN`` rather
        than filled: a window is built only where the whole span is present, so a
        ``NaN`` reaching a model would mean the window index is wrong, and it is
        far better for that to be loud.
    split:
        The autoencoder run's own schedule, with a fraction-based single split
        already resolved to timestamps against the *full* panel — which is what
        makes this model's boundaries land exactly where the autoencoder's did.
    """

    dates: pd.DatetimeIndex
    entities: tuple[str, ...]
    values: FloatMatrix
    split: SplitConfig

    @property
    def num_periods(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_entities(self) -> int:
        return int(self.values.shape[1])

    @property
    def observed(self) -> int:
        """Number of ``(date, entity)`` cells that carry a residual."""
        return int(np.isfinite(self.values).sum())

    def describe(self) -> str:
        """One line naming the shape and how much of it is populated."""
        cells = self.num_periods * self.num_entities
        density = self.observed / cells if cells else float("nan")
        return (
            f"residuals -> periods={self.num_periods} "
            f"({self.dates[0].date()}..{self.dates[-1].date()}) "
            f"entities={self.num_entities} observed={self.observed:,} ({density:.1%} dense)"
        )


def build_residual_panel(cae: PretrainedCae, device: str) -> ResidualPanel:
    """Replay ``cae`` over the panel it was fitted on and collect its residuals.

    Reads the cache first and writes it on the way out; the key covers the fitted
    weights, the autoencoder's data configuration and the identity of every source
    file the pipeline reads, so anything that would change a residual changes the
    key.
    """
    cache = build_cache(cae.config.data_pipeline.cache)
    key = _residual_key(cae)
    cached = _load_cached(cache, key)
    if cached is not None:
        logger.info("cache hit  %-22s %s", CACHE_STAGE, key[:12])
        return cached

    logger.info("cache miss %-22s %s", CACHE_STAGE, key[:12])
    panel = assemble_panel(cae.config.data_pipeline)
    split = cae.config.data_pipeline.split
    panel_dates = pd.Index(sorted(panel.index.get_level_values(split.date_level).unique()))
    sections = build_cross_section_panel(panel, cae.config.data)
    # Re-expressed in the units of the sequence that will actually be cut. Two of
    # the split's settings do not survive the change of sequence: a fraction-based
    # cut date resolves differently over a subsequence than over the panel, and
    # ``burn_in`` is a count of *panel* dates, so applying that same integer to the
    # shorter residual sequence would discard more history than was asked for.
    # ``rebase_split`` converts both, which is why what this panel carries is
    # already in its own units and the windowing does not rebase it a second time.
    rebased = rebase_split(split, panel_dates, sections.dates)
    _require_matching_columns(cae, sections.beta_columns, sections.portfolio_columns)
    dropped = len(panel_dates) - len(sections.dates)
    if dropped:
        logger.info(
            "%d of %d periods produced no cross-section (data.min_cross_section=%d) and carry "
            "no residual.",
            dropped,
            len(panel_dates),
            cae.config.data.min_cross_section,
        )

    residuals = _replay(cae, sections.sections, device=device)
    built = ResidualPanel(
        dates=pd.DatetimeIndex(sections.dates, name=split.date_level),
        entities=residuals.entities,
        values=residuals.values,
        split=rebased,
    )
    _store_cached(cache, key, built, cae)
    return built


@dataclass(frozen=True)
class _Replayed:
    """The dense residual matrix and the entity axis it was laid out on."""

    entities: tuple[str, ...]
    values: FloatMatrix


@torch.no_grad()
def _replay(cae: PretrainedCae, sections: tuple[Any, ...], *, device: str) -> _Replayed:
    """Run the frozen model over every period and scatter the residuals into a matrix.

    The entity axis is built first, from a pass that touches no tensors, so the
    matrix can be allocated once at its final size instead of being grown or
    assembled from a long frame with one row per observation — which for a daily
    panel of a thousand stocks over fifteen years is several million rows and an
    object array of as many strings.
    """
    entities = tuple(sorted({entity for section in sections for entity in section.entities}))
    position = {entity: index for index, entity in enumerate(entities)}
    values = np.full((len(sections), len(entities)), np.nan, dtype=np.float32)

    resolved = resolve_device(device)
    model = cae.model.to(resolved)
    for row, section in enumerate(sections):
        output = model(
            section.beta_inputs.to(resolved),
            section.portfolios.to(resolved),
        )
        residual = section.returns.to(resolved) - output.fitted_returns
        columns = np.fromiter(
            (position[entity] for entity in section.entities),
            dtype=np.int64,
            count=len(section.entities),
        )
        values[row, columns] = residual.cpu().numpy().astype(np.float32)
    model.to("cpu")
    return _Replayed(entities=entities, values=values)


def _require_matching_columns(
    cae: PretrainedCae, beta_columns: tuple[str, ...], portfolio_columns: tuple[str, ...]
) -> None:
    """Refuse to replay a model against a panel whose columns have moved.

    Both orders must match exactly, not merely as sets: the networks' input widths
    are inferred from the data, so dimension ``p`` is defined only by position. A
    permuted panel produces residuals that look entirely plausible and mean
    nothing.
    """
    for role, fitted, rebuilt in (
        ("beta", cae.beta_columns, beta_columns),
        ("portfolio", cae.portfolio_columns, portfolio_columns),
    ):
        if fitted == rebuilt:
            continue
        missing = sorted(set(fitted) - set(rebuilt))
        added = sorted(set(rebuilt) - set(fitted))
        detail = (
            f"missing={missing} unexpected={added}"
            if (missing or added)
            else "the same names in a different order"
        )
        raise ResidualError(
            f"run {cae.run_id!r} was fitted on {len(fitted)} {role} columns but the rebuilt "
            f"panel has {len(rebuilt)}: {detail}. The feature configuration has changed since "
            "the fit, so the network's inputs no longer mean what they meant then; refit the "
            "autoencoder or pin the feature config back."
        )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def _residual_key(cae: PretrainedCae) -> str:
    """Digest everything a residual depends on: weights, config and source files."""
    sources: dict[str, str] = {}
    for name, spec in cae.config.data_pipeline.ingestion.panels.items():
        for path in source_paths(spec):
            sources[f"{name}:{path}"] = path
    return stage_key(
        {
            "stage": CACHE_STAGE,
            "weights": cae.weights_digest,
            "cae_data": asdict(cae.config.data),
            "pipeline": asdict(cae.config.data_pipeline),
            "beta_columns": list(cae.beta_columns),
            "portfolio_columns": list(cae.portfolio_columns),
        },
        sources=sources,
        fingerprint=cae.config.data_pipeline.cache.fingerprint,  # type: ignore[arg-type]
    )


def _load_cached(cache: PanelCache, key: str) -> ResidualPanel | None:
    """Read a cached residual matrix, or ``None`` on any kind of miss."""
    panel = cache.load(key)
    if panel is None:
        return None
    try:
        spec = read_manifest(cache.path_for(key)).get("spec") or {}
        saved_split = spec["split"]
        split = cast(
            SplitConfig,
            OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SplitConfig), saved_split)),
        )
    except Exception:  # noqa: BLE001 - a manifest we cannot read is simply a miss
        logger.warning("cached residuals %s carry no usable split; rebuilding.", key[:12])
        return None
    frame = panel.frame
    return ResidualPanel(
        dates=pd.DatetimeIndex(frame.index),
        entities=tuple(str(column) for column in frame.columns),
        values=frame.to_numpy(dtype=np.float32, copy=False),
        split=split,
    )


def _store_cached(
    cache: PanelCache, key: str, residuals: ResidualPanel, cae: PretrainedCae
) -> None:
    """Write the residual matrix, recording the split it was resolved with.

    Stored wide — one row per date, one column per entity — because that is the
    shape the windowing reads. A long frame would cost a pivot on every load and
    repeat the two keys on every one of several million rows.
    """
    frame = pd.DataFrame(residuals.values, index=residuals.dates, columns=list(residuals.entities))
    cache.store(
        key,
        Panel(frame=frame, date_name=str(residuals.dates.name), entity_name=None),
        spec={
            "stage": CACHE_STAGE,
            "run_id": cae.run_id,
            "weights": cae.weights_digest,
            # The resolved schedule: recovering it here is what lets a cache hit
            # skip rebuilding the panel it was resolved against.
            "split": asdict(residuals.split),
        },
    )
