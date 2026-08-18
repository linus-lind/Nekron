"""Compose every Hydra config group so a broken YAML fails fast.

The stage configs are only validated when something composes them, which in normal
use is a training run. This makes that check a first-class, dependency-light step:
it exercises the real ``configs/`` tree against the real structured schemas without
importing the model packages (and therefore without needing torch or mlflow).
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from nekron.align.registry import build_aligner
from nekron.data.config import PanelSpec, to_schema, to_selection
from nekron.data.filters import build_filter
from nekron.data.registry import build_source
from nekron.data_adapter.config import DataAdapterConfig, register_configs
from nekron.features.config import build_pipeline as build_feature_pipeline
from nekron.preprocessing.config import build_pipeline as build_preprocessing_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

# Each case is (label, overrides) mounting real group files at their real paths.
CASES: list[tuple[str, list[str]]] = [
    (
        "crsp single-panel",
        [
            "+data@ingestion.panels.crsp=crsp",
            "+preprocessing@preprocessing.panels.crsp=crsp",
            "+features@features.merged=crsp",
            "+alignment@alignment=crsp",
        ],
    ),
    (
        "multi-panel example",
        [
            "+data@ingestion.panels.crsp=crsp",
            "+data@ingestion.panels.compustat=example_compustat",
            "+data@ingestion.panels.ff=example_ff_factors",
            "+data@ingestion.panels.sic=example_sic",
            "+data@ingestion.panels.ccm=example_ccm_link",
            "+alignment@alignment=example_multi_panel",
        ],
    ),
]


def deep_check(cfg: DataAdapterConfig) -> None:
    """Resolve every registered ``type:`` string the config names.

    Composing proves the YAML *shape* is right, which is the cheap half. It says
    nothing about whether ``type: rolling_aggreagtion`` names a real featurizer —
    and that typo would otherwise surface only after every panel had been ingested
    and preprocessed, minutes into a run. Everything here is constructed without
    reading a single row of data.
    """
    for name, spec in cfg.ingestion.panels.items():
        _check_panel(name, spec)
    for name, item in cfg.preprocessing.panels.items():
        build_preprocessing_pipeline(item)
        del name
    build_preprocessing_pipeline(cfg.preprocessing.merged)
    for name, feature in cfg.features.panels.items():
        build_feature_pipeline(feature)
        del name
    build_feature_pipeline(cfg.features.merged)
    for join in cfg.alignment.joins:
        build_aligner(join.aligner.type, join.aligner.params)
        if join.link is not None and join.link.mode not in ("table", "spine", "identity"):
            raise ValueError(f"join on {join.panel!r} has unknown link mode {join.link.mode!r}.")
    _check_alignment_names(cfg)


def _check_panel(name: str, spec: PanelSpec) -> None:
    """Build one source's schema, selection, filters and backend."""
    schema = to_schema(spec.schema)
    selection = to_selection(spec.selection)
    filters = tuple(build_filter(item.type, item.params) for item in spec.filters)
    build_source(spec.source, schema, selection, ())
    del name, filters


def _check_alignment_names(cfg: DataAdapterConfig) -> None:
    """Every panel the alignment refers to must actually be configured."""
    known = set(cfg.ingestion.panels)
    referenced = {cfg.alignment.spine}
    for join in cfg.alignment.joins:
        referenced.add(join.panel)
        if join.link is not None and join.link.table is not None:
            referenced.add(join.link.table)
    missing = sorted(referenced - known)
    if missing:
        raise ValueError(f"alignment refers to panels that are not configured: {missing}.")


def check_model_configs() -> int:
    """Validate each model config's own ``data_pipeline`` block against the schema.

    The group files are checked above, but a model config also *overrides* them
    inline, and that block is the part a renesting breaks. Merging it against
    :class:`DataAdapterConfig` on its own catches a stale path without importing
    the model package — which would drag in torch and mlflow.
    """
    failures = 0
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = OmegaConf.load(path)
        block = raw.get("data_pipeline") if isinstance(raw, DictConfig) else None
        if block is None:
            continue
        try:
            OmegaConf.merge(OmegaConf.structured(DataAdapterConfig), block)
        except Exception as exc:  # noqa: BLE001 - report every file, not just the first
            failures += 1
            print(f"FAIL {path.name} data_pipeline: {type(exc).__name__}: {exc}")
            continue
        print(f"ok   {path.name} data_pipeline block validates")
    return failures


def main() -> int:
    register_configs()
    ConfigStore.instance().store(name="_config_check", node=DataAdapterConfig)
    failures = check_model_configs()
    for label, overrides in CASES:
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
                cfg = compose(config_name="_config_check", overrides=overrides)
                resolved = OmegaConf.to_object(cfg)
        except Exception as exc:  # noqa: BLE001 - report every case, not just the first
            failures += 1
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            continue
        assert isinstance(resolved, DataAdapterConfig)
        try:
            deep_check(resolved)
        except Exception as exc:  # noqa: BLE001 - report every case, not just the first
            failures += 1
            print(f"FAIL {label} (resolving registered types): {type(exc).__name__}: {exc}")
            continue
        print(
            f"ok   {label}: {len(resolved.ingestion.panels)} panel(s), "
            f"spine={resolved.alignment.spine!r}, {len(resolved.alignment.joins)} join(s), "
            "all registered types resolve"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
