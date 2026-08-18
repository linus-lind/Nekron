"""Tests for :mod:`nekron.tracking`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from nekron.provenance import config_hash
from nekron.tracking import (
    CONFIG_FILE,
    CONFIG_HASH_IGNORE,
    CONFIG_HASH_PARAM,
    MAX_PARAM_VALUE,
    MlflowConfig,
    MlflowTracker,
    _flatten,
)


@dataclass
class Leaf:
    type: str = "rolling_aggregation"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Root:
    name: str = "run"
    depth: int = 3
    windows: list[int] = field(default_factory=lambda: [1, 5, 21])
    steps: list[Leaf] = field(default_factory=lambda: [Leaf(), Leaf(type="rsi")])


def test_mlflow_is_not_imported_just_by_importing_the_module() -> None:
    """Importing a plain config dataclass must not pull in mlflow's dependency tree."""
    source = Path("src/nekron/tracking.py").read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines() if line.startswith(("import mlflow", "from mlflow"))
    ]
    assert module_level == []


def test_flatten_recurses_into_a_list_of_structures() -> None:
    """Joining these instead is what produced a param value MLflow refuses.

    A list of scalars is still one comma-joined value — that is readable and
    short — but a list of featurizer specs has to become one param per field.
    """
    flat: dict[str, Any] = {}
    _flatten("", {"windows": [1, 5, 21], "steps": [{"type": "a"}, {"type": "b"}]}, flat)

    assert flat["windows"] == "1,5,21"
    assert flat["steps.0.type"] == "a"
    assert flat["steps.1.type"] == "b"


def test_flatten_produces_names_mlflow_accepts() -> None:
    """Param names may not contain brackets, so list indices are dotted."""
    flat: dict[str, Any] = {}
    _flatten("", {"steps": [{"params": {"window": 5}}]}, flat)

    assert "steps.0.params.window" in flat
    assert not any("[" in name or "]" in name for name in flat)


def test_flatten_truncates_an_over_long_value() -> None:
    flat: dict[str, Any] = {}
    _flatten("", {"huge": ["x" * 100] * 200}, flat)

    value = flat["huge"]
    assert len(value) == MAX_PARAM_VALUE
    assert value.endswith("...")


def test_disabled_tracker_is_inert(tmp_path: Path) -> None:
    """Training code calls the tracker unconditionally, so disabled must be a no-op."""
    with MlflowTracker(MlflowConfig(enabled=False)) as tracker:
        tracker.log_config(Root())
        tracker.log_metrics({"loss": 1.0}, step=0)
        tracker.log_model(object())

    assert not list(tmp_path.iterdir())


def test_a_nested_config_reaches_a_real_mlflow_run(tmp_path: Path) -> None:
    """End to end against a real store: the whole config must be loggable.

    ``log_config`` is a run's first tracked call, i.e. after the data pipeline has
    already run, so a rejected param does not fail fast — it throws away the
    expensive part of the job.
    """
    mlflow = pytest.importorskip("mlflow", reason="mlflow is required by the tracker")
    database = tmp_path / "mlflow.db"
    config = MlflowConfig(
        enabled=True, tracking_uri=f"sqlite:///{database}", experiment_name="test", log_model=False
    )

    with MlflowTracker(config) as tracker:
        tracker.log_config(Root())
        tracker.log_metrics({"kept": 0.5, "dropped": float("nan")}, step=0)
        run_id = mlflow.active_run().info.run_id

    mlflow.set_tracking_uri(f"sqlite:///{database}")
    run = mlflow.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["steps.1.type"] == "rsi"
    assert run.data.params["windows"] == "1,5,21"
    assert run.data.metrics == {"kept": 0.5}


def test_log_config_records_a_hash_that_ignores_the_seed_and_the_tracking_section() -> None:
    """Seed replicates of one configuration have to be findable by query."""

    @dataclass
    class Seeded:
        seed: int = 42
        lr: float = 1e-3

    @dataclass
    class Root2:
        train: Seeded = field(default_factory=Seeded)
        mlflow: dict[str, Any] = field(default_factory=lambda: {"run_name": "baseline"})

    same = config_hash(asdict(Root2(train=Seeded(seed=7))), ignore=CONFIG_HASH_IGNORE)
    base = config_hash(asdict(Root2()), ignore=CONFIG_HASH_IGNORE)
    moved = config_hash(asdict(Root2(train=Seeded(lr=3e-4))), ignore=CONFIG_HASH_IGNORE)

    assert same == base
    assert moved != base


def test_a_run_carries_its_provenance_and_its_configured_tags(tmp_path: Path) -> None:
    """Every run says which commit, which machine and which campaign produced it."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow is required by the tracker")
    database = tmp_path / "mlflow.db"
    config = MlflowConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{database}",
        experiment_name="test",
        log_model=False,
        tags={"campaign": "cae-v1", "trial": "T07"},
    )

    with MlflowTracker(config, tags={"role": "screen"}) as tracker:
        tracker.log_config(Root())
        run_id = mlflow.active_run().info.run_id

    mlflow.set_tracking_uri(f"sqlite:///{database}")
    run = mlflow.get_run(run_id)

    assert run.data.tags["campaign"] == "cae-v1"
    assert run.data.tags["trial"] == "T07"
    assert run.data.tags["role"] == "screen"
    assert run.data.tags["git_dirty"] in {"true", "false", ""}
    assert run.data.tags["cudnn_benchmark"] in {"true", "false"}
    assert run.data.params[CONFIG_HASH_PARAM]
    assert CONFIG_FILE in {f.path for f in mlflow.MlflowClient().list_artifacts(run_id)}
