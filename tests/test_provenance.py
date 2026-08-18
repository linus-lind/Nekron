"""Tests for :mod:`nekron.provenance`."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from nekron.provenance import (
    HASH_LENGTH,
    config_hash,
    environment_identity,
    git_identity,
    run_provenance,
)


@dataclass
class Train:
    seed: int = 42
    epochs: int = 250
    checkpoint_dir: str = "checkpoints"


@dataclass
class Tracking:
    run_name: str = "baseline"


@dataclass
class Config:
    train: Train = field(default_factory=Train)
    mlflow: Tracking = field(default_factory=Tracking)
    lr: float = 1e-3


IGNORE = ("train.seed", "train.checkpoint_dir", "mlflow")


def test_seed_replicates_of_one_configuration_share_a_hash() -> None:
    """The whole point: replicates have to be findable without remembering them."""
    first = config_hash(Config(train=Train(seed=42)), ignore=IGNORE)
    second = config_hash(Config(train=Train(seed=2047)), ignore=IGNORE)

    assert first == second
    assert len(first) == HASH_LENGTH


def test_a_changed_hyperparameter_moves_the_hash() -> None:
    assert config_hash(Config(lr=1e-3), ignore=IGNORE) != config_hash(
        Config(lr=3e-4), ignore=IGNORE
    )


def test_an_ignored_path_may_name_a_whole_section() -> None:
    """``mlflow`` decides where results are written, not what they are."""
    moved = Config(mlflow=Tracking(run_name="trial-07"))

    assert config_hash(moved, ignore=IGNORE) == config_hash(Config(), ignore=IGNORE)


def test_ignoring_an_absent_path_is_not_an_error() -> None:
    """One ignore list serves configurations that do not share every section."""
    assert config_hash({"lr": 1e-3}, ignore=("train.seed", "mlflow", "a.b.c"))


def test_the_hash_does_not_mutate_the_configuration_it_is_given() -> None:
    payload: dict[str, Any] = {"train": {"seed": 42, "epochs": 250}}
    config_hash(payload, ignore=IGNORE)

    assert payload == {"train": {"seed": 42, "epochs": 250}}


def test_key_order_does_not_change_the_hash() -> None:
    """Canonical JSON sorts keys; two spellings of one configuration are one hash."""
    left = config_hash({"a": 1, "b": {"x": 1, "y": 2}}, ignore=())
    right = config_hash({"b": {"y": 2, "x": 1}, "a": 1}, ignore=())

    assert left == right


def test_git_identity_answers_with_strings_whatever_the_repository_state() -> None:
    """Provenance is recorded on the way into real work; it must never raise."""
    identity = git_identity()

    assert set(identity) == {"git_commit", "git_branch", "git_dirty", "git_untracked"}
    assert all(isinstance(value, str) for value in identity.values())
    if identity["git_commit"]:
        assert identity["git_dirty"] in {"true", "false"}
        assert identity["git_untracked"].isdigit()


def test_environment_identity_records_the_determinism_flags() -> None:
    """A seed-noise measurement taken with autotuning on is measuring the hardware."""
    identity = environment_identity()

    assert identity["python_version"]
    for name in ("cudnn_deterministic", "cudnn_benchmark", "tf32_matmul"):
        assert identity[name] in {"true", "false"}


def test_run_provenance_merges_both_identities() -> None:
    provenance = run_provenance()

    assert "git_dirty" in provenance
    assert "torch_version" in provenance
    assert provenance["pid"].isdigit()


def test_untracked_files_are_counted_rather_than_folded_into_the_dirty_flag() -> None:
    """A repository with unrelated work in progress must not mark every run dirty.

    The two answer different questions: ``git_dirty`` says the commit is a false
    claim about what ran, ``git_untracked`` says there is material on disk the
    commit does not describe.
    """
    identity = git_identity()
    if not identity["git_commit"]:
        pytest.skip("not a git repository")

    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert identity["git_dirty"] == ("true" if tracked.stdout.strip() else "false")
