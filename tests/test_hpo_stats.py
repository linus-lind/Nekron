"""Tests for the tuning-decision arithmetic in ``scripts/hpo_stats.py``.

Everything here runs against fold tables built by hand, so the statistics are
pinned against closed forms rather than against MLflow.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hpo_stats", Path(__file__).resolve().parents[1] / "scripts" / "hpo_stats.py"
)
assert _SPEC is not None and _SPEC.loader is not None
hpo_stats = importlib.util.module_from_spec(_SPEC)
sys.modules["hpo_stats"] = hpo_stats
_SPEC.loader.exec_module(hpo_stats)


def run(
    run_id: str,
    scores: Sequence[float],
    *,
    seed: str = "42",
    config_hash: str = "",
    train_size: int = 1260,
    test_size: int = 252,
    sst: float = 100.0,
) -> Any:
    """One parent run's fold table, with sums consistent with ``scores``.

    ``config_hash`` defaults to the first letter of ``run_id``, so runs named
    ``i*`` and ``c*`` read as two different configurations without every call site
    saying so.
    """
    folds = pd.DataFrame(
        {
            "fold_index": range(len(scores)),
            "status": ["completed"] * len(scores),
            "train_size": [train_size] * len(scores),
            "test_size": [test_size] * len(scores),
            # The held-out window, which decides. The validation window trails it
            # by a constant here, so a test that reads the wrong one is visible.
            "test_total_r2": list(scores),
            "test_sst_total": [sst] * len(scores),
            "test_sse_total": [sst * (1.0 - value) for value in scores],
            "val_total_r2": [value - 0.01 for value in scores],
            "val_sst_total": [sst] * len(scores),
            "val_sse_total": [sst * (1.01 - value) for value in scores],
        }
    )
    return hpo_stats.RunFolds(
        run_id=run_id,
        folds=folds,
        params={
            "train.seed": seed,
            "config_hash": config_hash or run_id[0],
            "data_key": "cd4a5575d1ea",
        },
        tags={"git_commit": "898a363", "git_dirty": "false"},
        metrics={
            "folds/planned": float(len(scores)),
            "folds/failed": 0.0,
            "folds/diverged": 0.0,
            "folds/hit_epoch_cap": 0.0,
        },
    )


# --------------------------------------------------------------------------- #
# Reading one run
# --------------------------------------------------------------------------- #


def test_pooling_is_a_ratio_of_sums_and_not_a_mean_of_ratios() -> None:
    """Folds differ in how much variation they carry; their R-squareds do not average."""
    folds = run("a", [0.10, 0.20]).folds.copy()
    folds.loc[1, ["test_sst_total", "test_sse_total"]] = [900.0, 720.0]  # a much larger fold
    heavy = hpo_stats.RunFolds(run_id="a", folds=folds)

    assert heavy.pooled("test_total_r2") == pytest.approx(1.0 - (90.0 + 720.0) / 1000.0)
    assert heavy.pooled("test_total_r2") != pytest.approx(0.15)


def test_failed_folds_are_excluded_rather_than_counted_as_zero() -> None:
    folds = run("a", [0.10, 0.20, 0.30]).folds.copy()
    folds.loc[1, "status"] = "failed"
    partial = hpo_stats.RunFolds(run_id="a", folds=folds)

    assert list(partial.scores("test_total_r2")) == [0.10, 0.30]


def test_the_window_ratio_is_the_reused_data_term() -> None:
    assert run("a", [0.1], train_size=1260, test_size=252).window_ratio == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def test_the_noise_floor_reports_both_a_parametric_and_a_range_band() -> None:
    """The range band assumes nothing: it is a gap two identical runs actually produced."""
    seeds = [run(f"r{i}", [value] * 4, seed=str(i)) for i, value in enumerate([0.10, 0.12, 0.14])]
    floor = hpo_stats.noise_floor(seeds, "test_total_r2")

    assert floor["pooled_mean"] == pytest.approx(0.12)
    assert floor["sigma_seed"] == pytest.approx(0.02)
    assert floor["range_band"] == pytest.approx(0.04)
    assert floor["degrees_of_freedom"] == 2


def test_a_one_seed_candidate_faces_a_wider_band_than_a_three_seed_one() -> None:
    """Fewer replicates measure the candidate less precisely, so more is demanded of it."""
    seeds = [
        run(f"r{i}", [v] * 4, seed=str(i)) for i, v in enumerate([0.10, 0.12, 0.14, 0.11, 0.13])
    ]
    floor = hpo_stats.noise_floor(seeds, "test_total_r2")

    assert floor["screen_band_1_seed"] > floor["confirm_band_3_seeds"]
    sigma, quantile = floor["sigma_seed"], floor["t_quantile"]
    assert floor["screen_band_1_seed"] == pytest.approx(
        quantile * sigma * math.sqrt(1 / 5 + 1.0), rel=1e-3
    )


def test_a_noise_floor_needs_more_than_one_replicate() -> None:
    with pytest.raises(SystemExit, match="at least two replicates"):
        hpo_stats.noise_floor([run("a", [0.1])], "test_total_r2")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def test_the_corrected_standard_error_exceeds_the_naive_one() -> None:
    """Overlapping training windows make the naive error far too small."""
    incumbent = [run("i", [0.10, 0.12, 0.08, 0.11])]
    candidate = [run("c", [0.13, 0.14, 0.12, 0.15])]
    result = hpo_stats.compare(incumbent, candidate, "test_total_r2", sigma=0.0)

    assert result["se_corrected"] > result["se_naive"]
    assert result["se_corrected"] == pytest.approx(
        result["delta_sd_across_folds"] * math.sqrt(1 / 4 + 0.2), rel=1e-5
    )


def test_a_uniform_improvement_wins_every_fold() -> None:
    incumbent = [run("i", [0.10, 0.12, 0.08, 0.11])]
    candidate = [run("c", [0.11, 0.13, 0.09, 0.12])]
    result = hpo_stats.compare(incumbent, candidate, "test_total_r2", sigma=0.0)

    assert result["fold_win_rate"] == 1.0
    assert result["delta_per_fold_mean"] == pytest.approx(0.01)
    assert result["delta_sd_across_folds"] == pytest.approx(0.0, abs=1e-12)


def test_seeds_are_averaged_within_a_fold_before_folds_are_differenced() -> None:
    """Three seeds over four folds are four paired differences, not twelve."""
    incumbent = [run("i1", [0.10] * 4), run("i2", [0.12] * 4)]
    candidate = [run("c1", [0.14] * 4), run("c2", [0.16] * 4)]
    result = hpo_stats.compare(incumbent, candidate, "test_total_r2", sigma=0.0)

    assert result["folds_compared"] == 4
    assert result["delta_per_fold_mean"] == pytest.approx(0.04)


def test_only_folds_both_sides_completed_are_compared() -> None:
    incumbent = [run("i", [0.10, 0.12, 0.08, 0.11])]
    short = run("c", [0.13, 0.14, 0.12])
    result = hpo_stats.compare(incumbent, [short], "test_total_r2", sigma=0.0)

    assert result["folds_compared"] == 3


def test_a_comparison_over_fewer_than_two_shared_folds_is_refused() -> None:
    with pytest.raises(SystemExit, match="paired comparison needs"):
        hpo_stats.compare([run("i", [0.1])], [run("c", [0.2])], "test_total_r2", sigma=0.0)


def test_the_gap_is_expressed_in_multiples_of_the_seed_noise_when_one_is_given() -> None:
    incumbent = [run("i", [0.10, 0.10, 0.10, 0.10])]
    candidate = [run("c", [0.12, 0.12, 0.12, 0.12])]
    result = hpo_stats.compare(incumbent, candidate, "test_total_r2", sigma=0.01)

    assert result["delta_pooled"] == pytest.approx(0.02)
    assert result["delta_over_sigma"] == pytest.approx(2.0)


def test_no_sigma_means_no_multiple_rather_than_a_division_by_zero() -> None:
    result = hpo_stats.compare(
        [run("i", [0.1, 0.1, 0.1])], [run("c", [0.2, 0.2, 0.2])], "test_total_r2", sigma=0.0
    )

    assert "delta_over_sigma" not in result


# --------------------------------------------------------------------------- #
# Audit and comparability
# --------------------------------------------------------------------------- #


def test_the_audit_block_carries_the_gate_counts_and_the_provenance() -> None:
    """A trial note's admissibility fields are read off the run, never retyped."""
    folds = run("c", [0.1, 0.2]).folds
    scored = hpo_stats.RunFolds(
        run_id="c",
        folds=folds,
        params={"train.seed": "42", "config_hash": "e3fa", "data_key": "cd4a5575d1ea9999"},
        tags={"git_commit": "898a363ff312a3e7", "git_dirty": "false", "gpu_name": "RTX 4090"},
        metrics={
            "folds/planned": 11.0,
            "folds/completed": 11.0,
            "folds/failed": 0.0,
            "folds/hit_epoch_cap": 2.0,
            "fit/fit_seconds_total": 1800.0,
        },
    )
    block = hpo_stats.audit([scored])

    assert block["folds_hit_cap"] == 2.0
    assert block["runtime_min"] == pytest.approx(30.0)
    assert block["data_key"] == "cd4a5575d1ea"
    assert block["git_commit"] == "898a363ff312"
    assert block["device"] == "RTX 4090"


def test_a_noise_floor_refuses_replicates_of_different_configurations() -> None:
    """The spread of two configurations is not the spread of one across seeds."""
    mixed = [run("a", [0.1] * 3, config_hash="one"), run("b", [0.2] * 3, config_hash="two")]

    with pytest.raises(SystemExit, match="do not share a config_hash"):
        hpo_stats.noise_floor(mixed, "test_total_r2")


def test_a_changed_data_key_fails_a_gate() -> None:
    """The comparison would otherwise measure the data change, not the configuration."""
    moved = run("c", [0.13, 0.14, 0.12])
    moved.params["data_key"] = "0000deadbeef"  # type: ignore[index]

    result = hpo_stats.compare([run("i", [0.10, 0.12, 0.08])], [moved], "test_total_r2", sigma=0.0)

    assert result["gates_passed"] is False
    assert any("data_key disagrees" in f for f in result["gate_failures"])


def test_a_dirty_working_tree_fails_a_gate() -> None:
    """A commit recorded from a dirty tree does not describe the code that ran."""
    dirty = run("c", [0.13, 0.14, 0.12])
    dirty.tags["git_dirty"] = "true"  # type: ignore[index]

    result = hpo_stats.compare([run("i", [0.10, 0.12, 0.08])], [dirty], "test_total_r2", sigma=0.0)

    assert result["gate_failures"] == ["git_dirty=true"]


def test_comparing_a_configuration_with_itself_fails_a_gate() -> None:
    """An A/A control is a replicate, not a trial, and must not read as one."""
    result = hpo_stats.compare(
        [run("a1", [0.10, 0.12, 0.08], config_hash="same")],
        [run("a2", [0.11, 0.11, 0.09], config_hash="same")],
        "test_total_r2",
        sigma=0.0,
    )

    assert any("replicate, not a trial" in f for f in result["gate_failures"])


def test_a_failed_or_capped_or_diverged_fold_fails_a_gate() -> None:
    for key, name in (
        ("folds/failed", "folds_failed"),
        ("folds/diverged", "folds_diverged"),
        ("folds/hit_epoch_cap", "folds_hit_cap"),
    ):
        broken = run("c", [0.13, 0.14, 0.12])
        broken.metrics[key] = 2.0  # type: ignore[index]
        result = hpo_stats.compare(
            [run("i", [0.10, 0.12, 0.08])], [broken], "test_total_r2", sigma=0.0
        )
        assert any(f.startswith(name) for f in result["gate_failures"]), name


def test_the_campaign_reference_is_checked_only_when_supplied() -> None:
    """A gate that silently skips when it cannot find its reference is worse than none."""
    pair = ([run("i", [0.10, 0.12, 0.08])], [run("c", [0.13, 0.14, 0.12])])

    assert hpo_stats.compare(*pair, "test_total_r2", sigma=0.0)["gates_passed"] is True

    mismatched = hpo_stats.compare(
        *pair, "test_total_r2", sigma=0.0, expect=hpo_stats.Expected(folds=99)
    )
    assert any("folds_planned != campaign" in f for f in mismatched["gate_failures"])


def test_a_clean_comparison_passes_every_gate() -> None:
    result = hpo_stats.compare(
        [run("i", [0.10, 0.12, 0.08])],
        [run("c", [0.13, 0.14, 0.12])],
        "test_total_r2",
        sigma=0.0,
        expect=hpo_stats.Expected(data_key="cd4a5575d1ea", git_commit="898a363", folds=3),
    )

    assert result["gates_passed"] is True
    assert result["gate_failures"] == []
