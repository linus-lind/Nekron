"""Turn a set of MLflow runs into the numbers a tuning decision is made from.

Two questions, one for each half of a tuning campaign.

``noise``
    How far apart do replicates of *one* configuration land when only the seed
    differs? That spread is the floor below which no comparison means anything,
    and every acceptance threshold in the campaign is a multiple of it.

``compare``
    Is a candidate configuration better than the incumbent by more than that
    floor, consistently across folds rather than on the strength of one, and did
    the fits behind it actually converge?

Both read the ``folds.parquet`` artifact each parent run writes, so they see the
per-fold numbers rather than the aggregate, which is what a paired comparison
needs. Neither writes anything: the output is a block of YAML meant to be pasted
straight into a trial note's frontmatter, so the record and the arithmetic behind
it cannot drift apart, and the only fields left to type by hand are the ones a
person actually decides — which parameter moved, and why.

Why the comparison is paired and corrected
------------------------------------------
Fold scores from one run are not independent draws. Walk-forward training windows
overlap between adjacent folds — entirely, under an expanding window — and market
returns are regime-persistent, so a naive standard error over folds is far too
small and a t-statistic built on it is far too large. Two corrections apply, and
both are cheap:

*Pair the folds.* Compare fold ``k`` of the candidate with fold ``k`` of the
incumbent, not the two means. Whatever made fold 7 hard is then differenced away.

*Inflate the standard error.* Nadeau and Bengio's correction for reused training
data replaces ``s/sqrt(K)`` with ``s * sqrt(1/K + n_test/n_train)``. With a
five-year train window and a one-year test window that is more than twice the
naive figure — which is the difference between a threshold that admits noise and
one that does not.

Usage
-----
::

    python scripts/hpo_stats.py noise --experiment conditional_autoencoder \\
        --config-hash 23869ae4670a
    python scripts/hpo_stats.py compare --incumbent <run-id> --candidate <run-id> \\
        --sigma 0.00042
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SUMMARY_FILE = "folds.parquet"
"""The per-fold table a sweep attaches to its parent run."""

POOLED_SUMS = {
    "test_total_r2": ("test_sse_total", "test_sst_total"),
    "test_predictive_r2": ("test_sse_pred", "test_sst_pred"),
    "val_total_r2": ("val_sse_total", "val_sst_total"),
    "val_predictive_r2": ("val_sse_pred", "val_sst_pred"),
}
"""Which summary columns a pooled statistic is reassembled from, per fold column.

A pooled R-squared is a ratio of summed errors to summed totals over every scored
period. It cannot be recovered from the folds' own ratios, so the sums travel in
the summary table and are re-summed here.

The ``test_*`` columns are the fold's held-out window — what a configuration is
compared with another on. The ``val_*`` columns are the window early stopping
chose the epoch on, and are available for diagnosis rather than for decisions: a
validation score far above the held-out one says the epoch was chosen on noise.
"""

DEFAULT_COLUMN = "test_total_r2"
"""The per-fold column decisions are made on unless told otherwise."""

GATE_METRICS = {
    "folds_planned": "folds/planned",
    "folds_completed": "folds/completed",
    "folds_failed": "folds/failed",
    "folds_hit_cap": "folds/hit_epoch_cap",
    "folds_diverged": "folds/diverged",
}
"""Run-level metrics that decide whether a run's number is admissible at all.

Admissibility comes before quality. A sweep with a failed fold is scored on a
different sample from the one it is compared against; a sweep whose fits ran out
of epochs is a statement about the budget as much as about the configuration; a
sweep that diverged has no number. None of the three is visible in the score.
"""


def _sig(value: float) -> float:
    """``value`` to six significant figures.

    Significant figures rather than decimal places: the quantities compared here
    span several orders of magnitude — a pooled R-squared near 0.03 and a
    difference near 0.00004 — and a fixed number of decimals would either bury the
    first in noise or round the second away entirely.
    """
    return float(f"{value:.6g}")


def t_quantile(degrees: int) -> float:
    """The one-sided 95% Student-t quantile at ``degrees`` degrees of freedom.

    Student-t and not a normal quantile, because the standard deviation the
    threshold is built from is itself estimated from a handful of seeds. At the
    four degrees of freedom five replicates give, the quantile is 2.13 against the
    1.64 a normal approximation would use — a 30% wider band, and the honest one:
    the 95% interval for a standard deviation on four degrees of freedom runs from
    0.60 to 2.87 times the estimate.
    """
    from scipy import stats

    return float(stats.t.ppf(0.95, max(degrees, 1)))


@dataclass(frozen=True)
class RunFolds:
    """One parent run: its per-fold table, and what identifies the run itself."""

    run_id: str
    folds: pd.DataFrame
    params: Mapping[str, str] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def seed(self) -> str:
        return self.params.get("train.seed", "")

    @property
    def completed(self) -> pd.DataFrame:
        """The folds that ran to completion; a failed one is excluded, not zeroed."""
        return self.folds.loc[self.folds["status"] == "completed"]

    def scores(self, column: str) -> pd.Series:
        """The named column, indexed by fold, over the folds that completed."""
        done = self.completed
        return pd.Series(
            done[column].astype("float64").to_numpy(),
            index=done["fold_index"].astype(int).to_numpy(),
            name=self.run_id,
        )

    def pooled(self, column: str) -> float:
        """The column's pooled value over every scored period of every fold."""
        sums = POOLED_SUMS.get(column)
        if sums is None or not set(sums) <= set(self.folds.columns):
            return float("nan")
        sse, sst = (float(self.completed[name].sum()) for name in sums)
        return 1.0 - sse / sst if sst > 0 else float("nan")

    @property
    def window_ratio(self) -> float:
        """``n_test / n_train``, the reused-data term of the corrected error."""
        done = self.completed
        if not len(done) or "train_size" not in done.columns:
            return 0.0
        return float(done["test_size"].mean()) / float(done["train_size"].mean())


# --------------------------------------------------------------------------- #
# Reading runs
# --------------------------------------------------------------------------- #


def load_runs(run_ids: Sequence[str], tracking_uri: str) -> list[RunFolds]:
    """Fetch each run's fold table and identity, in the order the ids were given."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    loaded = []
    for run_id in run_ids:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=SUMMARY_FILE)
        data = client.get_run(run_id).data
        loaded.append(
            RunFolds(
                run_id=run_id,
                folds=pd.read_parquet(path),
                params=dict(data.params),
                tags=dict(data.tags),
                metrics=dict(data.metrics),
            )
        )
    return loaded


PARENT_TAG = "tags.mlflow.parentRunId"
"""Column MLflow returns for a nested run's parent; absent or null on a parent."""


def find_runs(
    experiment: str,
    tracking_uri: str,
    *,
    config_hash: str,
    campaign: str,
    trial: str,
) -> list[str]:
    """Every finished parent run matching the given filters, oldest first.

    Parent runs only. A fold's nested run repeats its parent's configuration hash
    and inherits its campaign tags, so counting folds as replicates would report
    the spread *across folds* as if it were the spread across seeds — a number
    that is both larger and about something else entirely.

    The parent filter is applied client-side on purpose. A search expression of
    the form ``tags.mlflow.parentRunId = ''`` matches nothing rather than matching
    the runs that have no such tag, so a server-side version of this silently
    returns an empty list instead of the parents it was asked for.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    clauses = ["attributes.status = 'FINISHED'"]
    if config_hash:
        clauses.append(f"params.config_hash = '{config_hash}'")
    if campaign:
        clauses.append(f"tags.campaign = '{campaign}'")
    if trial:
        clauses.append(f"tags.trial = '{trial}'")
    frame = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=" and ".join(clauses),
        output_format="pandas",
        order_by=["attributes.start_time ASC"],
    )
    return [] if frame.empty else list(parents_only(frame)["run_id"])


def parents_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of a search result that are not nested under another run."""
    if PARENT_TAG not in frame.columns:
        return frame
    return frame.loc[frame[PARENT_TAG].isna()]


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def noise_floor(runs: Sequence[RunFolds], column: str) -> dict[str, Any]:
    """The spread of one configuration's headline number across its seeds.

    Two bands are reported because they fail in opposite directions. The
    *parametric* band assumes the pooled scores are roughly normal and pays for
    the standard deviation being estimated from a handful of seeds by using a
    Student-t quantile. The *range* band assumes nothing at all: it is simply the
    largest gap any two of these identical runs actually produced, and no
    difference smaller than a gap already observed between two copies of the same
    thing is worth acting on.
    """
    pooled = [run.pooled(column) for run in runs]
    replicates = len(pooled)
    if replicates < 2:
        raise SystemExit("a noise floor needs at least two replicates of one configuration.")
    hashes = {run.params.get("config_hash", "") for run in runs}
    if len(hashes) > 1:
        raise SystemExit(
            "these runs do not share a config_hash; a noise floor is the spread of ONE "
            "configuration across seeds, and mixing configurations measures something else."
        )

    sigma = statistics.stdev(pooled)
    degrees = replicates - 1
    quantile = t_quantile(degrees)
    return {
        "metric": column,
        "pooled": [_sig(value) for value in pooled],
        "pooled_mean": _sig(statistics.fmean(pooled)),
        "sigma_seed": _sig(sigma),
        "range_band": _sig(max(pooled) - min(pooled)),
        "per_fold_mean": _sig(statistics.fmean(float(r.scores(column).mean()) for r in runs)),
        # What a difference has to clear, given that a candidate is measured with
        # fewer replicates than the baseline was: the standard error of a
        # difference of means is sigma * sqrt(1/n_base + 1/n_cand), and the
        # quantile carries the uncertainty in sigma itself.
        "screen_band_1_seed": _sig(quantile * sigma * math.sqrt(1 / replicates + 1.0)),
        "confirm_band_3_seeds": _sig(quantile * sigma * math.sqrt(1 / replicates + 1 / 3)),
        "t_quantile": _sig(quantile),
        "degrees_of_freedom": degrees,
        "gates_passed": not _noise_failures(runs),
        "gate_failures": _noise_failures(runs),
        **audit(runs),
    }


def _noise_failures(runs: Sequence[RunFolds]) -> list[str]:
    """The gates a noise floor must clear before its spread is usable as a threshold."""
    return evaluate_gates([], runs, Expected())


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def compare(
    incumbent: Sequence[RunFolds],
    candidate: Sequence[RunFolds],
    column: str,
    sigma: float,
    expect: Expected | None = None,
) -> dict[str, Any]:
    """The paired, dependence-corrected difference between two configurations.

    Seeds are averaged within a fold before folds are differenced, so seed noise
    enters the fold spread instead of inflating the number of independent
    observations: three seeds over eleven folds are eleven paired differences, not
    thirty-three.
    """
    left = _mean_by_fold(incumbent, column)
    right = _mean_by_fold(candidate, column)
    shared = left.index.intersection(right.index)
    if len(shared) < 2:
        raise SystemExit(
            f"the two sides share {len(shared)} completed folds; a paired comparison needs "
            "at least two, and both sides must have run the same schedule."
        )
    if len(shared) < max(len(left), len(right)):
        print(
            f"# warning: comparing over {len(shared)} shared folds; "
            f"incumbent has {len(left)}, candidate has {len(right)}",
            file=sys.stderr,
        )

    differences = (right[shared] - left[shared]).astype("float64")
    folds = len(differences)
    mean = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    ratio = statistics.fmean(run.window_ratio for run in (*incumbent, *candidate))
    corrected = deviation * math.sqrt(1 / folds + ratio)

    pooled_left = statistics.fmean(run.pooled(column) for run in incumbent)
    pooled_right = statistics.fmean(run.pooled(column) for run in candidate)
    delta = pooled_right - pooled_left

    result: dict[str, Any] = {
        "metric": column,
        "seeds_incumbent": len(incumbent),
        "seeds_candidate": len(candidate),
        "folds_compared": folds,
        "incumbent_pooled": _sig(pooled_left),
        "candidate_pooled": _sig(pooled_right),
        "delta_pooled": _sig(delta),
        "delta_per_fold_mean": _sig(mean),
        "delta_sd_across_folds": _sig(deviation),
        "se_naive": _sig(deviation / math.sqrt(folds)),
        "se_corrected": _sig(corrected),
        "window_ratio": _sig(ratio),
        "t_corrected": _sig(mean / corrected) if corrected > 0 else float("nan"),
        "fold_win_rate": _sig(float((differences > 0).mean())),
        "fold_wins": int((differences > 0).sum()),
    }
    if sigma > 0:
        result["sigma_seed"] = sigma
        result["delta_over_sigma"] = _sig(delta / sigma)
    failures = evaluate_gates(incumbent, candidate, expect or Expected())
    result["gates_passed"] = not failures
    result["gate_failures"] = failures
    result["incumbent_runs"] = [run.run_id for run in incumbent]
    result.update(audit(candidate))
    return result


@dataclass(frozen=True)
class Expected:
    """What the campaign says a trial's runs must agree with.

    Empty fields are not checked. They are supplied on the command line rather
    than read from the campaign note, because this script does not know where the
    vault is and a gate that silently skips when it cannot find its reference is
    worse than no gate.
    """

    data_key: str = ""
    git_commit: str = ""
    folds: int = 0


def evaluate_gates(
    incumbent: Sequence[RunFolds], candidate: Sequence[RunFolds], expect: Expected
) -> list[str]:
    """Every admissibility gate the runs fail, as short strings; empty means admissible.

    Admissibility is not quality. A trial that fails any of these is `invalid` —
    the run is not evidence either way and the coordinate it moved stays open —
    which is a different verdict from `rejected`, and the two must not be decided
    by the same reading of the same numbers.

    Everything checkable from the runs themselves is checked here. What needs the
    campaign as a reference is checked only when ``expect`` supplies it.
    """
    runs = [*incumbent, *candidate]
    failures: list[str] = []

    for name, key in (("folds_failed", "folds/failed"), ("folds_diverged", "folds/diverged")):
        total = sum(run.metrics.get(key, 0.0) for run in runs)
        if total > 0:
            failures.append(f"{name}={total:.0f}")
    capped = sum(run.metrics.get("folds/hit_epoch_cap", 0.0) for run in runs)
    if capped > 0:
        failures.append(
            f"folds_hit_cap={capped:.0f} (raise the epoch budget campaign-wide, re-run)"
        )
    if any(run.tags.get("git_dirty") == "true" for run in runs):
        failures.append("git_dirty=true")

    # Agreement between the two sides: the comparison is only about the
    # configuration if everything else was held fixed.
    for label, values in (
        ("data_key", {run.params.get("data_key", "") for run in runs}),
        ("git_commit", {run.tags.get("git_commit", "") for run in runs}),
        ("folds_planned", {run.metrics.get("folds/planned", -1.0) for run in runs}),
        ("train.selection_metric", {run.params.get("train.selection_metric", "") for run in runs}),
    ):
        if len(values) > 1:
            failures.append(f"{label} disagrees across runs: {sorted(map(str, values))}")

    if candidate and incumbent:
        left = {run.params.get("config_hash", "") for run in incumbent}
        right = {run.params.get("config_hash", "") for run in candidate}
        if left == right:
            failures.append(
                "config_hash identical on both sides — this is a replicate, not a trial"
            )

    # Against the campaign, when it was named.
    first = runs[0]
    if expect.data_key and not first.params.get("data_key", "").startswith(expect.data_key):
        failures.append(f"data_key != campaign ({first.params.get('data_key', '')[:12]})")
    if expect.git_commit and not first.tags.get("git_commit", "").startswith(expect.git_commit):
        failures.append(f"git_commit != campaign ({first.tags.get('git_commit', '')[:12]})")
    if expect.folds and any(run.metrics.get("folds/planned", 0.0) != expect.folds for run in runs):
        failures.append(f"folds_planned != campaign ({expect.folds})")
    return failures


def _mean_by_fold(runs: Sequence[RunFolds], column: str) -> pd.Series:
    """One value per fold, averaged over the replicates that ran that fold."""
    frame = pd.concat([run.scores(column) for run in runs], axis=1)
    return frame.mean(axis=1).dropna()


def audit(runs: Sequence[RunFolds]) -> dict[str, Any]:
    """The provenance and admissibility fields a trial note has to carry.

    Read off the runs rather than retyped, because every one of them is a fact
    about what happened that nobody can reconstruct later: which commit, whether
    the tree was clean, which panel, and how many folds ended in a state that
    makes the score inadmissible.
    """
    first = runs[0]
    gates = {
        name: _sig(sum(run.metrics.get(key, 0.0) for run in runs))
        for name, key in GATE_METRICS.items()
        if any(key in run.metrics for run in runs)
    }
    seconds = sum(run.metrics.get("fit/fit_seconds_total", 0.0) for run in runs)
    return {
        **gates,
        "runtime_min": _sig(seconds / 60.0),
        "config_hash": first.params.get("config_hash", ""),
        "data_key": first.params.get("data_key", "")[:12],
        "git_commit": first.tags.get("git_commit", "")[:12],
        "git_dirty": first.tags.get("git_dirty", ""),
        "device": first.tags.get("gpu_name", "") or first.tags.get("platform", ""),
        "seeds": [run.seed for run in runs],
        "runs": [run.run_id for run in runs],
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def emit(payload: Mapping[str, Any]) -> None:
    """Print the result as a frontmatter-shaped YAML block."""
    for key, value in payload.items():
        if isinstance(value, list):
            items = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
            print(f"{key}: [{items}]")
        elif isinstance(value, str):
            print(f'{key}: "{value}"')
        else:
            print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tuning-decision statistics from MLflow runs.")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        choices=sorted(POOLED_SUMS),
        help="the per-fold summary column decisions are made on",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    floor = commands.add_parser("noise", help="spread of one configuration across its seeds")
    floor.add_argument("--runs", nargs="*", default=[], help="parent run ids of the replicates")
    floor.add_argument(
        "--experiment", default="", help="find the replicates instead of naming them"
    )
    floor.add_argument("--config-hash", default="", help="the configuration the replicates share")

    duel = commands.add_parser("compare", help="candidate against incumbent, paired over folds")
    duel.add_argument("--incumbent", nargs="+", required=True, help="parent run ids")
    duel.add_argument("--candidate", nargs="+", required=True, help="parent run ids")
    duel.add_argument("--expect-data-key", default="", help="the campaign's data_key")
    duel.add_argument("--expect-commit", default="", help="the campaign's git_commit")
    duel.add_argument("--expect-folds", type=int, default=0, help="the campaign's fold count")
    duel.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="the campaign's seed-noise standard deviation, to express the gap as a multiple",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "noise":
        run_ids = list(args.runs)
        if not run_ids:
            if not args.experiment or not (args.config_hash or args.campaign or args.trial):
                raise SystemExit(
                    "pass --runs, or --experiment with at least one of "
                    "--config-hash / --campaign / --trial."
                )
            run_ids = find_runs(
                args.experiment,
                args.tracking_uri,
                config_hash=args.config_hash,
                campaign=args.campaign,
                trial=args.trial,
            )
            if not run_ids:
                raise SystemExit("no finished parent run matches those filters.")
            print(f"# {len(run_ids)} replicates found", file=sys.stderr)
        floor = noise_floor(load_runs(run_ids, args.tracking_uri), args.column)
        emit(floor)
        return _exit_code(floor)

    result = compare(
        load_runs(args.incumbent, args.tracking_uri),
        load_runs(args.candidate, args.tracking_uri),
        args.column,
        args.sigma,
        Expected(
            data_key=args.expect_data_key,
            git_commit=args.expect_commit,
            folds=args.expect_folds,
        ),
    )
    emit(result)
    return _exit_code(result)


def _exit_code(result: Mapping[str, Any]) -> int:
    """Non-zero when a gate failed, so a caller cannot miss it.

    The block is printed either way: a trial that is `invalid` still has to be
    recorded, and a record that keeps only admissible runs is not auditable.
    """
    failures = result.get("gate_failures") or []
    if failures:
        print(f"# GATES FAILED: {'; '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
