"""What a run has to record about itself before its numbers mean anything.

A metric is only evidence if the thing that produced it can be named again. Three
identities do that, and none of them is recoverable after the fact:

``git_identity``
    Which code ran, and whether the working tree matched it. A commit SHA alone is
    a half-truth on a machine where anything is uncommitted, which is the normal
    state of a machine that is being worked on.
``environment_identity``
    Which interpreter, which torch, which device, and whether the nondeterministic
    fast paths were on. Two runs of one seed on one commit are not the same run if
    one of them autotuned its kernels.
``config_hash``
    Which *configuration* ran, with the seed and the tracking settings removed —
    so "the same configuration under a different seed" is a query rather than a
    thing to remember. Comparing two configurations is the whole point of a
    tuning campaign, and it cannot be done from flattened parameters that differ
    in one entry out of six hundred.

Every function here answers with strings and never raises. Provenance is recorded
on the way into a run that is about to do real work; a missing ``git`` executable
or a torch built without CUDA is a blank field, not a failed training run.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from nekron.cache import canonical_json

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

HASH_LENGTH = 12
"""Hex characters kept from a configuration digest.

Twelve is git's own abbreviation length and collides at a rate far below the
number of configurations a search will ever hold, while staying short enough to
read off a note and type into a query.
"""

GIT_TIMEOUT_SECONDS = 5.0
"""How long a git call may take before the run proceeds without provenance."""


def config_hash(config: DataclassInstance | Mapping[str, Any], *, ignore: Sequence[str]) -> str:
    """Digest ``config`` with the dotted paths in ``ignore`` removed.

    Two runs share a hash exactly when they would train the same model on the same
    data, which is what makes a set of seed replicates findable and a repeated
    configuration detectable. ``ignore`` names what is deliberately outside that
    equivalence — the seed, whose whole purpose is to differ between replicates,
    and the tracking settings, which change where results are written and not what
    they are.

    A path names a whole subtree: ``"mlflow"`` drops the section, ``"train.seed"``
    drops the one entry. Removing a path that is not there is not an error, so one
    ignore list can serve configurations that do not all have the same sections.
    """
    payload = dict(asdict(config) if is_dataclass(config) else config)
    for path in ignore:
        _drop(payload, path.split("."))
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def _drop(payload: dict[str, Any], path: Sequence[str]) -> None:
    """Remove ``path`` from ``payload`` in place, tolerating an absent branch."""
    head, *rest = path
    if not rest:
        payload.pop(head, None)
        return
    branch = payload.get(head)
    if isinstance(branch, dict):
        # Copied rather than mutated: the caller's config dict is shared with
        # whatever else holds a reference to the same nested mapping.
        nested = dict(branch)
        _drop(nested, rest)
        payload[head] = nested


@lru_cache(maxsize=1)
def git_identity() -> dict[str, str]:
    """The commit a run was launched from, and whether the tree was clean.

    ``git_dirty`` is the load-bearing half, and it counts *tracked* modifications
    only. A run recorded against a commit whose tracked files had been edited is
    not reproducible from that commit, and the only moment anyone can know that is
    while the run is starting. Untracked files are reported separately as a count:
    they are worth recording, but a repository that carries unrelated work in
    progress would otherwise mark every run dirty, and a flag that is always true
    is a flag nobody reads.

    Cached: a cross-validated run opens one MLflow run per fold and the answer
    cannot change within a process.
    """
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {"git_commit": "", "git_branch": "", "git_dirty": "", "git_untracked": ""}
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Tracked modifications only. Those are the ones that make the commit a
        # false claim: the file exists in it, and what ran was something else.
        "git_dirty": "true" if _git("status", "--porcelain", "--untracked-files=no") else "false",
        # Untracked files counted, not folded into the flag above. A repository
        # with unrelated work in progress would otherwise report every run as
        # dirty, and a flag that is always true stops being read. They still
        # belong in the record: an untracked module can shadow an imported one,
        # and nothing in the commit would show it.
        "git_untracked": str(len(_git("ls-files", "--others", "--exclude-standard").splitlines())),
    }


def _git(*args: str) -> str:
    """One git command's trimmed stdout, or ``""`` if it cannot be run."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def environment_identity() -> dict[str, str]:
    """Interpreter, libraries, device and the determinism flags in force.

    The determinism entries are not decoration. TF32 matrix multiplication and
    cuDNN's autotuning benchmark both make a run irreproducible at its own seed,
    and both are properties of the process rather than of the configuration — so a
    seed-noise measurement taken with them on is measuring the hardware as much as
    the initialization, and nothing in the configuration would ever say so.
    """
    identity = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
    }
    identity.update(_torch_identity())
    return identity


def _torch_identity() -> dict[str, str]:
    """Torch's version, the accelerator it can reach, and its determinism flags."""
    try:
        import torch
    except ImportError:
        return {}

    backends = torch.backends
    identity = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "cuda_version": torch.version.cuda or "",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cudnn_deterministic": str(backends.cudnn.deterministic).lower(),
        "cudnn_benchmark": str(backends.cudnn.benchmark).lower(),
        "tf32_matmul": str(backends.cuda.matmul.allow_tf32).lower(),
        "tf32_cudnn": str(backends.cudnn.allow_tf32).lower(),
    }
    return identity


def run_provenance() -> dict[str, str]:
    """Everything about the machine and the code, as MLflow tags.

    Read once per run and attached before any training happens, so a run that
    crashes halfway still says what it was.
    """
    return {**git_identity(), **environment_identity(), "pid": str(os.getpid())}
