"""Reusable MLflow experiment tracking shared by all models.

Provides :class:`MlflowConfig`, the tracking settings a model embeds in its
configuration, and :class:`MlflowTracker`, a context-managed MLflow run. When
:attr:`MlflowConfig.enabled` is False every method is a no-op, so training code
can call the tracker unconditionally.

A run can nest. :meth:`MlflowTracker.child` opens a run underneath the active
one, which is what a cross-validated fit needs: the sweep as a whole is one run
carrying the configuration and the aggregate results, and each fold is a run of
its own underneath it carrying that fold's curve, model and metrics. MLflow
parents a nested run to whatever run is active on the thread, so a child is only
correct while its parent is inside its own ``with`` block.

Every run — parent and child alike — is tagged with its own provenance on the way
in: the commit, whether the tree was dirty, the interpreter, the accelerator and
the determinism flags. It is attached before any work happens so that a run which
dies partway still says what it was, and it is attached to children too so that a
fold recovered on its own is interpretable without walking back to its parent.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import pandas as pd

from nekron.provenance import config_hash, run_provenance

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# MLflow truncates a param value longer than this (and warns); truncating here
# instead keeps the cut visible in the value itself.
MAX_PARAM_VALUE = 6000

# MLflow's own client batches params at 100 per request, so this only has to be
# under that; it is kept explicit because a managed tracking server may cap a
# single call lower than the open-source one does.
PARAM_CHUNK = 90

CONFIG_FILE = "config.json"
"""The resolved configuration, attached to a run as an artifact.

Flattened parameters are a searchable index of a configuration, not a copy of it:
a long list is comma-joined and a long value is truncated, so the parameter view
cannot be read back as the thing that ran. The artifact can.
"""

CONFIG_HASH_PARAM = "config_hash"
"""Parameter naming the configuration a run trained, seed excluded."""

CONFIG_HASH_IGNORE: tuple[str, ...] = ("train.seed", "train.checkpoint_dir", "mlflow")
"""Dotted paths excluded from :data:`CONFIG_HASH_PARAM`.

The seed is what distinguishes replicates of one configuration and so cannot be
part of its identity; the tracking section decides where results are written, and
the checkpoint directory where weights are written, not what either of them is.
Leaving the checkpoint directory in would give every fold of one sweep a
different hash, since a multi-fold run gives each fold its own directory.
"""


@dataclass
class MlflowConfig:
    """MLflow experiment-tracking settings.

    Parameters
    ----------
    tags:
        Free-form tags attached to every run this configuration opens, parent and
        child. What a search is organized by rather than what a model is: which
        campaign a run belongs to, which trial it is, whether it is a screening
        run or a confirmation. Set from the command line as
        ``+mlflow.tags.trial=T07``, so a run can be labeled without editing a
        configuration file and without the label entering the config hash.
    """

    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "Default"
    run_name: str | None = None
    log_every_n_epochs: int = 1
    log_model: bool = True
    tags: dict[str, str] = field(default_factory=dict)


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    """Flatten a nested config dict into dotted scalar params for MLflow.

    A list of scalars becomes one comma-joined value, but a list of *structures*
    is recursed into. Joining those instead is what turns a feature list into a
    single value tens of thousands of characters long — well past what MLflow
    accepts — and the run then dies on its first tracked call, which is after the
    whole data pipeline has already run.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(obj, (list, tuple)):
        if any(isinstance(item, dict | list | tuple) for item in obj):
            for position, item in enumerate(obj):
                # Dotted, not bracketed: MLflow rejects a param name containing
                # brackets, and this matches Hydra's own list-override syntax
                # (``featurizers.0.params.input_column``).
                _flatten(f"{prefix}.{position}", item, out)
        else:
            out[prefix] = _truncate(",".join(map(str, obj)))
    elif isinstance(obj, str):
        out[prefix] = _truncate(obj)
    else:
        out[prefix] = obj


def _truncate(value: str) -> str:
    """Cap a param value at MLflow's limit, marking it so the cut is visible."""
    if len(value) <= MAX_PARAM_VALUE:
        return value
    return value[: MAX_PARAM_VALUE - 3] + "..."


class MlflowTracker:
    """Context-managed MLflow run; degrades to a no-op when disabled."""

    def __init__(
        self,
        cfg: MlflowConfig,
        *,
        run_name: str | None = None,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self._run_name = cfg.run_name if run_name is None else run_name
        self._nested = nested
        # Configured tags first, so a caller passing the same key for one run wins
        # over the blanket setting; provenance last, because it is measured rather
        # than chosen and nothing should be able to overwrite it with a claim.
        self._tags = {**cfg.tags, **(dict(tags) if tags else {})}
        self._mlflow: Any | None = None
        self._active = False

    def __enter__(self) -> MlflowTracker:
        if not self.cfg.enabled:
            return self

        # Imported here rather than at module scope: importing mlflow costs
        # seconds and pulls in a large dependency tree, and this module also
        # exports :class:`MlflowConfig`, which every model config imports whether
        # or not tracking is enabled.
        import mlflow

        self._mlflow = mlflow
        if not self._nested:
            # A nested run inherits both from the run it is opened under, and
            # re-pointing the tracking URI while that parent is active would
            # strand it.
            mlflow.set_tracking_uri(self.cfg.tracking_uri)
            mlflow.set_experiment(self.cfg.experiment_name)
        mlflow.start_run(run_name=self._run_name, nested=self._nested)
        mlflow.set_tags({**self._tags, **run_provenance()})
        self._active = True
        return self

    def child(self, run_name: str, *, tags: Mapping[str, str] | None = None) -> MlflowTracker:
        """A run nested under this one, sharing its settings.

        Only meaningful while ``self`` is inside its own ``with`` block: MLflow
        parents a nested run to whatever run is active on the thread, so a child
        opened after its parent has ended would attach to the wrong run or to
        none at all.
        """
        return MlflowTracker(self.cfg, run_name=run_name, nested=True, tags=tags)

    @property
    def run_id(self) -> str | None:
        """The active run's id, or ``None`` when tracking is disabled.

        What a later analysis needs in order to find this run again — every model
        and artifact logged here is addressed as ``runs:/<run_id>/<name>``.
        """
        if not self._active or self._mlflow is None:
            return None
        run = self._mlflow.active_run()
        return None if run is None else str(run.info.run_id)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            self._active = False

    def log_config(self, config: DataclassInstance) -> None:
        """Record ``config`` three ways: as parameters, as a digest, as an artifact.

        The three are not redundant. Parameters are what a search query filters on
        one field at a time; the digest is what identifies the whole configuration
        in one field, so that replicates of it can be found without comparing six
        hundred parameters; the artifact is the only one of the three that can be
        read back as the configuration itself, since the parameter view joins lists
        and truncates long values.
        """
        if not self._active or self._mlflow is None:
            return
        payload = asdict(config)
        flat: dict[str, Any] = {}
        _flatten("", payload, flat)
        self.log_params(flat)
        self.log_params({CONFIG_HASH_PARAM: config_hash(payload, ignore=CONFIG_HASH_IGNORE)})
        self.log_json(payload, CONFIG_FILE)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run parameters, chunked to stay under a batch limit.

        MLflow refuses to change a parameter that is already recorded, so a value
        may be written once per run. Re-logging the *same* value is accepted, but
        anything that varies between folds belongs on the fold's own run.
        """
        if not self._active or self._mlflow is None:
            return
        items = list(params.items())
        for start in range(0, len(items), PARAM_CHUNK):
            self._mlflow.log_params(dict(items[start : start + PARAM_CHUNK]))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Attach searchable tags to the run.

        Tags, unlike params, are what ``mlflow.search_runs`` filters on cheaply
        and what a later analysis uses to pick one fold out of a sweep.
        """
        if not self._active or self._mlflow is None:
            return
        self._mlflow.set_tags(dict(tags))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self._active or self._mlflow is None:
            return
        clean = {k: v for k, v in metrics.items() if v == v}  # drop NaNs
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self._active or self._mlflow is None:
            return
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_json(self, payload: Any, filename: str, *, artifact_path: str | None = None) -> None:
        """Attach a small JSON document to the run.

        For the things an analysis needs *before* it is willing to unpickle a
        model — the column order the network was trained on, above all.
        """
        if not self._active or self._mlflow is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dataframe(
        self, frame: pd.DataFrame, filename: str, *, artifact_path: str | None = None
    ) -> None:
        """Write a frame to Parquet and attach it to the run.

        Deliberately not ``mlflow.log_table``: that call *appends* when the
        artifact already exists, re-downloading and rewriting the whole table
        each time, so logging one frame per fold under one name grows
        quadratically and silently concatenates folds that were meant to stay
        apart. Parquet also round-trips dtypes exactly, which a results table
        read back months later depends on.
        """
        if not self._active or self._mlflow is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            frame.to_parquet(path, index=False)
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model(self, model: Any) -> None:
        if not self._active or self._mlflow is None or not self.cfg.log_model:
            return
        # Serialize the module with pickle; the default "pt2" traces the forward
        # graph and would require an input_example, which does not fit models whose
        # forward takes several tensors. ``name=`` is the MLflow 3 spelling — it
        # replaced ``artifact_path=`` — which is why pyproject requires mlflow>=3.
        self._mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
