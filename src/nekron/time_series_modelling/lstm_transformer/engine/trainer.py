"""Training / evaluation loop for the LSTM-Transformer residual model.

Adam with an exponentially decaying learning rate, gradient clipping, early
stopping on the validation loss, a saved best checkpoint, and the per-epoch
optimizer diagnostics from :mod:`nekron.nn.diagnostics` that say which
hyperparameter to move next. One batch of residual windows is one optimizer step.

No objective, no fit
--------------------
The objective is configured, and ``loss.objective="none"`` — the default while
none is chosen — produces no objective at all. A trainer built that way is still
a complete, inspectable object: it owns the model, the optimizer, the schedule
and the probes, and :meth:`Trainer.predict` runs. Only :meth:`Trainer.fit`
refuses, and it says exactly what is missing. That is the shape that lets the
whole pipeline be run and checked before there is anything to optimize.

Selection is on the validation loss, and lower is better; :class:`EarlyStopping`
maximizes, so it is handed the negated loss and :class:`FitResult` reports the
loss itself.
"""

from __future__ import annotations

import math
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from nekron.nn import (
    EarlyStopping,
    FitResult,
    build_optimization,
    resolve_device,
    same_device,
    save_checkpoint,
    set_seed,
    should_log_epoch,
)
from nekron.nn.diagnostics import ActivationProbe, weight_summary
from nekron.tracking import MlflowTracker

from ..config import LstmTransformerConfig
from ..data.windows import ResidualWindows, WindowSplits
from ..losses import ScoreObjective, build_objective
from ..model import LstmTransformer

FloatArray = npt.NDArray[np.float64]

CHECKPOINT_FILE = "lstm_transformer_best.pt"


class TrainerError(Exception):
    """Raised when a run is asked to do something its configuration does not describe."""


class Trainer:
    """Owns the model, optimizer and window splits for a single training run."""

    def __init__(
        self,
        model: LstmTransformer,
        cfg: LstmTransformerConfig,
        splits: WindowSplits,
        tracker: MlflowTracker,
        *,
        provenance: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.splits = splits
        self.tracker = tracker
        self.provenance = dict(provenance or {})
        self.device = resolve_device(cfg.train.device)
        if not same_device(splits.train.device, self.device):
            raise TrainerError(
                f"the windows live on {splits.train.device} but train.device resolves to "
                f"{self.device}; build the splits on the device the trainer will use."
            )
        self.model = model.to(self.device)
        self.objective: ScoreObjective | None = build_objective(cfg.loss)
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        optimization = build_optimization(self.model, self.cfg.optim, self.cfg.train)
        self.optimizer = optimization.optimizer
        self.scheduler = optimization.scheduler
        self.probe = optimization.probe
        self._optimization = optimization

    # ----------------------------------------------------------------- #
    # Fitting
    # ----------------------------------------------------------------- #

    def _require_objective(self) -> ScoreObjective:
        if self.objective is None:
            raise TrainerError(
                "no training objective is configured (loss.objective='none'), so there is "
                "nothing to optimize. Implement one in "
                "nekron.time_series_modelling.lstm_transformer.losses, add its name to "
                "config.OBJECTIVES, and set loss.objective to it."
            )
        return self.objective

    def _train_epoch(
        self, objective: ScoreObjective, generator: torch.Generator
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for batch in self.splits.train.iter_batches(
            self.cfg.data.batch_size, shuffle=self.cfg.data.shuffle, generator=generator
        ):
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch.sequences)
            breakdown = objective(output, batch, self.model.parameters())
            breakdown.total.backward()  # type: ignore[no-untyped-call]
            self.probe.step()
            # Accumulated as device tensors and read once at the end of the epoch:
            # converting each step's loss to a float here would synchronize the
            # device on every step.
            for name, value in breakdown.detached_components().items():
                totals[name] = value if name not in totals else totals[name] + value
            steps += 1
        if steps == 0:
            raise TrainerError("the training split produced no batches.")
        return {name: float(value) / steps for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, windows: ResidualWindows) -> float:
        """Mean objective value over ``windows``, weighted by window count.

        Weighted rather than averaged over batches, so the last partial batch
        counts for what it contains instead of for a whole one.
        """
        objective = self._require_objective()
        self.model.eval()
        total = torch.zeros((), device=self.device)
        count = 0
        for batch in windows.iter_batches(self.cfg.data.batch_size, shuffle=False):
            output = self.model(batch.sequences)
            breakdown = objective(output, batch, self.model.parameters())
            total = total + breakdown.total.detach() * len(batch)
            count += len(batch)
        return float(total) / count if count else float("nan")

    @torch.no_grad()
    def evaluate_per_period(self, windows: ResidualWindows) -> pd.DataFrame:
        """One row per scored date: the loss summed over that date's windows.

        The counterpart to the conditional autoencoder's method of the same name,
        and it exists for the same reason. The *sums* are recorded, not the ratio:
        a mean over a set of dates cannot be recovered from a set of per-date
        means — each has a different denominator — so pooling folds, years or
        regimes after the fact is only possible from the numerator and the
        denominator kept apart.

        Grouping happens inside the same pass that computes the loss, so this
        costs one forward pass over the split, not two.
        """
        objective = self._require_objective()
        self.model.eval()
        totals: dict[int, float] = {}
        counts: dict[int, int] = {}
        for batch in windows.iter_batches(self.cfg.data.batch_size, shuffle=False):
            output = self.model(batch.sequences)
            # The objective is defined over a batch, so a per-window loss is only
            # available where it decomposes. Scoring each date's windows as their
            # own batch would change the objective for any cross-sectional
            # objective, so the batch's mean is attributed to the dates it covers
            # in proportion to how many windows each contributed.
            loss = float(objective(output, batch, self.model.parameters()).total.detach())
            ends, counts_in_batch = np.unique(batch.date_indices.cpu().numpy(), return_counts=True)
            for end, n in zip(ends.tolist(), counts_in_batch.tolist(), strict=True):
                totals[end] = totals.get(end, 0.0) + loss * n
                counts[end] = counts.get(end, 0) + n
        order = sorted(counts)
        return pd.DataFrame(
            {
                "date": pd.DatetimeIndex([windows.dates[i] for i in order]),
                "n_windows": [counts[i] for i in order],
                "loss_sum": [totals[i] for i in order],
            }
        )

    def log_summary(self) -> None:
        """Record the configuration, the provenance and the shape of the run.

        Split out from :meth:`fit` because a run with no objective still has all
        three, and a build that is never recorded cannot be compared with the fit
        that eventually follows it.
        """
        self.tracker.log_config(self.cfg)
        if self.provenance:
            self.tracker.set_tags(self.provenance)
        self.tracker.log_metrics(
            {
                "model/num_parameters": float(self.model.num_parameters()),
                "data/train_windows": float(len(self.splits.train)),
                "data/val_windows": float(len(self.splits.val)),
                "data/test_windows": float(len(self.splits.test)),
            },
            step=0,
        )

    def fit(self) -> FitResult:
        started = time.perf_counter()
        """Train until the validation loss stops improving, then restore the best model."""
        objective = self._require_objective()
        if len(self.splits.val) == 0:
            raise TrainerError("the validation split is empty; cannot select a model.")
        set_seed(self.cfg.train.seed)
        generator = torch.Generator()
        generator.manual_seed(self.cfg.train.seed)
        self.log_summary()

        stopper = EarlyStopping(
            self.cfg.train.early_stopping_patience,
            min_delta=self.cfg.train.min_delta,
        )
        history: list[dict[str, float]] = []
        with ActivationProbe(self.model, self.cfg.train.diagnostics) as activations:
            for epoch in range(self.cfg.train.epochs):
                train_logs = self._train_epoch(objective, generator)
                val_loss = self.evaluate(self.splits.val)
                record = {
                    **train_logs,
                    "val/loss": val_loss,
                    "lr": self._optimization.learning_rate,
                    **self.probe.summary(),
                    **activations.summary(),
                    **weight_summary(self.model, self.cfg.train.diagnostics),
                }
                self.scheduler.step()
                history.append(record)

                if should_log_epoch(
                    epoch,
                    interval=self.cfg.mlflow.log_every_n_epochs,
                    epochs=self.cfg.train.epochs,
                ):
                    self.tracker.log_metrics(record, step=epoch)

                # EarlyStopping maximizes; the objective is a loss.
                if stopper.update(-val_loss, epoch, self.model):
                    break

        stopper.restore(self.model)
        best = -stopper.best_metric if math.isfinite(stopper.best_metric) else float("nan")
        self._save_checkpoint(stopper.best_epoch, best)
        return FitResult(
            best_metric=best,
            best_epoch=stopper.best_epoch,
            epoch_budget=self.cfg.train.epochs,
            fit_seconds=time.perf_counter() - started,
            history=history,
        )

    # ----------------------------------------------------------------- #
    # Inference
    # ----------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, windows: ResidualWindows) -> pd.DataFrame:
        """One row per window: the date it ends on, its entity and its scores.

        Long-form and joinable, because that is the only shape in which a score
        can be put back next to the panel it came from. Runs without an objective
        — an untrained model still scores — which is what makes the pipeline
        inspectable before there is anything to optimize.

        The batches are accumulated as arrays and turned into a frame once, at the
        end. A split here is millions of windows, and both of the obvious
        alternatives scale badly at that size: a frame per batch concatenated at
        the end reallocates every column, and looking each entity name up in
        Python costs one interpreter round trip per row. The names are taken by
        one vectorized take instead, and the dates by another.
        """
        self.model.eval()
        scores: list[FloatArray] = []
        ends: list[npt.NDArray[np.int64]] = []
        entities: list[npt.NDArray[np.int64]] = []
        targets: list[FloatArray] = []
        for batch in windows.iter_batches(self.cfg.data.batch_size, shuffle=False):
            scores.append(self.model(batch.sequences).scores.cpu().numpy().astype(np.float64))
            ends.append(batch.date_indices.cpu().numpy().astype(np.int64))
            entities.append(batch.entity_indices.cpu().numpy().astype(np.int64))
            if batch.targets is not None:
                targets.append(batch.targets.cpu().numpy().astype(np.float64))

        columns = ["date", "entity", *(f"score_{i}" for i in range(self.model.cfg.head.out_dim))]
        if not scores:
            return pd.DataFrame({name: [] for name in columns})

        stacked = np.concatenate(scores)
        names = np.asarray(windows.entity_names, dtype=object)
        frame = pd.DataFrame(
            {
                "date": windows.dates.to_numpy()[np.concatenate(ends)],
                "entity": names[np.concatenate(entities)],
                **{f"score_{i}": stacked[:, i] for i in range(stacked.shape[1])},
            }
        )
        if targets:
            frame["target"] = np.concatenate(targets)
        return frame

    # ----------------------------------------------------------------- #
    # Checkpointing
    # ----------------------------------------------------------------- #

    def _save_checkpoint(self, best_epoch: int, best_metric: float) -> None:
        save_checkpoint(
            self.model,
            directory=self.cfg.train.checkpoint_dir,
            filename=CHECKPOINT_FILE,
            config=self.cfg,
            best_epoch=best_epoch,
            best_metric=best_metric,
            tracker=self.tracker,
            # What the weights mean is fixed by the residuals they were fitted on,
            # and those are fixed by the autoencoder run and the window length. A
            # checkpoint without this cannot be interpreted once the autoencoder
            # has been refitted.
            extras={
                "provenance": self.provenance,
                "seq_len": self.splits.seq_len,
                "num_channels": self.splits.num_channels,
            },
        )
