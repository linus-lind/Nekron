"""Training / evaluation loop for the conditional autoencoder.

AdamW with an exponentially decaying learning rate, an L1-penalized mean squared
pricing error, per-period forward passes accumulated over ``data.batch_periods``
periods per optimizer step, early stopping on a validation R-squared, and a saved
best checkpoint. Each period's cross-section is a training example, so the beta
network's batch normalization is applied over that period's stocks.

Every epoch records one row of metrics: the loss and its components, the fit on
both splits, and the optimizer diagnostics from :mod:`nekron.nn.diagnostics` that
say which hyperparameter to move next. See :meth:`Trainer._epoch_metrics` for what
each one is for.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from nekron.nn import (
    EarlyStopping,
    FitResult,
    build_optimization,
    resolve_device,
    save_checkpoint,
    set_seed,
    should_log_epoch,
)
from nekron.nn.diagnostics import ActivationProbe, weight_summary
from nekron.tracking import MlflowTracker

from ..config import ConditionalAutoencoderConfig
from ..data.dataset import CrossSectionDataset, PanelSplits
from ..losses import CaeLoss
from ..metrics import (
    FactorForecast,
    RSquared,
    SplitScore,
    factor_diagnostics,
    per_period_errors,
    r_squared,
)
from ..model import ConditionalAutoencoder

FloatArray = npt.NDArray[np.float64]

CHECKPOINT_FILE = "conditional_autoencoder_best.pt"
"""Named rather than inlined: another package defaults to reading this artifact."""

_SELECTION_ATTR = {"total_r2": "total", "predictive_r2": "predictive"}


@dataclass
class InferenceResult:
    """Per-period model outputs (variable stock counts kept as per-period arrays).

    ``dates`` carries the actual period so a result can be joined back onto a
    panel; ``date_indices`` are positions in the cross-section sequence the dataset
    is a window onto — the whole panel under a fold schedule, so they are
    comparable across folds, and the split itself under a single-split build.
    """

    date_indices: list[int]
    dates: list[pd.Timestamp]
    entities: list[tuple[str, ...]]
    factors: FloatArray  # [W, K]
    fitted_returns: list[FloatArray]  # per period [N]
    betas: list[FloatArray]  # per period [N, K]


class Trainer:
    """Owns the model, optimizer and data splits for a single training run."""

    def __init__(
        self,
        model: ConditionalAutoencoder,
        cfg: ConditionalAutoencoderConfig,
        splits: PanelSplits,
        tracker: MlflowTracker,
    ) -> None:
        self.cfg = cfg
        self.splits = splits
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.loss_fn = CaeLoss(cfg.loss)
        self.tracker = tracker
        if cfg.train.selection_metric not in _SELECTION_ATTR:
            raise ValueError(
                f"selection_metric must be one of {sorted(_SELECTION_ATTR)}; "
                f"got {cfg.train.selection_metric!r}."
            )
        if len(splits.train) == 0:
            raise ValueError(
                "training split is empty; there is nothing to fit, and nothing to warm the "
                "predictive forecast on."
            )
        if len(splits.val) == 0:
            raise ValueError("validation split is empty; cannot select a model.")
        # No further minimum on the validation split. It used to need two periods
        # under ``predictive_r2`` selection, because the forecast started cold there
        # and the first period went unscored; it now arrives carrying the training
        # window, so one validation period is one scored validation period.
        #
        # [W, P]: the factor network's entire input for the training split, stacked
        # once. The managed portfolios are solved when a cross-section is built and
        # no model parameter enters them, so this cannot go stale.
        self._train_portfolios = torch.stack(
            [splits.train[i].portfolios for i in range(len(splits.train))]
        ).to(self.device)
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        # The L1 penalty is handed to the optimization rather than left in the loss
        # whenever the configured mode is proximal; ``CaeLoss.proximal`` returns
        # None under the subgradient mode, where the loss carries it instead.
        optimization = build_optimization(
            self.model,
            self.cfg.optim,
            self.cfg.train,
            proximal=self.loss_fn.proximal(),
        )
        self.optimizer = optimization.optimizer
        self.scheduler = optimization.scheduler
        self.probe = optimization.probe
        self._optimization = optimization

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        order = torch.randperm(len(self.splits.train)).tolist()
        batch_periods = self.cfg.data.batch_periods
        totals: dict[str, torch.Tensor] = {}
        # The in-sample fit is accumulated from the training forward passes
        # themselves rather than from a second pass over the split: at ~2500
        # periods that pass would cost about as much as the epoch it reports on.
        # The weights move underneath it, exactly as they do under the loss.
        sse = torch.zeros((), device=self.device)
        sst = torch.zeros((), device=self.device)
        n_steps = 0
        for start in range(0, len(order), batch_periods):
            indices = order[start : start + batch_periods]
            self.optimizer.zero_grad(set_to_none=True)
            fitted_parts: list[torch.Tensor] = []
            target_parts: list[torch.Tensor] = []
            for i in indices:
                section = self.splits.train[i]
                beta_inputs = section.beta_inputs.to(self.device)
                returns = section.returns.to(self.device)
                output = self.model(beta_inputs, section.portfolios.to(self.device))
                fitted_parts.append(output.fitted_returns)
                target_parts.append(returns)
            fitted = torch.cat(fitted_parts)
            target = torch.cat(target_parts)
            breakdown = self.loss_fn(fitted, target, self.model.parameters())
            breakdown.total.backward()  # type: ignore[no-untyped-call]
            self.probe.step()
            with torch.no_grad():
                sse += (fitted - target).pow(2).sum()
                sst += target.pow(2).sum()
            for key, value in breakdown.detached_components().items():
                totals[key] = value if key not in totals else totals[key] + value
            n_steps += 1
        logs = {key: float(value) / max(n_steps, 1) for key, value in totals.items()}
        total_sst = float(sst)
        logs["train/total_r2"] = 1.0 - float(sse) / total_sst if total_sst > 0.0 else float("nan")
        return logs

    @torch.no_grad()
    def _forward_split(
        self, dataset: CrossSectionDataset
    ) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray]]:
        """Loadings, factors and realized returns for every period, in date order.

        Split out from :meth:`evaluate` because the epoch's validation pass feeds
        both the R-squared and the factor diagnostics, and running the split twice
        to produce two summaries of the same forward pass would double the cost of
        evaluation.
        """
        self.model.eval()
        betas: list[FloatArray] = []
        factors: list[FloatArray] = []
        returns: list[FloatArray] = []
        for i in range(len(dataset)):
            section = dataset[i]
            output = self.model(
                section.beta_inputs.to(self.device), section.portfolios.to(self.device)
            )
            betas.append(output.betas.cpu().numpy().astype(np.float64))
            factors.append(output.factors.cpu().numpy().astype(np.float64))
            returns.append(section.returns.numpy().astype(np.float64))
        return betas, factors, returns

    @torch.no_grad()
    def _factors(self, portfolios: torch.Tensor) -> FloatArray:
        """Fitted factors ``[W, K]`` for a ``[W, P]`` stack of managed portfolios.

        Advancing a forecast needs the factors and nothing else, and the factors are
        the factor network's output on one period's managed portfolios. The network
        is a plain MLP whose batch normalization :class:`~..config.ModelConfig`
        refuses and whose dropout is off in eval mode, so nothing couples the rows
        of a batch and this computes the same *function* as ``W`` separate calls.

        Not necessarily the same floats: a ``[W, P]`` matrix product and a ``[P]``
        matrix-vector product reach different BLAS kernels, and on some backends
        (CPU among them) the two agree only to float32 rounding — around ``1e-7``
        relative, which is the precision the model's own outputs carry and several
        orders below anything an R-squared reports. A window that is scored takes
        its factors from the full forward pass it needs for its loadings anyway;
        this route is for the windows that are only walked.

        That is what makes a train-seeded validation metric affordable every epoch:
        the beta network, which is the expensive half of a forward pass because it
        runs over every stock rather than over one vector, is never touched.
        """
        self.model.eval()
        if portfolios.shape[0] == 0:
            return np.zeros((0, self.model.cfg.num_factors), dtype=np.float64)
        factors: torch.Tensor = self.model.factor_network(portfolios)
        return factors.cpu().numpy().astype(np.float64)

    def _split_factors(self, dataset: CrossSectionDataset) -> FloatArray:
        """:meth:`_factors` over a whole split, in date order."""
        if len(dataset) == 0:
            return np.zeros((0, self.model.cfg.num_factors), dtype=np.float64)
        stacked = torch.stack([dataset[i].portfolios for i in range(len(dataset))])
        return self._factors(stacked.to(self.device))

    def _cold_forecast(self) -> FactorForecast:
        """The configured estimator, with no history observed.

        Every evaluation path opens from here rather than from a literal, so a run
        configured for ``ewma`` cannot be scored with an expanding mean anywhere.
        """
        return self.cfg.factor_forecast.build()

    def evaluate(
        self,
        dataset: CrossSectionDataset,
        *,
        forecast: FactorForecast | None = None,
    ) -> RSquared:
        """Total and predictive R-squared of the model over a dataset in date order.

        ``forecast`` is the state earlier splits left the factor forecast in, which
        is what lets the predictive statistic reflect the whole factor history
        rather than only this split's. Omitted, the split starts cold — in the
        configured estimator, never in a hard-coded one.
        """
        betas, factors, returns = self._forward_split(dataset)
        return r_squared(
            betas,
            factors,
            returns,
            forecast=self._cold_forecast() if forecast is None else forecast,
        )

    def score(
        self,
        dataset: CrossSectionDataset,
        *,
        forecast: FactorForecast | None = None,
    ) -> SplitScore:
        """One row per period, plus the forecast state the split ends in.

        The rows are the squared errors behind :meth:`evaluate`'s ratios, and are
        what a fold records about a window. Summing them reproduces :meth:`evaluate`
        up to the order the floating-point additions happen in, and keeping them is
        the only way to pool a statistic across folds afterwards — an R-squared is a
        ratio of sums, so a set of per-fold R-squareds cannot be combined into the
        R-squared of their union. It is also what a distribution over periods is
        drawn from.

        The forecast comes back with them because the next split in chronological
        order is seeded from it, and one forward pass produces both: a fold walks
        each of its windows exactly once.
        """
        betas, factors, returns = self._forward_split(dataset)
        dates = [dataset[i].date for i in range(len(dataset))]
        return per_period_errors(
            betas,
            factors,
            returns,
            dates,
            forecast=self._cold_forecast() if forecast is None else forecast,
        )

    def advance_forecast(
        self,
        dataset: CrossSectionDataset,
        *,
        forecast: FactorForecast | None = None,
    ) -> FactorForecast:
        """``forecast`` walked over every period of ``dataset``, in date order.

        How a window that is not itself scored — a fold's training split — still
        reaches the windows that are. It replaces a sum-and-count summary, which
        could be added across splits but cannot express a forecast that discounts
        the past: an exponentially weighted state has to be *walked* through a split
        rather than assembled from one.
        """
        start = self._cold_forecast() if forecast is None else forecast
        return start.extended(self._split_factors(dataset))

    @torch.no_grad()
    def predict(self, dataset: CrossSectionDataset) -> InferenceResult:
        """Collect per-period factors, loadings and fitted returns (in date order)."""
        self.model.eval()
        date_indices: list[int] = []
        dates: list[pd.Timestamp] = []
        entities: list[tuple[str, ...]] = []
        factors: list[FloatArray] = []
        fitted_returns: list[FloatArray] = []
        betas: list[FloatArray] = []
        for i in range(len(dataset)):
            section = dataset[i]
            output = self.model(
                section.beta_inputs.to(self.device), section.portfolios.to(self.device)
            )
            date_indices.append(section.date_index)
            dates.append(section.date)
            entities.append(section.entities)
            factors.append(output.factors.cpu().numpy().astype(np.float64))
            fitted_returns.append(output.fitted_returns.cpu().numpy().astype(np.float64))
            betas.append(output.betas.cpu().numpy().astype(np.float64))
        stacked = (
            np.stack(factors)
            if factors
            else np.zeros((0, self.model.cfg.num_factors), dtype=np.float64)
        )
        return InferenceResult(
            date_indices=date_indices,
            dates=dates,
            entities=entities,
            factors=stacked,
            fitted_returns=fitted_returns,
            betas=betas,
        )

    def _selected_metric(self, rsq: RSquared) -> float:
        return float(getattr(rsq, _SELECTION_ATTR[self.cfg.train.selection_metric]))

    def _epoch_metrics(
        self, train_logs: dict[str, float], activations: ActivationProbe
    ) -> tuple[dict[str, float], RSquared]:
        """One epoch's row of metrics, and the validation fit early stopping reads.

        The row is deliberately small — every entry answers a question some
        hyperparameter is the answer to, and anything that only restates another
        entry is left out:

        ``loss/*`` and ``train/total_r2`` against ``val/*``
            The fit, and the gap between the two splits that says whether the model
            is over- or under-regularized. ``train/total_r2`` is accumulated across
            the epoch as the weights move, so it lags a clean in-sample score by
            about half an epoch; the gap it forms with ``val/total_r2`` is what it
            is for, not its level.
        ``val/predictive_r2``
            Scored against a forecast that opens carrying the whole training
            window, so every validation period is scored and the number is the one
            the fold also reports as ``val/final_predictive_r2``. It is *not*
            comparable with runs recorded before the forecast was carried across
            the split boundary, which scored validation from a cold start.
        ``lr``, ``opt/update_ratio``
            The learning rate this epoch trained at, and what it actually moved.
        ``grad/*``
            The pre-clip gradient norms, which is what a clipping threshold is set
            from, and how often the current threshold fires.
        ``weights/*``
            What weight decay and the L1 penalty are doing to the weights.
        ``act/*``
            Whether the chosen activation is passing gradient or switching itself
            off.
        ``factors/*``
            Whether the ``K`` latent factors are all earning their place.
        """
        # Validation opens carrying the whole training window, so it scores every
        # period it has instead of throwing its first one away — and this is the same
        # state, reached the same way, that a fold seeds its test window from, so the
        # epoch metric and the fold's final number are one statistic rather than two.
        train_forecast = self._cold_forecast().extended(self._factors(self._train_portfolios))
        betas, factors, returns = self._forward_split(self.splits.val)
        val_rsq = r_squared(betas, factors, returns, forecast=train_forecast)
        factors_summary = factor_diagnostics(betas, factors)
        record = {
            **train_logs,
            "val/total_r2": val_rsq.total,
            "val/predictive_r2": val_rsq.predictive,
            # Read before the scheduler steps: this is the rate the epoch just
            # trained at, which is the one the rest of the row has to be read
            # against.
            "lr": self._optimization.learning_rate,
            "factors/effective": factors_summary.effective,
            "factors/max_abs_corr": factors_summary.max_abs_correlation,
            **self.probe.summary(),
            **activations.summary(),
            **weight_summary(self.model, self.cfg.train.diagnostics),
        }
        return record, val_rsq

    def fit(self) -> FitResult:
        started = time.perf_counter()
        set_seed(self.cfg.train.seed)
        self.tracker.log_config(self.cfg)
        self.tracker.log_metrics({"model/num_parameters": self.model.num_parameters()}, step=0)

        stopper = EarlyStopping(
            self.cfg.train.early_stopping_patience,
            min_delta=self.cfg.train.min_delta,
        )
        history: list[dict[str, float]] = []

        with ActivationProbe(self.model, self.cfg.train.diagnostics) as activations:
            for epoch in range(self.cfg.train.epochs):
                train_logs = self._train_epoch()
                record, val_rsq = self._epoch_metrics(train_logs, activations)
                self.scheduler.step()

                history.append(record)
                # The final epoch is always logged, whatever the interval, so a run's
                # last recorded metrics are the ones early stopping actually acted on.
                if should_log_epoch(
                    epoch,
                    interval=self.cfg.mlflow.log_every_n_epochs,
                    epochs=self.cfg.train.epochs,
                ):
                    self.tracker.log_metrics(record, step=epoch)

                if stopper.update(self._selected_metric(val_rsq), epoch, self.model):
                    break

        stopper.restore(self.model)
        # One value, reported and written. Previously the checkpoint received the
        # raw ``-inf`` of a run in which no epoch ever improved while FitResult
        # reported ``nan`` for the same run, so the two disagreed about a fit that
        # had failed.
        best = stopper.best_metric if math.isfinite(stopper.best_metric) else float("nan")
        self._save_checkpoint(stopper.best_epoch, best)
        return FitResult(
            best_metric=best,
            best_epoch=stopper.best_epoch,
            epoch_budget=self.cfg.train.epochs,
            fit_seconds=time.perf_counter() - started,
            history=history,
        )

    def _save_checkpoint(self, best_epoch: int, best_metric: float) -> None:
        save_checkpoint(
            self.model,
            directory=self.cfg.train.checkpoint_dir,
            filename=CHECKPOINT_FILE,
            config=self.cfg,
            best_epoch=best_epoch,
            best_metric=best_metric,
            tracker=self.tracker,
            # Both column tuples, in network-input order. The input widths are
            # inferred from the data, so dimension ``p`` of the beta network means
            # "the p-th of these names" and nothing else: a checkpoint without them
            # cannot be interpreted per feature after the feature config has moved
            # on.
            extras={
                "beta_columns": self.splits.beta_columns,
                "portfolio_columns": self.splits.portfolio_columns,
            },
        )
