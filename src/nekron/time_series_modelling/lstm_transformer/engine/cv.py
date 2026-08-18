"""Cross-validated fitting of the LSTM-Transformer residual model.

The sweep itself — the schedule, the nested runs, the seeding order, the failure
policy, the summary table and the pooling discipline — is model-agnostic and lives
in :mod:`nekron.cv`, shared with every other model here. What is in this module is
only what that driver cannot know: how this model's windows are cut, how one fold
is fitted and scored, and what its numbers are called.

Pooling a loss
--------------
This model's per-period rows carry a summed loss and a window count, and the
sweep's headline number is the ratio of those two sums. That is the same
discipline the conditional autoencoder's R-squared follows and for the same
reason: folds differ in how many windows they score, so the mean of their scores
weights a thin fold like a thick one. The columns differ; the rule does not.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd
import torch

from nekron.cv import (
    FoldOutcome,
    SweepError,
    SweepReport,
    SweepResult,
    fold_checkpoint_dir,
    fold_seed,
    run_sweep,
    sweep_tags,
)
from nekron.data_adapter.splitting import FoldBounds, FoldWindow
from nekron.tracking import MlflowTracker

from ..config import LstmTransformerConfig
from ..data.residuals import ResidualPanel
from ..data.windows import WindowPanel, build_window_panel, window_schedule
from ..model import LstmTransformer
from .trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["CrossValidationError", "CrossValidationResult", "build_schedule", "run_folds"]

CrossValidationError = SweepError
"""The sweep's failure type, under this package's name for it."""

CrossValidationResult = SweepResult
"""A finished sweep. See :class:`nekron.cv.SweepResult`."""

REPORT = SweepReport(headline="test loss", spread_column="test_loss", spread_label="loss")

SPREAD_COLUMNS = ("test_loss",)

LOSS_SUM = "loss_sum"
WINDOW_COUNT = "n_windows"
"""The two columns a fold contributes, so the sweep can pool rather than average."""


def build_schedule(
    panel: ResidualPanel, cfg: LstmTransformerConfig, *, device: torch.device
) -> tuple[WindowPanel, tuple[FoldBounds, ...]]:
    """Build every window once, and cut the schedule over the periods that carry them.

    Ordered this way for the reason the sweep exists: folds overlap, so the
    residual matrix and the validity scan are done once and each fold takes a
    constant-time view.
    """
    windows = build_window_panel(panel, cfg.data, device=device)
    return windows, window_schedule(panel, cfg.data)


def run_folds(
    windows: WindowPanel,
    bounds: tuple[FoldBounds, ...],
    cfg: LstmTransformerConfig,
    tracker: MlflowTracker,
) -> CrossValidationResult:
    """Fit one model per fold, each inside its own nested run under ``tracker``."""
    tracker.log_config(cfg)

    def body(
        fold: FoldBounds, window: FoldWindow, run: MlflowTracker, num_folds: int
    ) -> FoldOutcome:
        return _run_fold(windows, fold, window, cfg, run, num_folds)

    split = windows.panel.split if cfg.data.inherit_cae_split else cfg.data.split
    return run_sweep(
        bounds,
        windows.dates,
        body,
        policy=cfg.cv,
        tracker=tracker,
        seed=cfg.train.seed,
        pool=pool_loss,
        spread_columns=SPREAD_COLUMNS,
        report=REPORT,
        tags=sweep_tags(split, len(bounds)),
    )


def _mean(rows: pd.DataFrame) -> float:
    """The loss over one fold's rows — the same ratio the sweep pools."""
    return pool_loss(rows)["loss"]


def pool_loss(rows: pd.DataFrame) -> dict[str, float]:
    """The sweep's headline loss, over every scored window of every fold.

    The ratio of the summed losses to the summed window counts, never the mean of
    the folds' means: a fold that scored a hundred windows would otherwise count
    for as much as one that scored a hundred thousand.

    A fold whose test window yielded no windows contributes ``n_windows=0`` and a
    ``NaN`` loss, and drops out rather than poisoning the total — but only because
    :meth:`pandas.Series.sum` skips missing values. :func:`numpy.sum` does not, so
    this must stay a pandas reduction; a test pins the behaviour.
    """
    if not len(rows) or WINDOW_COUNT not in rows.columns:
        return {"loss": float("nan")}
    count = float(rows[WINDOW_COUNT].sum())
    return {"loss": float(rows[LOSS_SUM].sum()) / count if count > 0.0 else float("nan")}


def _run_fold(
    windows: WindowPanel,
    bounds: FoldBounds,
    window: FoldWindow,
    cfg: LstmTransformerConfig,
    run: MlflowTracker,
    num_folds: int,
) -> FoldOutcome:
    """One fold, start to finish. The driver has already reseeded and opened ``run``."""
    fold_cfg = _fold_config(cfg, bounds.index, num_folds)
    splits = windows.splits(bounds)
    model = LstmTransformer.from_config(
        fold_cfg, in_dim=splits.num_channels, seq_len=splits.seq_len
    )
    trainer = Trainer(model, fold_cfg, splits, run)
    fit = trainer.fit()

    # One row per scored date, carrying the sums the pooling is a ratio of rather
    # than the ratio itself: a mean cannot be recombined across folds, its
    # numerator and denominator can. Per *date* rather than per fold, so the
    # sweep's period count means what it says and the result carries the
    # distribution across dates that a single fold-level row would not.
    rows = trainer.evaluate_per_period(splits.test)
    test_loss = _mean(rows)
    val_loss = trainer.evaluate(splits.val)
    del window
    return FoldOutcome(
        scores={"val_loss": val_loss, "test_loss": test_loss},
        metrics={"test/loss": test_loss, "val/final_loss": val_loss},
        rows=rows,
        best_epoch=fit.best_epoch,
        best_metric=fit.best_metric,
    )


def _fold_config(cfg: LstmTransformerConfig, index: int, num_folds: int) -> LstmTransformerConfig:
    """``cfg`` as fold ``index`` sees it: its own seed, its own checkpoint directory.

    A single-fold schedule under the default seeding is handed the configuration
    unchanged, so an ordinary run still writes its checkpoint where it always has.
    The two-line :func:`dataclasses.replace` stays here rather than in
    :mod:`nekron.cv` because ``replace`` is typed against a concrete dataclass; the
    *policy* it applies is shared.
    """
    if num_folds == 1 and cfg.cv.seed_mode == "fixed":
        return cfg
    return replace(
        cfg,
        train=replace(
            cfg.train,
            seed=fold_seed(cfg.train.seed, index, cfg.cv.seed_mode),
            checkpoint_dir=fold_checkpoint_dir(cfg.train.checkpoint_dir, index, num_folds),
        ),
    )
