"""LSTM-Transformer model over conditional-autoencoder residuals.

A fitted conditional autoencoder prices each date's cross-section and leaves a
residual ``u_{i,t} = r_{i,t} - beta(z_{i,t})' f_t`` behind — one series per
entity. This package models those series: it recovers the frozen autoencoder from
its MLflow run, replays it to build the residual panel, cuts each entity's series
into fixed-length windows wherever the whole span is present, and scores each
window with

    bidirectional LSTM -> transformer stack -> temporal pooling -> MLP score head.

Every stage is configured through Hydra and built from the reusable blocks in
:mod:`nekron.nn`; nothing that governs model behaviour is hard-coded inside the
modules.

The objective is deliberately absent — see :mod:`.losses`. Everything around it
is finished, so a run today builds the residuals, the windows, the model and the
trainer, records them, and stops short of fitting.

Typical use::

    from nekron.time_series_modelling import lstm_transformer as lt

    cfg = lt.to_config(composed_config)
    cae = lt.load_pretrained_cae(cfg.cae)
    panel = lt.build_residual_panel(cae, device=cfg.cae.device)
    splits = lt.build_window_splits(panel, cfg.data, device=lt.resolve_device(cfg.train.device))
    model = lt.LstmTransformer.from_config(
        cfg, in_dim=splits.num_channels, seq_len=splits.seq_len
    )
    with lt.MlflowTracker(cfg.mlflow) as tracker:
        result = lt.Trainer(model, cfg, splits, tracker).fit()
"""

from __future__ import annotations

from nekron.nn import DiagnosticsConfig, resolve_device, set_seed
from nekron.tracking import MlflowConfig, MlflowTracker

from .config import (
    CaeConfig,
    CrossValidationConfig,
    DataConfig,
    LossConfig,
    LstmConfig,
    LstmTransformerConfig,
    ModelConfig,
    OptimConfig,
    PoolingConfig,
    ScoreHeadConfig,
    TrainConfig,
    TransformerConfig,
    register_configs,
    to_config,
)
from .data import (
    PretrainedCae,
    ResidualError,
    ResidualPanel,
    ResidualWindows,
    WindowBatch,
    WindowError,
    WindowPanel,
    WindowSplits,
    build_residual_panel,
    build_window_panel,
    build_window_splits,
    load_pretrained_cae,
    window_schedule,
)
from .engine import (
    CHECKPOINT_FILE,
    CrossValidationError,
    CrossValidationResult,
    FitResult,
    Trainer,
    TrainerError,
    build_schedule,
    run_folds,
)
from .losses import LossBreakdown, ScoreObjective, build_objective
from .model import LstmTransformer, ScoreOutput
from .modules import ScoreHead, SequenceEncoder

__version__ = "0.1.0"

__all__ = [
    # config
    "LstmTransformerConfig",
    "CaeConfig",
    "CrossValidationConfig",
    "DataConfig",
    "ModelConfig",
    "LstmConfig",
    "TransformerConfig",
    "PoolingConfig",
    "ScoreHeadConfig",
    "LossConfig",
    "OptimConfig",
    "TrainConfig",
    "MlflowConfig",
    "DiagnosticsConfig",
    "register_configs",
    "to_config",
    # data
    "load_pretrained_cae",
    "PretrainedCae",
    "build_residual_panel",
    "ResidualPanel",
    "ResidualError",
    "build_window_splits",
    "build_window_panel",
    "window_schedule",
    "WindowPanel",
    "WindowSplits",
    "ResidualWindows",
    "WindowBatch",
    "WindowError",
    # model
    "LstmTransformer",
    "ScoreOutput",
    "SequenceEncoder",
    "ScoreHead",
    # loss
    "build_objective",
    "ScoreObjective",
    "LossBreakdown",
    # engine
    "Trainer",
    "TrainerError",
    "FitResult",
    "CHECKPOINT_FILE",
    "build_schedule",
    "run_folds",
    "CrossValidationResult",
    "CrossValidationError",
    "MlflowTracker",
    # utils
    "set_seed",
    "resolve_device",
    "__version__",
]
