"""Conditional Autoencoder asset-pricing model.

A PyTorch + MLflow implementation of Gu, Kelly & Xiu, "Autoencoder Asset Pricing
Models" (Journal of Econometrics, 2021): a conditional latent-factor model in
which factor loadings are a neural-network function of per-stock input columns
(the beta network) and the latent factors are a neural-network function of the
managed portfolios ``x = (Z^T Z)^{-1} Z^T r`` (the factor network). Their inner product
reconstructs the cross-section of returns.

Typical use — the data adapter builds the panel and splits it, then this package
turns each date into one training example::

    from nekron.asset_pricing import conditional_autoencoder as cae
    from nekron.data_adapter import build_datasets

    cfg = cae.to_config(composed_config)
    splits = build_datasets(cfg.data_pipeline)
    panels = cae.build_panel_splits(splits, cfg.data)
    model = cae.ConditionalAutoencoder.from_config(
        cfg,
        num_beta_columns=panels.num_beta_columns,
        num_portfolios=panels.num_portfolios,
    )
    with cae.MlflowTracker(cfg.mlflow) as tracker:
        result = cae.Trainer(model, cfg, panels, tracker).fit()
"""

from __future__ import annotations

from nekron.nn import DiagnosticsConfig, build_activation, resolve_device, set_seed
from nekron.tracking import MlflowConfig, MlflowTracker

from .analysis import (
    FoldArtifacts,
    characteristic_importance,
    fold_importances,
    importance_summary,
    load_folds,
)
from .config import (
    BetaNetworkConfig,
    ConditionalAutoencoderConfig,
    CrossValidationConfig,
    DataConfig,
    FactorForecastConfig,
    FactorNetworkConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    register_configs,
    to_config,
)
from .data import (
    CrossSection,
    CrossSectionDataset,
    CrossSectionPanel,
    PanelSplits,
    build_cross_section_panel,
    build_panel_splits,
)
from .engine import (
    DATA_KEY_PARAM,
    CrossValidationError,
    CrossValidationResult,
    FitResult,
    FoldResult,
    InferenceResult,
    Trainer,
    build_schedule,
    pooled_segment,
    pooled_validation,
    run_folds,
)
from .losses import CaeLoss, LossBreakdown
from .metrics import (
    FactorDiagnostics,
    FactorForecast,
    RSquared,
    SplitScore,
    factor_diagnostics,
    per_period_errors,
    pooled_r_squared,
    r_squared,
)
from .model import CaeOutput, ConditionalAutoencoder
from .modules import MLP, BetaNetwork, FactorNetwork

__version__ = "0.1.0"

__all__ = [
    # config
    "ConditionalAutoencoderConfig",
    "DataConfig",
    "ModelConfig",
    "BetaNetworkConfig",
    "FactorNetworkConfig",
    "LossConfig",
    "FactorForecastConfig",
    "OptimConfig",
    "TrainConfig",
    "CrossValidationConfig",
    "MlflowConfig",
    "DiagnosticsConfig",
    "register_configs",
    "to_config",
    "build_panel_splits",
    "build_cross_section_panel",
    "PanelSplits",
    "CrossSectionPanel",
    "CrossSection",
    "CrossSectionDataset",
    "ConditionalAutoencoder",
    "CaeOutput",
    "BetaNetwork",
    "FactorNetwork",
    "MLP",
    "CaeLoss",
    "LossBreakdown",
    "r_squared",
    "RSquared",
    "factor_diagnostics",
    "FactorDiagnostics",
    "per_period_errors",
    "pooled_r_squared",
    "SplitScore",
    "FactorForecast",
    "Trainer",
    "FitResult",
    "InferenceResult",
    "MlflowTracker",
    "build_schedule",
    "run_folds",
    "pooled_segment",
    "pooled_validation",
    "DATA_KEY_PARAM",
    "CrossValidationResult",
    "CrossValidationError",
    "FoldResult",
    "load_folds",
    "FoldArtifacts",
    "characteristic_importance",
    "fold_importances",
    "importance_summary",
    "set_seed",
    "resolve_device",
    "build_activation",
    "__version__",
]
