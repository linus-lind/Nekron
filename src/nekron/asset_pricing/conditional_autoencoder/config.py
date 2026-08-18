"""Typed configuration for the conditional autoencoder, wired to Hydra.

The dataclasses below are the single schema *and* the source of the default values.
They are registered with Hydra's :class:`ConfigStore` so a run is composed from the
single ``configs/conditional_autoencoder.yaml`` (see
:func:`register_configs`), validated against this schema, then materialized into a
typed :class:`ConditionalAutoencoderConfig` with :func:`to_config`. Override anything
from the CLI, e.g.::

    python -m nekron.asset_pricing.conditional_autoencoder \
        optim.lr=1e-3 train.epochs=50 model.num_factors=6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from nekron.cv.config import (
    FOLD_ERROR_POLICIES,
    SEED_MODES,
    CrossValidationConfig,
    FoldErrorPolicy,
    SeedMode,
)
from nekron.data_adapter import DataAdapterConfig
from nekron.data_adapter import register_configs as register_data_adapter_configs
from nekron.nn.config import OptimConfig
from nekron.nn.config import TrainConfig as BaseTrainConfig
from nekron.tracking import MlflowConfig

from .metrics import FactorForecast

__all__ = [
    "BetaNetworkConfig",
    "ConditionalAutoencoderConfig",
    "CrossValidationConfig",
    "DataConfig",
    "FACTOR_FORECAST_MODES",
    "FactorForecastConfig",
    "FactorNetworkConfig",
    "FoldErrorPolicy",
    "FOLD_ERROR_POLICIES",
    "L1_MODES",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "SEED_MODES",
    "SeedMode",
    "TrainConfig",
    "register_configs",
    "to_config",
]
"""``CrossValidationConfig`` and its vocabulary are re-exported from
:mod:`nekron.cv.config`, where they moved so every model could share one sweep
policy. They stay importable from here because that is where this model's
configuration has always been assembled."""

L1_MODES = frozenset({"proximal", "subgradient"})
"""How the L1 penalty reaches the optimizer; see :class:`LossConfig.l1_mode`."""

FACTOR_FORECAST_MODES = frozenset({"expanding_mean", "ewma"})
"""How predictive R-squared forecasts next period's factors; see
:class:`FactorForecastConfig.mode`."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Panel-data ingestion and per-period cross-section settings.

    Input is a :class:`pandas.DataFrame` with a two-level MultiIndex
    ``(date, entity)`` whose columns hold the beta-network input features, the
    portfolio-forming characteristics and the realized return. Each date's
    cross-section becomes one training example: the beta-input matrix ``Z_beta``,
    the portfolio-characteristic matrix ``Z_port`` and the return vector ``r``.

    Parameters
    ----------
    portfolio_columns:
        Characteristics used to form the managed portfolios; empty uses every beta
        column.
    return_column:
        The prediction target.
    exclude_columns:
        Further columns to keep out of the beta inputs. The beta set is otherwise
        *every* column except the target, which is only safe while the target is
        the only forward-looking column in the panel — adding a second horizon to
        the forward-returns featurizer would otherwise feed the model the future
        without anything complaining. Name any such column here.
    standardize:
        Whether to rank-transform the characteristics and z-score the returns
        within each period.
    min_cross_section:
        Minimum number of stocks with a finite return for a period to be used. It
        must exceed ``len(portfolio_columns)``; see
        :class:`ConditionalAutoencoderConfig`.
    batch_periods:
        Number of periods per optimizer step.
    """

    portfolio_columns: list[str] = field(default_factory=list)
    return_column: str = MISSING
    exclude_columns: list[str] = field(default_factory=list)
    date_level: int = 0
    entity_level: int = 1
    standardize: bool = True
    min_cross_section: int = 2
    batch_periods: int = 1


# --------------------------------------------------------------------------- #
# Model modules
# --------------------------------------------------------------------------- #


@dataclass
class BetaNetworkConfig:
    """Factor-loading network mapping per-stock beta inputs to loadings.

    A fully connected stack ``P -> hidden_dims -> K`` applied row-wise to each
    stock in the cross-section. An empty :attr:`hidden_dims` yields a single
    linear map.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [32, 16, 8])
    activation: str = "relu"
    batch_norm: bool = False
    dropout: float = 0.0
    bias: bool = True
    track_running_stats: bool = True


@dataclass
class FactorNetworkConfig:
    """Factor network mapping the managed portfolios to latent factors.

    A fully connected stack ``P -> hidden_dims -> K`` applied once per period to
    the managed-portfolio vector. An empty :attr:`hidden_dims` yields a single
    linear map.
    """

    hidden_dims: list[int] = field(default_factory=list)
    activation: str = "relu"
    batch_norm: bool = False
    dropout: float = 0.0
    bias: bool = False
    track_running_stats: bool = True


@dataclass
class ModelConfig:
    """Top-level model architecture: factor count plus both networks.

    :attr:`num_factors` (``K``) is the shared output width of both networks, over
    which the reconstruction inner product ``beta . f`` is taken. The two input
    widths are inferred from the data — the beta network from the number of feature
    columns, the factor network from the number of portfolio columns.
    """

    num_factors: int = 5
    beta_network: BetaNetworkConfig = field(default_factory=BetaNetworkConfig)
    factor_network: FactorNetworkConfig = field(default_factory=FactorNetworkConfig)

    def __post_init__(self) -> None:
        if self.num_factors <= 0:
            raise ValueError("num_factors must be positive.")
        if any(width <= 0 for width in self.beta_network.hidden_dims):
            raise ValueError("beta_network.hidden_dims must all be positive.")
        if any(width <= 0 for width in self.factor_network.hidden_dims):
            raise ValueError("factor_network.hidden_dims must all be positive.")
        if self.factor_network.batch_norm:
            raise ValueError(
                "factor_network.batch_norm is unsupported: the factor network consumes a "
                "single managed-portfolio vector per period, which has no batch dimension."
            )


# --------------------------------------------------------------------------- #
# Loss / optimization / training
# --------------------------------------------------------------------------- #


@dataclass
class LossConfig:
    """Mean squared pricing error plus an L1 (LASSO) penalty on the weights.

    Parameters
    ----------
    l1_lambda:
        The LASSO coefficient of ``MSE + l1_lambda * sum(|W|)``, applied to the
        weight matrices (``ndim >= 2``) of both networks. Its meaning depends on
        :attr:`l1_mode`, and the two are not on the same scale: see below.
    l1_mode:
        How the penalty reaches the optimizer.

        ``"proximal"`` keeps it out of the loss and soft-thresholds the weights
        after each step (see :mod:`nekron.nn.proximal`). Weights shrink by
        ``lr * l1_lambda`` per step, land on exact zeros, and can leave zero again
        if the pricing error asks them to.

        ``"subgradient"`` adds ``l1_lambda * sum(|W|)`` to the loss, which is the
        textbook formulation and the wrong one under an adaptive optimizer. The
        penalty's gradient is ``l1_lambda * sign(w)`` — constant magnitude, stable
        sign — so Adam's normalization divides ``l1_lambda`` out and shrinks at
        ``lr`` instead. It is retained to reproduce runs recorded under it.

        The rate differs by a factor of ``1 / l1_lambda``, so a coefficient carried
        across from a subgradient run will be inert: at ``lr=1e-3``, a proximal
        ``l1_lambda`` near ``1e-1`` reproduces the pressure a subgradient
        ``1e-4`` applied. Re-tune it against ``weights/zero_frac`` rather than
        translating it.
    """

    l1_lambda: float = 1e-4
    l1_mode: str = "proximal"

    def __post_init__(self) -> None:
        if self.l1_lambda < 0.0:
            raise ValueError(f"loss.l1_lambda must not be negative; got {self.l1_lambda}.")
        if self.l1_mode not in L1_MODES:
            raise ValueError(
                f"loss.l1_mode must be one of {sorted(L1_MODES)}; got {self.l1_mode!r}."
            )


@dataclass
class TrainConfig(BaseTrainConfig):
    """Training-loop control, plus which validation statistic selects the model.

    Everything else is :class:`nekron.nn.TrainConfig`. ``selection_metric`` stays
    here because it names a *statistic this model computes*: it is one of the two
    R-squareds in :mod:`..metrics`, and both are higher-is-better, which is not
    true of every model's objective.
    """

    early_stopping_patience: int = 5
    grad_clip_norm: float = 0.0
    selection_metric: str = "total_r2"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class FactorForecastConfig:
    """How predictive R-squared forecasts the next period's factors.

    The predictive statistic prices a period against ``lambda_{t-1}``, a forecast of
    that period's factors built from earlier periods only. This chooses the
    estimator. Total R-squared is untouched by it — that statistic scores the factor
    the model actually fitted for the period it is scoring, so there is nothing to
    forecast.

    The forecast is warmed on the training window and then carried, in date order,
    through validation and into test, whichever estimator is chosen: see
    :class:`~..metrics.FactorForecast` for the recursion both share.

    Parameters
    ----------
    mode:
        ``"expanding_mean"`` is the prevailing historical mean of every factor
        observed so far — the estimator this statistic has always used, and the one
        that weights a factor from five years ago exactly as heavily as last
        period's. ``"ewma"`` discounts the past geometrically, which is the right
        family if the factor premia move; it is also the higher-variance one, so a
        gain has to clear that. Under a premium that does *not* move, the
        exponentially weighted forecast carries roughly ``alpha * n / 2`` times the
        variance of the mean it replaces, which is why any gain shows up at long
        memories rather than short ones.
    ewma_alpha:
        The weight the newest factor enters with, in ``(0, 1]``: larger forgets
        faster, ``1.0`` forecasts last period's factor and nothing else. A factor
        ``k`` periods old carries weight ``(1 - alpha)^k``, so the memory is easiest
        read as a halflife of ``ln(0.5) / ln(1 - alpha)`` *periods* — dates, not
        calendar days — which is what the default states. Read only under
        ``mode: ewma`` and inert otherwise, though it is still validated and still
        recorded as a run parameter, so a parameter diff between two
        ``expanding_mean`` runs can show it moving with no number moving behind it.
    """

    mode: str = "expanding_mean"
    ewma_alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.mode not in FACTOR_FORECAST_MODES:
            raise ValueError(
                f"factor_forecast.mode must be one of {sorted(FACTOR_FORECAST_MODES)}; "
                f"got {self.mode!r}."
            )
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(
                f"factor_forecast.ewma_alpha must lie in (0, 1]; got {self.ewma_alpha}."
            )

    def build(self) -> FactorForecast:
        """A forecast of the configured kind, with no history observed yet.

        The one place a mode string becomes an estimator, so no other code has to
        know the vocabulary — and no evaluation path can default to an expanding
        mean while the configuration says ``ewma``.
        """
        if self.mode == "ewma":
            return FactorForecast.ewma(self.ewma_alpha)
        return FactorForecast.expanding_mean()


@dataclass
class ConditionalAutoencoderConfig:
    """Root configuration object aggregating every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    data_pipeline: DataAdapterConfig = field(default_factory=DataAdapterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    factor_forecast: FactorForecastConfig = field(default_factory=FactorForecastConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    def __post_init__(self) -> None:
        if self.model.beta_network.batch_norm and self.data.min_cross_section < 2:
            raise ValueError(
                "beta_network.batch_norm requires data.min_cross_section >= 2 "
                "(batch normalization needs at least two stocks per period)."
            )
        portfolios = len(self.data.portfolio_columns)
        if portfolios and self.data.min_cross_section <= portfolios:
            raise ValueError(
                f"data.min_cross_section ({self.data.min_cross_section}) must exceed the "
                f"number of portfolio_columns ({portfolios}). The managed portfolios solve "
                "a cross-sectional regression of returns on those characteristics, so a "
                "period with no more stocks than characteristics fits it exactly — the "
                "portfolio vector handed to the factor network is then a lossless encoding "
                "of that period's realized returns, which is the target."
            )


# --------------------------------------------------------------------------- #
# Hydra integration
# --------------------------------------------------------------------------- #

CONFIG_SCHEMA_NAME = "base_conditional_autoencoder"


def register_configs() -> None:
    """Register the model schema and the data-adapter schemas with Hydra's ConfigStore."""
    register_data_adapter_configs()
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=ConditionalAutoencoderConfig)


def to_config(cfg: DictConfig) -> ConditionalAutoencoderConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed config object.

    Runs each dataclass ``__post_init__``, so schema/value violations raise here.
    """
    return cast(ConditionalAutoencoderConfig, OmegaConf.to_object(cfg))
