"""Typed configuration for the LSTM-Transformer residual model, wired to Hydra.

The dataclasses below are the single schema *and* the source of the default
values. They are registered with Hydra's :class:`ConfigStore` so a run is composed
from ``configs/lstm_transformer.yaml`` (see :func:`register_configs`), validated
against this schema, then materialized into a typed
:class:`LstmTransformerConfig` with :func:`to_config`. Override anything from the
CLI, e.g.::

    python -m nekron.time_series_modelling.lstm_transformer \
        cae.run_id=3f9c... data.seq_len=126 model.transformer.num_heads=4

What is *not* here
------------------
There is no ``data_pipeline`` block. The residuals this model consumes are the
pricing errors of a particular fitted conditional autoencoder, and that model's
own configuration — which sources, which cleaning, which features, which target,
which split — is recovered from its MLflow run (see :mod:`.data.residuals`).
Restating it here would create a second copy that can silently disagree with the
one the autoencoder was actually fitted under, and residuals computed against a
different feature panel are not residuals at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from nekron.cv.config import CrossValidationConfig
from nekron.data_adapter.config import SplitConfig
from nekron.nn.config import OptimConfig as BaseOptimConfig
from nekron.nn.config import TrainConfig as BaseTrainConfig
from nekron.nn.pooling import POOLING_KINDS
from nekron.nn.transformer import NORM_POSITIONS, POSITIONAL_ENCODINGS
from nekron.tracking import MlflowConfig

__all__ = [
    "CaeConfig",
    "CrossValidationConfig",
    "DataConfig",
    "LossConfig",
    "LstmConfig",
    "LstmTransformerConfig",
    "ModelConfig",
    "OBJECTIVES",
    "OptimConfig",
    "PoolingConfig",
    "ScoreHeadConfig",
    "TrainConfig",
    "TransformerConfig",
    "register_configs",
    "to_config",
]
"""``CrossValidationConfig`` is re-exported from :mod:`nekron.cv.config`, and
``OptimConfig`` / ``TrainConfig`` subclass their counterparts in
:mod:`nekron.nn.config` so every model shares one optimization, training and sweep
vocabulary. The two subclasses exist only to restate the defaults this model was
tuned at, which differ from the shared ones and would otherwise be silently
inherited."""

# --------------------------------------------------------------------------- #
# The frozen conditional autoencoder
# --------------------------------------------------------------------------- #


@dataclass
class CaeConfig:
    """Which fitted conditional autoencoder produces the residuals.

    Parameters
    ----------
    run_id:
        The MLflow run of the autoencoder fit. Its checkpoint artifact carries the
        weights, the column order they were fitted against *and* the full
        configuration of the data pipeline that produced them, which is what makes
        the residuals reproducible from a run id alone.
    tracking_uri:
        Where that run lives. Defaults to the project's local store.
    checkpoint_artifact:
        Name of the checkpoint artifact inside the run.
    device:
        Where the frozen autoencoder runs while residuals are computed. ``"cpu"``
        by default rather than ``"auto"``: this is a one-off inference pass whose
        output is cached, so reproducibility is worth more than the seconds an
        accelerator would save.
    """

    run_id: str = MISSING
    tracking_uri: str = "sqlite:///mlflow.db"
    checkpoint_artifact: str = "conditional_autoencoder_best.pt"
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """How the residual series are cut into fixed-length windows.

    Parameters
    ----------
    seq_len:
        Steps per window. A window exists only where the entity has a residual on
        *every* one of those consecutive periods — a gap ends the run and the
        count restarts.
    stride:
        Keep one window in every ``stride`` end positions. Applied to the date
        axis, so the surviving windows line up across entities; ``1`` keeps every
        window.
    target_horizon:
        Steps beyond the window's last date at which the target residual is read.
        ``0`` — the default — carries no target at all, which is the honest
        setting while no objective is configured. A positive value also requires
        the target's date to fall inside the *same* split as the window it
        belongs to, so no window is ever labelled from the split after it.
    batch_size:
        Windows per optimizer step.
    shuffle:
        Whether training windows are visited in random order each epoch.
    inherit_cae_split:
        Take the train/validation/test boundaries from the autoencoder's own run.
        This is the default and is what keeps the frozen autoencoder out of sample
        on this model's test window: the dates it was fitted on stay this model's
        training dates. Set it false to cut a different schedule with
        :attr:`split`, and be explicit about why.
    split:
        The schedule used when :attr:`inherit_cae_split` is false.

    Notes
    -----
    A window is assigned to the split containing its **last** date; the periods
    before it may reach back across a boundary. That is deliberate. The alternative
    — demanding the whole window sit inside one split — costs ``seq_len - 1``
    windows at every boundary, which at the default 252 leaves a one-year
    validation window with almost nothing in it, and it buys no protection: a
    window's lookback is entirely in the past of the date it is assigned to, so
    nothing later than that date is ever read.
    """

    seq_len: int = 252
    stride: int = 1
    target_horizon: int = 0
    batch_size: int = 256
    shuffle: bool = True
    inherit_cae_split: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError(f"data.seq_len must be positive; got {self.seq_len}.")
        if self.stride < 1:
            raise ValueError(f"data.stride must be positive; got {self.stride}.")
        if self.target_horizon < 0:
            raise ValueError(
                f"data.target_horizon must not be negative; got {self.target_horizon}."
            )
        if self.batch_size < 1:
            raise ValueError(f"data.batch_size must be positive; got {self.batch_size}.")


# --------------------------------------------------------------------------- #
# Model modules
# --------------------------------------------------------------------------- #


@dataclass
class LstmConfig:
    """The recurrent encoder that reads the residual window first.

    ``bidirectional`` is the paper-faithful default and is safe here because the
    whole window precedes the date being scored: the backward pass mixes
    information across the window, never across its edge. It does mean that a
    causal transformer downstream is *not* enough to make the model as a whole
    causal step-by-step — see :attr:`TransformerConfig.causal`.
    """

    hidden_dim: int = 128
    num_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    bidirectional: bool = True
    output_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"model.lstm.hidden_dim must be positive; got {self.hidden_dim}.")
        if self.num_layers < 1:
            raise ValueError(f"model.lstm.num_layers must be positive; got {self.num_layers}.")


@dataclass
class TransformerConfig:
    """The attention stack that reads the LSTM's hidden-state sequence.

    Parameters
    ----------
    d_model:
        The stack's working width. The LSTM's output is projected to it when the
        two differ, and passed through untouched when they match.
    num_layers:
        Blocks in the stack. ``0`` is legal and means "no attention", which is the
        control a sweep over depth needs.
    num_heads:
        Attention heads. Must divide :attr:`d_model`, and under
        ``positional_encoding="rope"`` must divide it into an even head width.
    ff_dim:
        Width of each block's feed-forward hidden layer (torch's
        ``dim_feedforward``); the convention is ``4 * d_model``.
    dropout:
        Torch's single dropout rate, applied to the attention weights, inside the
        feed-forward net, and to each sublayer's output before it re-enters the
        residual stream. One knob rather than two because that is the surface
        :class:`torch.nn.TransformerEncoderLayer` exposes.
    norm_position:
        ``"pre"`` normalizes each sublayer's input and leaves an unnormalized
        residual path (the default, and what lets a deep stack train without a
        warm-up); ``"post"`` normalizes the residual sum, as in the original
        formulation.
    positional_encoding:
        ``"none"``, ``"sinusoidal"`` or ``"rope"``.
    causal:
        Whether a step may attend to later steps of the window. ``False`` lets
        every step see the whole window, which is correct here — the window lies
        entirely in the past of the date being scored. Setting it true restricts
        the *attention* only; a bidirectional LSTM upstream has already mixed the
        window in both directions, so the two together are step-wise causal only
        with ``model.lstm.bidirectional=false``.
    """

    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 1024
    activation: str = "gelu"
    dropout: float = 0.1
    bias: bool = True
    norm_position: str = "pre"
    positional_encoding: str = "rope"
    rope_base: float = 10000.0
    causal: bool = False

    def __post_init__(self) -> None:
        if self.num_layers < 0:
            raise ValueError(
                f"model.transformer.num_layers must not be negative; got {self.num_layers}."
            )
        if self.num_heads < 1:
            raise ValueError(f"model.transformer.num_heads must be positive; got {self.num_heads}.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"model.transformer.d_model ({self.d_model}) must be divisible by num_heads "
                f"({self.num_heads})."
            )
        if self.norm_position not in NORM_POSITIONS:
            raise ValueError(
                f"model.transformer.norm_position must be one of {list(NORM_POSITIONS)}; "
                f"got {self.norm_position!r}."
            )
        if self.positional_encoding not in POSITIONAL_ENCODINGS:
            raise ValueError(
                "model.transformer.positional_encoding must be one of "
                f"{list(POSITIONAL_ENCODINGS)}; got {self.positional_encoding!r}."
            )
        head_dim = self.d_model // self.num_heads
        if self.positional_encoding == "rope" and head_dim % 2 != 0:
            raise ValueError(
                f"rotary embeddings need an even head width, but d_model / num_heads = "
                f"{self.d_model} / {self.num_heads} = {head_dim}."
            )
        if self.positional_encoding == "sinusoidal" and self.d_model % 2 != 0:
            raise ValueError(f"sinusoidal encoding needs an even d_model; got {self.d_model}.")
        if self.ff_dim < 1:
            raise ValueError(f"model.transformer.ff_dim must be positive; got {self.ff_dim}.")


@dataclass
class PoolingConfig:
    """How the encoded sequence is reduced to one vector per window.

    ``kind`` is one of :data:`~nekron.nn.pooling.POOLING_KINDS`. The default
    concatenates the last step with the mean over the window, so the head sees
    both the state the encoder ended in and the level it held throughout.
    """

    kind: str = "last_mean"
    attention_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in POOLING_KINDS:
            raise ValueError(
                f"model.pooling.kind must be one of {list(POOLING_KINDS)}; got {self.kind!r}."
            )
        if self.kind == "attention" and self.attention_dim < 1:
            raise ValueError(
                f"model.pooling.attention_dim must be positive; got {self.attention_dim}."
            )


@dataclass
class ScoreHeadConfig:
    """The MLP that turns the pooled vector into scores.

    Its input width is the pooling layer's output width and is therefore never
    configured; ``out_dim`` is the number of scores per window, which the eventual
    objective decides.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64])
    out_dim: int = 1
    activation: str = "gelu"
    batch_norm: bool = False
    dropout: float = 0.0
    bias: bool = True
    track_running_stats: bool = True

    def __post_init__(self) -> None:
        if self.out_dim < 1:
            raise ValueError(f"model.head.out_dim must be positive; got {self.out_dim}.")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("model.head.hidden_dims must all be positive.")


@dataclass
class ModelConfig:
    """The four stages: recurrent encoder, attention stack, pooling, score head.

    Only the *input* width is missing, and deliberately so: it is the number of
    channels a residual window carries, which comes from the data.
    """

    lstm: LstmConfig = field(default_factory=LstmConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    head: ScoreHeadConfig = field(default_factory=ScoreHeadConfig)


# --------------------------------------------------------------------------- #
# Loss / optimization / training
# --------------------------------------------------------------------------- #

OBJECTIVES: tuple[str, ...] = ("none",)
"""Implemented training objectives. See :mod:`.losses`."""


@dataclass
class LossConfig:
    """What the scores are trained against.

    ``"none"`` is the only implemented objective: the score head produces scores,
    and nothing is yet asked of them. A run configured this way builds the data,
    the model and the trainer and stops before fitting, rather than optimizing a
    placeholder that would have to be unlearned later.
    """

    objective: str = "none"

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"loss.objective must be one of {list(OBJECTIVES)}; got {self.objective!r}. "
                "Implement it in nekron.time_series_modelling.lstm_transformer.losses and "
                "add it there."
            )


@dataclass
class OptimConfig(BaseOptimConfig):
    """Adam, at a rate suited to an attention stack.

    Everything else is :class:`nekron.nn.OptimConfig`. Only the default rate is
    restated: ``1e-3`` trains the autoencoder's shallow networks well and is too
    large here, and a default that has to be remembered in the YAML is a default
    that will eventually be forgotten.
    """

    lr: float = 1e-4


@dataclass
class TrainConfig(BaseTrainConfig):
    """Training-loop control, clipping by default.

    Everything else is :class:`nekron.nn.TrainConfig`. Attention stacks produce
    occasional very large gradients early in training, so unlike the autoencoder
    this model clips out of the box. Set the threshold from a run's
    ``grad/norm_p90`` and watch ``grad/clipped_frac``; past ~0.5 it has stopped
    being clipping and become an obscure way of lowering the learning rate.
    """

    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0


@dataclass
class LstmTransformerConfig:
    """Root configuration object aggregating every sub-config."""

    cae: CaeConfig = field(default_factory=CaeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    def __post_init__(self) -> None:
        if self.model.head.batch_norm and self.data.batch_size < 2:
            raise ValueError(
                "model.head.batch_norm requires data.batch_size >= 2 (batch normalization "
                "needs at least two windows per step)."
            )


# --------------------------------------------------------------------------- #
# Hydra integration
# --------------------------------------------------------------------------- #

CONFIG_SCHEMA_NAME = "base_lstm_transformer"


def register_configs() -> None:
    """Register the model schema with Hydra's ConfigStore. Safe to call twice."""
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=LstmTransformerConfig)


def to_config(cfg: DictConfig) -> LstmTransformerConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed config object.

    Runs each dataclass ``__post_init__``, so schema/value violations raise here.
    """
    return cast(LstmTransformerConfig, OmegaConf.to_object(cfg))
