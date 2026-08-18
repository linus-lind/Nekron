"""Feature creation: reusable featurizers over a ``(date, entity)`` panel.

The package turns a preprocessed ``(date, entity)`` panel into a feature panel
sharing the same index. Featurizers are small, composable, configuration-driven
steps; a :class:`FeaturePipeline` runs them in order, sharing one entity-major
layout (:class:`PanelContext`) so grouped time-series and cross-sectional
computations are done efficiently and only once per panel.

Because the steps run top to bottom, the working panel accumulates the original
columns plus every produced feature. What a run outputs is a projection of it,
described by two :class:`ColumnSelection` instances — one over the input columns,
one over the produced features — so a run can retain a subset of the raw panel and
drop the scaffolding features that only exist to feed later steps.
"""

from __future__ import annotations

from .base import FeatureError, Featurizer, KeyedFeaturizer, PanelContext
from .calendar import CalendarFeatures
from .config import (
    ColumnSelectionConfig,
    FeatureConfig,
    FeaturizerSpec,
    build_pipeline,
    register_configs,
    to_config,
    to_selection,
)
from .crosssectional import (
    CrossSectionalBucket,
    CrossSectionalDemean,
    CrossSectionalMinMax,
    CrossSectionalRank,
    CrossSectionalWinsorize,
    CrossSectionalZScore,
    GroupNeutralize,
)
from .liquidity import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    DollarVolume,
    KyleLambda,
    MarketCap,
    RollSpread,
    Turnover,
    ZeroReturnFraction,
)
from .momentum import (
    MACD,
    DistanceFromRollingExtreme,
    Momentum,
    MovingAverage,
    MovingAverageRatio,
    PriceToMovingAverage,
)
from .normalization import RollingMinMax, RollingRank, RollingZScore
from .oscillators import (
    CCI,
    RSI,
    AwesomeOscillator,
    Stochastic,
    UltimateOscillator,
    WilliamsR,
)
from .pipeline import FeaturePipeline
from .price import CandlePosition, HighLowRange, LogPrice
from .registry import (
    build_featurizer,
    register_featurizer,
    registered_featurizers,
)
from .returns import (
    CompoundReturns,
    ForwardReturns,
    IntradayReturn,
    LogReturns,
    OvernightReturn,
    SimpleReturns,
)
from .rolling import RollingAggregation
from .selection import ColumnSelection
from .temporal import Difference, TemporalShift
from .trend import ADX, Aroon, LinearTrendSlope, TrueStrengthIndex
from .volatility import (
    AverageTrueRange,
    BollingerBands,
    DownsideDeviation,
    EWMAVolatility,
    GarmanKlassVolatility,
    MaxDrawdown,
    ParkinsonVolatility,
    RealizedVolatility,
    RogersSatchellVolatility,
    UlcerIndex,
    YangZhangVolatility,
)
from .volume import (
    AccumulationDistribution,
    ChaikinMoneyFlow,
    EaseOfMovement,
    ForceIndex,
    MoneyFlowIndex,
    OnBalanceVolume,
    VolumePriceTrend,
)

__all__ = [
    # core
    "Featurizer",
    "KeyedFeaturizer",
    "FeatureError",
    "PanelContext",
    "FeaturePipeline",
    "ColumnSelection",
    # config / registry
    "ColumnSelectionConfig",
    "FeatureConfig",
    "FeaturizerSpec",
    "build_pipeline",
    "register_configs",
    "to_config",
    "to_selection",
    "build_featurizer",
    "register_featurizer",
    "registered_featurizers",
    # returns / price
    "SimpleReturns",
    "LogReturns",
    "ForwardReturns",
    "CompoundReturns",
    "IntradayReturn",
    "OvernightReturn",
    "LogPrice",
    "HighLowRange",
    "CandlePosition",
    # momentum / trend
    "Momentum",
    "MovingAverage",
    "PriceToMovingAverage",
    "MovingAverageRatio",
    "MACD",
    "DistanceFromRollingExtreme",
    "ADX",
    "Aroon",
    "TrueStrengthIndex",
    "LinearTrendSlope",
    # volatility
    "RealizedVolatility",
    "EWMAVolatility",
    "ParkinsonVolatility",
    "GarmanKlassVolatility",
    "RogersSatchellVolatility",
    "YangZhangVolatility",
    "AverageTrueRange",
    "DownsideDeviation",
    "BollingerBands",
    "UlcerIndex",
    "MaxDrawdown",
    # liquidity
    "AmihudIlliquidity",
    "DollarVolume",
    "Turnover",
    "MarketCap",
    "RollSpread",
    "CorwinSchultzSpread",
    "KyleLambda",
    "ZeroReturnFraction",
    # oscillators
    "RSI",
    "Stochastic",
    "WilliamsR",
    "CCI",
    "UltimateOscillator",
    "AwesomeOscillator",
    # volume / price-volume
    "OnBalanceVolume",
    "AccumulationDistribution",
    "ChaikinMoneyFlow",
    "MoneyFlowIndex",
    "ForceIndex",
    "EaseOfMovement",
    "VolumePriceTrend",
    # cross-sectional
    "CrossSectionalRank",
    "CrossSectionalZScore",
    "CrossSectionalDemean",
    "CrossSectionalMinMax",
    "CrossSectionalWinsorize",
    "CrossSectionalBucket",
    "GroupNeutralize",
    # time-series normalization
    "RollingZScore",
    "RollingMinMax",
    "RollingRank",
    # generic rolling / temporal / calendar
    "RollingAggregation",
    "TemporalShift",
    "Difference",
    "CalendarFeatures",
]
