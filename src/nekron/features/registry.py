"""Registry mapping a featurizer type name to its class.

Feature configurations name featurizers by a short string and pass a parameter
mapping; :func:`build_featurizer` looks up the class and instantiates it. Adding a
featurizer means writing a class satisfying :class:`~.base.Featurizer` and
registering it here under a name — no calling code changes.

List-valued parameters (the common case, since featurizers take tuples of windows
and output names) are coerced from the lists produced by configuration into the
tuples the frozen-dataclass featurizers expect.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import FeatureError, Featurizer
from .calendar import CalendarFeatures
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
from .price import CandlePosition, HighLowRange, LogPrice
from .returns import (
    CompoundReturns,
    ForwardReturns,
    IntradayReturn,
    LogReturns,
    OvernightReturn,
    SimpleReturns,
)
from .rolling import RollingAggregation
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

_REGISTRY: dict[str, type[Featurizer]] = {}


def register_featurizer(name: str, cls: type[Featurizer]) -> None:
    """Register a featurizer class under a type ``name`` (overwrites in place)."""
    _REGISTRY[name] = cls


def registered_featurizers() -> tuple[str, ...]:
    """Return the sorted names of all registered featurizer types."""
    return tuple(sorted(_REGISTRY))


def build_featurizer(name: str, params: Mapping[str, Any]) -> Featurizer:
    """Construct the featurizer registered under ``name`` from ``params``.

    List parameters are converted to tuples to match featurizer field types.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise FeatureError(
            f"unknown featurizer type {name!r}; registered types: {registered_featurizers()}."
        ) from None
    coerced = {
        key: tuple(value) if isinstance(value, list) else value for key, value in params.items()
    }
    try:
        return cls(**coerced)
    except (TypeError, ValueError, KeyError) as exc:
        raise FeatureError(f"cannot build featurizer {name!r}: {exc}.") from exc


_DEFAULTS: dict[str, type[Featurizer]] = {
    # returns
    "simple_returns": SimpleReturns,
    "log_returns": LogReturns,
    "forward_returns": ForwardReturns,
    "compound_returns": CompoundReturns,
    "intraday_return": IntradayReturn,
    "overnight_return": OvernightReturn,
    # price action
    "log_price": LogPrice,
    "high_low_range": HighLowRange,
    "candle_position": CandlePosition,
    # momentum / trend
    "momentum": Momentum,
    "moving_average": MovingAverage,
    "price_to_moving_average": PriceToMovingAverage,
    "moving_average_ratio": MovingAverageRatio,
    "macd": MACD,
    "distance_from_rolling_extreme": DistanceFromRollingExtreme,
    "adx": ADX,
    "aroon": Aroon,
    "true_strength_index": TrueStrengthIndex,
    "linear_trend_slope": LinearTrendSlope,
    # volatility
    "realized_volatility": RealizedVolatility,
    "ewma_volatility": EWMAVolatility,
    "parkinson_volatility": ParkinsonVolatility,
    "garman_klass_volatility": GarmanKlassVolatility,
    "rogers_satchell_volatility": RogersSatchellVolatility,
    "yang_zhang_volatility": YangZhangVolatility,
    "average_true_range": AverageTrueRange,
    "downside_deviation": DownsideDeviation,
    "bollinger_bands": BollingerBands,
    "ulcer_index": UlcerIndex,
    "max_drawdown": MaxDrawdown,
    # liquidity
    "amihud_illiquidity": AmihudIlliquidity,
    "dollar_volume": DollarVolume,
    "turnover": Turnover,
    "market_cap": MarketCap,
    "roll_spread": RollSpread,
    "corwin_schultz_spread": CorwinSchultzSpread,
    "kyle_lambda": KyleLambda,
    "zero_return_fraction": ZeroReturnFraction,
    # oscillators
    "rsi": RSI,
    "stochastic": Stochastic,
    "williams_r": WilliamsR,
    "cci": CCI,
    "ultimate_oscillator": UltimateOscillator,
    "awesome_oscillator": AwesomeOscillator,
    # volume / price-volume
    "on_balance_volume": OnBalanceVolume,
    "accumulation_distribution": AccumulationDistribution,
    "chaikin_money_flow": ChaikinMoneyFlow,
    "money_flow_index": MoneyFlowIndex,
    "force_index": ForceIndex,
    "ease_of_movement": EaseOfMovement,
    "volume_price_trend": VolumePriceTrend,
    # cross-sectional
    "cross_sectional_rank": CrossSectionalRank,
    "cross_sectional_zscore": CrossSectionalZScore,
    "cross_sectional_demean": CrossSectionalDemean,
    "cross_sectional_minmax": CrossSectionalMinMax,
    "cross_sectional_winsorize": CrossSectionalWinsorize,
    "cross_sectional_bucket": CrossSectionalBucket,
    "group_neutralize": GroupNeutralize,
    # time-series normalization
    "rolling_zscore": RollingZScore,
    "rolling_minmax": RollingMinMax,
    "rolling_rank": RollingRank,
    # generic rolling / temporal / calendar
    "rolling_aggregation": RollingAggregation,
    "temporal_shift": TemporalShift,
    "difference": Difference,
    "calendar_features": CalendarFeatures,
}

for _name, _cls in _DEFAULTS.items():
    register_featurizer(_name, _cls)
