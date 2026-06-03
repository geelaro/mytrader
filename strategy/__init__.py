from .base import (
    BaseStrategy, StrategyParams, ChandelierTrailingExit,
    register, get_strategy_map,
)
from .enhanced_macd import EnhancedMACDStrategy, EnhancedMACDParams  # deprecated — keep for golden tests
from .trend_follower import TrendFollower, TrendFollowerParams
from .weekly_macd import WeeklyMACD, WeeklyMACDParams
from .macd_kdj import MACDKDJStrategy, MACDKDJParams, WeeklyMACD_KDJ, WeeklyMACDKDJParams, DailyMACD_KDJ, DailyMACDKDJParams
from .bollinger_mean_reversion import BollingerMeanReversion, BollingerMeanReversionParams  # deprecated — kept for test imports
from .donchian_breakout import DonchianBreakout, DonchianBreakoutParams
from .atr_breakout import ATRBreakout, ATRBreakoutParams
from .bollinger_squeeze import BollingerSqueeze, BollingerSqueezeParams  # deprecated — kept for test imports
from .turtle_trading import TurtleTrading, TurtleTradingParams
from .spy_ma_breakout import SPYMABreakout, SPYMABreakoutParams
from .ensemble import StrategyEnsemble, EnsembleParams
from .rsi2_mean_reversion import RSI2MeanReversion, RSI2MeanReversionParams

# STRATEGY_MAP is populated by the @register("name") decorator on each
# strategy class (see strategy/base.py:register).  Adding a new strategy
# no longer requires touching this dict — just import its module above
# to trigger class definition (and thus the @register side effect).
# Late-registered strategies (e.g. via third-party packages) also show up
# automatically.
STRATEGY_MAP = get_strategy_map()

SIGNAL_LABEL = {1: "买入", -1: "卖出", 0: "—"}

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "ChandelierTrailingExit",
    "TrendFollower",
    "TrendFollowerParams",
    "WeeklyMACD",
    "WeeklyMACDParams",
    "MACDKDJStrategy",
    "MACDKDJParams",
    "WeeklyMACD_KDJ",
    "WeeklyMACDKDJParams",
    "DonchianBreakout",
    "DonchianBreakoutParams",
    "ATRBreakout",
    "ATRBreakoutParams",
    "TurtleTrading",
    "TurtleTradingParams",
    "DailyMACD_KDJ",
    "DailyMACDKDJParams",
    "SPYMABreakout",
    "SPYMABreakoutParams",
    "StrategyEnsemble",
    "EnsembleParams",
    "RSI2MeanReversion",
    "RSI2MeanReversionParams",
    "STRATEGY_MAP",
    "SIGNAL_LABEL",
]
# Deprecated classes (EnhancedMACDStrategy / BollingerMeanReversion /
# BollingerSqueeze) remain importable for golden-test regression coverage
# but are intentionally excluded from __all__ so IDEs no longer suggest them.
