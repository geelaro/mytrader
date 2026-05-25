from .base import BaseStrategy, StrategyParams, ChandelierTrailingExit
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

STRATEGY_MAP = {
    "trend_follower": TrendFollower,
    "weekly_macd": WeeklyMACD,
    "macd_kdj": MACDKDJStrategy,
    "weekly_macd_kdj": WeeklyMACD_KDJ,
    "donchian_breakout": DonchianBreakout,
    "atr_breakout": ATRBreakout,
    "turtle_trading": TurtleTrading,
    "daily_macd_kdj": DailyMACD_KDJ,
    "spy_ma_breakout": SPYMABreakout,
}

SIGNAL_LABEL = {1: "买入", -1: "卖出", 0: "—"}

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "ChandelierTrailingExit",
    "EnhancedMACDStrategy",
    "EnhancedMACDParams",
    "TrendFollower",
    "TrendFollowerParams",
    "WeeklyMACD",
    "WeeklyMACDParams",
    "MACDKDJStrategy",
    "MACDKDJParams",
    "WeeklyMACD_KDJ",
    "WeeklyMACDKDJParams",
    "BollingerMeanReversion",
    "BollingerMeanReversionParams",
    "DonchianBreakout",
    "DonchianBreakoutParams",
    "ATRBreakout",
    "ATRBreakoutParams",
    "BollingerSqueeze",
    "BollingerSqueezeParams",
    "TurtleTrading",
    "TurtleTradingParams",
    "DailyMACD_KDJ",
    "DailyMACDKDJParams",
    "SPYMABreakout",
    "SPYMABreakoutParams",
    "StrategyEnsemble",
    "EnsembleParams",
    "STRATEGY_MAP",
    "SIGNAL_LABEL",
]
