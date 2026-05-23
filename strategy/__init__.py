from .base import BaseStrategy, StrategyParams
from .enhanced_macd import EnhancedMACDStrategy, EnhancedMACDParams  # deprecated — keep for golden tests
from .trend_follower import TrendFollower, TrendFollowerParams
from .weekly_macd import WeeklyMACD, WeeklyMACDParams
from .weekly_macd_kdj import WeeklyMACD_KDJ, WeeklyMACDKDJParams
from .bollinger_mean_reversion import BollingerMeanReversion, BollingerMeanReversionParams  # deprecated — kept for test imports
from .donchian_breakout import DonchianBreakout, DonchianBreakoutParams
from .atr_breakout import ATRBreakout, ATRBreakoutParams
from .bollinger_squeeze import BollingerSqueeze, BollingerSqueezeParams  # deprecated — kept for test imports
from .turtle_trading import TurtleTrading, TurtleTradingParams
from .daily_macd_kdj import DailyMACD_KDJ, DailyMACDKDJParams

STRATEGY_MAP = {
    "trend_follower": TrendFollower,
    "weekly_macd": WeeklyMACD,
    "weekly_macd_kdj": WeeklyMACD_KDJ,
    "donchian_breakout": DonchianBreakout,
    "atr_breakout": ATRBreakout,
    "turtle_trading": TurtleTrading,
    "daily_macd_kdj": DailyMACD_KDJ,
}

SIGNAL_LABEL = {1: "买入", -1: "卖出", 0: "—"}

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "EnhancedMACDStrategy",
    "EnhancedMACDParams",
    "TrendFollower",
    "TrendFollowerParams",
    "WeeklyMACD",
    "WeeklyMACDParams",
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
    "STRATEGY_MAP",
    "SIGNAL_LABEL",
]
