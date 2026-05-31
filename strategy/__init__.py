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
from .rsi2_mean_reversion import RSI2MeanReversion, RSI2MeanReversionParams

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
    "rsi2_mean_reversion": RSI2MeanReversion,
}

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
