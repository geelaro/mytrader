from .base import BaseStrategy, StrategyParams
from .enhanced_macd import EnhancedMACDStrategy, EnhancedMACDParams
from .trend_follower import TrendFollower, TrendFollowerParams
from .weekly_macd import WeeklyMACD, WeeklyMACDParams
from .weekly_macd_kdj import WeeklyMACD_KDJ, WeeklyMACDKDJParams

# Single source of truth for strategy name → class mapping
STRATEGY_MAP = {
    "enhanced_macd": EnhancedMACDStrategy,
    "trend_follower": TrendFollower,
    "weekly_macd": WeeklyMACD,
    "weekly_macd_kdj": WeeklyMACD_KDJ,
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
    "STRATEGY_MAP",
    "SIGNAL_LABEL",
]
