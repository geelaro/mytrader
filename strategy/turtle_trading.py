"""Turtle Trading — dual-SMA trend filter + Donchian breakout + ATR trailing stop.

Adapted from the "大哥2.2" strategy for stock trading.
Entry: close breaks above N-day Donchian upper AND short SMA > long SMA.
Exit:  Chandelier trailing stop (close <= highest_since_entry - ATR * N).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr


@dataclass(frozen=True)
class TurtleTradingParams(StrategyParams):
    short_period: int = 20
    long_period: int = 50
    channel_period: int = 20
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    def validate(self):
        if not (self.short_period < self.long_period): raise ValueError("short_period must be < long_period")
        if not (self.channel_period >= 10): raise ValueError("channel_period must be >= 10")
        if not (self.trail_atr_mult > 0): raise ValueError("trail_atr_mult must be > 0")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


class TurtleTrading(BaseStrategy):
    """Dual-SMA trend filter + Donchian channel breakout + ATR trailing stop.

    Only takes long entries when the short-term SMA is above the long-term SMA,
    confirming an uptrend. Entry triggers on a Donchian channel breakout.
    Exit uses a Chandelier trailing stop from the highest price since entry.
    """

    params: TurtleTradingParams

    def __init__(self, **kwargs):
        super().__init__(TurtleTradingParams(**kwargs))

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.long_period, p.channel_period, p.atr_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Dual SMA — trend direction
        df["SMA_short"] = df["Close"].rolling(p.short_period).mean()
        df["SMA_long"] = df["Close"].rolling(p.long_period).mean()

        # Donchian Channel
        df["Donchian_upper"] = df["High"].rolling(p.channel_period).max().shift(1)
        df["Donchian_lower"] = df["Low"].rolling(p.channel_period).min().shift(1)

        df["ATR"] = compute_atr(df, p.atr_period)

        # Entry signals only — exit via check_exit
        df["Signal"] = 0
        buy = (
            (df["SMA_short"] > df["SMA_long"])
            & (df["Close"] > df["Donchian_upper"])
        )
        df.loc[buy, "Signal"] = 1

        return df

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr,
            self.params.risk_per_trade, self.params.trail_atr_mult,
            self.params.max_position_pct)

    # ------------------------------------------------------------------

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])

        chandelier_stop = highest_since_entry - self.params.trail_atr_mult * atr
        if price <= chandelier_stop:
            return True, "移动止损"

        return False, ""
