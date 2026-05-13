"""TrendFollower — MA + ADX filter + Chandelier trailing stop exit."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr


@dataclass(frozen=True)
class TrendFollowerParams(StrategyParams):
    short_ma: int = 20
    long_ma: int = 50
    adx_period: int = 14
    adx_threshold: float = 20.0
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    def validate(self):
        if not (self.short_ma < self.long_ma): raise ValueError("validation failed")
        if not (self.trail_atr_mult > 0): raise ValueError("validation failed")


class TrendFollower(BaseStrategy):
    """Breakout trend-follower with Chandelier trailing stop.

    Entry: MA uptrend + ADX confirms trend + +DI above -DI.
    Exit:  Chandelier trailing stop only — prices close below
           (highest_since_entry − trail_atr_mult × ATR).
    """

    params: TrendFollowerParams

    def __init__(self, **kwargs):
        super().__init__(TrendFollowerParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.long_ma, self.params.atr_period,
                   self.params.adx_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df["SMA_short"] = df["Close"].rolling(p.short_ma).mean()
        df["SMA_long"] = df["Close"].rolling(p.long_ma).mean()
        df["ATR"] = compute_atr(df, p.atr_period)

        # ADX
        high_diff = df["High"].diff()
        low_diff = -df["Low"].diff()
        plus_dm = pd.Series((high_diff > low_diff) & (high_diff > 0)) * high_diff
        plus_dm = plus_dm.clip(lower=0)
        minus_dm = pd.Series((low_diff > high_diff) & (low_diff > 0)) * low_diff
        minus_dm = minus_dm.clip(lower=0)

        atr_s = df["ATR"].replace(0, np.nan)
        plus_di = 100 * plus_dm.ewm(alpha=1 / p.adx_period, adjust=False).mean() / atr_s
        minus_di = 100 * minus_dm.ewm(alpha=1 / p.adx_period, adjust=False).mean() / atr_s

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["ADX"] = dx.ewm(alpha=1 / p.adx_period, adjust=False).mean()
        df["+DI"] = plus_di
        df["-DI"] = minus_di

        # Entry signals only
        df["Signal"] = 0
        buy = (
            (df["SMA_short"] > df["SMA_long"])
            & (df["ADX"] > p.adx_threshold)
            & (df["+DI"] > df["-DI"])
        )
        df.loc[buy, "Signal"] = 1
        return df

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        if pd.isna(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollar = capital * self.params.risk_per_trade
        stop_distance = atr * self.params.trail_atr_mult
        if stop_distance <= 0:
            return 0
        shares = int(risk_dollar / stop_distance)
        max_shares = int(capital * self.params.max_position_pct / price)
        return max(0, min(shares, max_shares))

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
