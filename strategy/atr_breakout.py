"""ATR Breakout — volatility-normalized breakout strategy.

Entry: close crosses above MA + N*ATR.
Exit:  Chandelier trailing stop.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr


@dataclass(frozen=True)
class ATRBreakoutParams(StrategyParams):
    ma_period: int = 20
    atr_period: int = 14
    breakout_atr_mult: float = 2.0
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    ma_type: str = "ema"

    def validate(self):
        if not (self.ma_period > 0):
            raise ValueError("ma_period must be > 0")
        if not (self.breakout_atr_mult > 0):
            raise ValueError("breakout_atr_mult must be > 0")
        if not (self.trail_atr_mult >= self.breakout_atr_mult):
            raise ValueError("trail_atr_mult must be >= breakout_atr_mult")
        if not (self.ma_type in ("ema", "sma")):
            raise ValueError("ma_type must be 'ema' or 'sma'")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be in (0, 1]")


class ATRBreakout(BaseStrategy):
    """Entry: close crosses above MA + N*ATR. Exit: Chandelier trailing stop."""

    params: ATRBreakoutParams

    def __init__(self, **kwargs):
        super().__init__(ATRBreakoutParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.ma_period, self.params.atr_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        if p.ma_type == "ema":
            df["MA"] = df["Close"].ewm(span=p.ma_period, adjust=False).mean()
        else:
            df["MA"] = df["Close"].rolling(p.ma_period).mean()

        df["ATR"] = compute_atr(df, p.atr_period)
        df["Upper_band"] = df["MA"] + p.breakout_atr_mult * df["ATR"]

        # Entry signals only
        df["Signal"] = 0
        buy = (
            (df["Close"] > df["Upper_band"])
            & (df["Close"].shift(1) <= df["Upper_band"].shift(1))
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
