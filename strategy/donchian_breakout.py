"""Donchian Breakout — Turtle-style channel breakout strategy.

Entry: close breaks above the N-bar Donchian upper channel.
Exit:  Chandelier trailing stop OR price falls back below channel midpoint.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr


@dataclass(frozen=True)
class DonchianBreakoutParams(StrategyParams):
    channel_period: int = 20
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    volume_confirm: bool = True

    def validate(self):
        if not (self.channel_period >= 10): raise ValueError("channel_period must be >= 10")
        if not (self.trail_atr_mult > 0): raise ValueError("trail_atr_mult must be positive")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


class DonchianBreakout(BaseStrategy):
    """Entry: close > highest high of last N bars. Exit: trailing stop or
    close < channel midpoint."""

    params: DonchianBreakoutParams

    def __init__(self, **kwargs):
        super().__init__(DonchianBreakoutParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.channel_period, self.params.atr_period, 20) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Donchian Channel
        df["Donchian_upper"] = df["High"].rolling(p.channel_period).max().shift(1)
        df["Donchian_lower"] = df["Low"].rolling(p.channel_period).min().shift(1)
        df["Donchian_mid"] = (df["Donchian_upper"] + df["Donchian_lower"]) / 2

        df["ATR"] = compute_atr(df, p.atr_period)

        # Volume MA for confirmation
        df["Volume_MA"] = df["Volume"].rolling(20).mean()

        # Entry signals only — exit is handled in check_exit
        df["Signal"] = 0
        buy = df["Close"] > df["Donchian_upper"]
        if p.volume_confirm:
            buy = buy & (df["Volume"] > df["Volume_MA"])
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

        # Channel midpoint — breakout failed
        mid = float(df["Donchian_mid"].iloc[i])
        if price <= mid:
            return True, "跌破通道中线"

        return False, ""
