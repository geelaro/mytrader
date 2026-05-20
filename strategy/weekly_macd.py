"""WeeklyMACD — single MACD crossover on weekly bars."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr, compute_macd, resample_weekly


@dataclass(frozen=True)
class WeeklyMACDParams(StrategyParams):
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    trail_atr_mult: float = 4.0
    max_position_pct: float = 0.95


class WeeklyMACD(BaseStrategy):
    """MACD golden/death cross on weekly bars + trailing stop.

    Entry: MACD crosses above signal line (golden cross) on weekly close.
    Exit:  MACD death cross OR price breaks trailing stop.
    """

    params: WeeklyMACDParams

    def __init__(self, **kwargs):
        super().__init__(WeeklyMACDParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.macd_slow, self.params.macd_signal) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        weekly = resample_weekly(df)
        weekly = compute_macd(weekly, p.macd_fast, p.macd_slow, p.macd_signal)
        weekly["ATR"] = compute_atr(weekly, 14)

        weekly["Signal"] = 0
        golden = (
            (weekly["MACD"] > weekly["MACD_signal"])
            & (weekly["MACD"].shift(1) <= weekly["MACD_signal"].shift(1))
        )
        death = (
            (weekly["MACD"] < weekly["MACD_signal"])
            & (weekly["MACD"].shift(1) >= weekly["MACD_signal"].shift(1))
        )
        weekly.loc[golden, "Signal"] = 1
        weekly.loc[death, "Signal"] = -1
        return weekly

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        if price <= 0:
            return 0
        return int(capital * self.params.max_position_pct / price)

    def check_exit(
        self, df, i, entry_price, highest_since_entry, position=None
    ) -> Tuple[bool, str]:
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        trail_stop = highest_since_entry - self.params.trail_atr_mult * atr
        if price <= trail_stop:
            return True, "移动止损"
        if int(df["Signal"].iloc[i]) == -1:
            return True, "MACD死叉"
        return False, ""
