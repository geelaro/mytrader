"""WeeklyMACD_KDJ — KDJ golden-cross entry + MACD death-cross exit, weekly."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import (
    BaseStrategy,
    StrategyParams,
    compute_atr,
    compute_kdj,
    compute_macd,
    resample_weekly,
)


@dataclass(frozen=True)
class WeeklyMACDKDJParams(StrategyParams):
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_n: int = 9
    kdj_k: int = 3
    kdj_d: int = 3
    trail_atr_mult: float = 4.0
    max_position_pct: float = 0.95


class WeeklyMACD_KDJ(BaseStrategy):
    """Weekly KDJ golden cross for buy, MACD death cross + trailing stop for sell.

    Entry: K line crosses above D line (KDJ golden cross).
    Exit:  MACD death cross OR price breaks trailing stop.
    """

    params: WeeklyMACDKDJParams

    def __init__(self, **kwargs):
        super().__init__(WeeklyMACDKDJParams(**kwargs))

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.macd_slow, p.macd_signal, p.kdj_n) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        weekly = resample_weekly(df)
        weekly = compute_macd(weekly, p.macd_fast, p.macd_slow, p.macd_signal)
        weekly = compute_kdj(weekly, p.kdj_n, p.kdj_k, p.kdj_d)
        weekly["ATR"] = compute_atr(weekly, 14)

        weekly["Signal"] = 0
        golden_kdj = (
            (weekly["K"] > weekly["D"])
            & (weekly["K"].shift(1) <= weekly["D"].shift(1))
        )
        death_macd = (
            (weekly["MACD"] < weekly["MACD_signal"])
            & (weekly["MACD"].shift(1) >= weekly["MACD_signal"].shift(1))
        )
        weekly.loc[golden_kdj, "Signal"] = 1
        weekly.loc[death_macd, "Signal"] = -1
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
