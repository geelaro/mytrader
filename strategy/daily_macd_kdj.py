"""Daily MACD_KDJ — KDJ golden-cross entry + MACD death-cross exit on daily bars."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import (
    BaseStrategy,
    StrategyParams,
    compute_atr,
    compute_kdj,
    compute_macd,
)


@dataclass(frozen=True)
class DailyMACDKDJParams(StrategyParams):
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_n: int = 9
    kdj_k: int = 3
    kdj_d: int = 3
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    def validate(self):
        if not (self.trail_atr_mult > 0): raise ValueError("validation failed")


class DailyMACD_KDJ(BaseStrategy):
    """KDJ golden cross for buy, MACD death cross for sell, on daily bars.

    Higher trade frequency than the weekly variant. ATR trailing stop
    provides downside protection between exit signals.
    """

    params: DailyMACDKDJParams

    def __init__(self, **kwargs):
        super().__init__(DailyMACDKDJParams(**kwargs))

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.macd_slow, p.macd_signal, p.kdj_n, p.atr_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        df = compute_macd(df, p.macd_fast, p.macd_slow, p.macd_signal)
        df = compute_kdj(df, p.kdj_n, p.kdj_k, p.kdj_d)
        df["ATR"] = compute_atr(df, p.atr_period)

        df["Signal"] = 0
        golden_kdj = (
            (df["K"] > df["D"])
            & (df["K"].shift(1) <= df["D"].shift(1))
        )
        death_macd = (
            (df["MACD"] < df["MACD_signal"])
            & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
        )
        df.loc[golden_kdj, "Signal"] = 1
        df.loc[death_macd, "Signal"] = -1
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

        trail_stop = highest_since_entry - atr * self.params.trail_atr_mult
        if price <= trail_stop:
            return True, "移动止损"

        if int(df["Signal"].iloc[i]) == -1:
            return True, "MACD死叉"

        return False, ""
