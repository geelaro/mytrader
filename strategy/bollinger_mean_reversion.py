"""Bollinger Mean Reversion — counter-trend strategy.

Entry: price at/under lower BB + RSI oversold + RSI turned up.
Exit:  price back to mid-BB (mean reversion) OR ATR stop-loss.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr, compute_bollinger


@dataclass(frozen=True)
class BollingerMeanReversionParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_turnup: float = 3.0
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    def validate(self):
        if not (self.bb_std > 0): raise ValueError("bb_std must be positive")
        if not (0 < self.rsi_oversold < 50): raise ValueError("rsi_oversold must be in (0, 50)")
        if not (self.rsi_turnup > 0): raise ValueError("rsi_turnup must be positive")
        if not (self.atr_stop_mult > 0): raise ValueError("atr_stop_mult must be positive")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


class BollingerMeanReversion(BaseStrategy):
    """Buy at lower BB when RSI is oversold and turning up; sell at mid-BB."""

    params: BollingerMeanReversionParams

    def __init__(self, **kwargs):
        super().__init__(BollingerMeanReversionParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.bb_period, self.params.rsi_period,
                   self.params.atr_period) + 10

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df = compute_bollinger(df, p.bb_period, p.bb_std)
        df["ATR"] = compute_atr(df, p.atr_period)

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / p.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # RSI minimum over last 5 bars — used to detect turn-up
        df["RSI_low5"] = df["RSI"].rolling(5).min()

        # ---- Signals ----
        df["Signal"] = 0

        buy = (
            (df["Close"] <= df["BB_lower"])
            & (df["RSI"] < p.rsi_oversold)
            & ((df["RSI"] - df["RSI_low5"]) >= p.rsi_turnup)
        )
        df.loc[buy, "Signal"] = 1

        sell = (
            (df["Close"] >= df["BB_mid"])
            & (df["Close"].shift(1) < df["BB_mid"].shift(1))
        )
        df.loc[sell, "Signal"] = -1

        return df

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        if pd.isna(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollar = capital * self.params.risk_per_trade
        stop_distance = atr * self.params.atr_stop_mult
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
        p = self.params

        stop_loss = entry_price - atr * p.atr_stop_mult
        if price <= stop_loss:
            return True, "止损"

        # Primary exit handled by Signal == -1 (mid-BB cross)
        if int(df["Signal"].iloc[i]) == -1:
            return True, "均值回归"

        return False, ""
