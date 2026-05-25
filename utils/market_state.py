"""Market state classifier — trend regime + volatility level from OHLCV data.

Usage:
    from utils.market_state import MarketStateClassifier

    classifier = MarketStateClassifier(spy_df)
    state = classifier.classify()
    print(state.regime, state.volatility)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    TRANSITIONAL = auto()


class Volatility(Enum):
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@dataclass
class MarketState:
    regime: MarketRegime
    volatility: Volatility
    adx: float
    ma20: float
    ma50: float
    ma200: float
    bb_width_pct: float


def is_trend_strategy(name: str, regime_map: Dict[str, Optional[str]]) -> bool:
    """Return True if *name* is a trend-following strategy.

    *regime_map* is ``{strategy_name: regime | None}``, typically built from
    ``STRATEGY_MAP`` by the caller to avoid a circular import from
    ``utils.market_state`` → ``strategy``.
    """
    return regime_map.get(name) == "trend"


def is_mean_reversion_strategy(name: str, regime_map: Dict[str, Optional[str]]) -> bool:
    """Return True if *name* is a mean-reversion strategy."""
    return regime_map.get(name) == "mean_reversion"


class MarketStateClassifier:
    """Compute market regime and volatility from a proxy (e.g. SPY) daily OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must have Open/High/Low/Close columns, date-sorted).
    adx_threshold : float
        ADX above this = trending, below = ranging (default 25).
    ma_short, ma_mid, ma_long : int
        MA periods for trend alignment.
    bb_period : int
        Bollinger Band calculation period.
    bb_lookback : int
        Lookback window for bandwidth percentile ranking.
    vol_high_pct, vol_low_pct : float
        Percentile thresholds for HIGH / LOW volatility.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        adx_threshold: float = 25.0,
        ma_short: int = 20,
        ma_mid: int = 50,
        ma_long: int = 200,
        bb_period: int = 20,
        bb_lookback: int = 252,
        vol_high_pct: float = 80.0,
        vol_low_pct: float = 20.0,
    ):
        self.df = df.copy()
        self.adx_threshold = adx_threshold
        self.ma_short = ma_short
        self.ma_mid = ma_mid
        self.ma_long = ma_long
        self.bb_period = bb_period
        self.bb_lookback = bb_lookback
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self._calculated = False

    def calculate(self):
        """Compute ADX, MAs, BB bandwidth, and percentile on the proxy data."""
        df = self.df
        if len(df) < max(self.ma_long, self.bb_lookback):
            self._calculated = True
            return  # not enough data, fallback defaults

        from strategy.base import compute_adx, compute_bollinger

        # MAs
        df["MA_short"] = df["Close"].rolling(self.ma_short).mean()
        df["MA_mid"] = df["Close"].rolling(self.ma_mid).mean()
        df["MA_long"] = df["Close"].rolling(self.ma_long).mean()

        # ADX (internally computes its own ATR)
        compute_adx(df, 14)

        # BB
        df = compute_bollinger(df, self.bb_period)

        # BB bandwidth percentile over lookback
        if "BB_width" in df.columns:
            bw = df["BB_width"].dropna()
            if len(bw) >= self.bb_lookback:
                recent = bw.iloc[-self.bb_lookback:]
                self._bb_width_pct = (
                    (recent.iloc[-1] < recent).sum() / len(recent) * 100
                )
            else:
                self._bb_width_pct = 50.0  # not enough history → neutral

        self._calculated = True

    def classify(self) -> MarketState:
        """Return the current market state. Falls back gracefully on thin data."""
        if not self._calculated:
            self.calculate()

        df = self.df
        n = len(df)

        # Fallback: not enough bars for reliable classification
        if n < max(self.ma_long, 50):
            return MarketState(
                regime=MarketRegime.TRANSITIONAL,
                volatility=Volatility.NORMAL,
                adx=0, ma20=0, ma50=0, ma200=0, bb_width_pct=50,
            )

        last = df.iloc[-1]
        adx = float(last.get("ADX", 0))
        ma_s = float(last.get("MA_short", 0))
        ma_m = float(last.get("MA_mid", 0))
        ma_l = float(last.get("MA_long", 0))
        bb_pct = getattr(self, "_bb_width_pct", 50.0)

        # Regime
        trend_strength_ok = adx > self.adx_threshold
        ma_aligned_up = ma_s > ma_m > ma_l > 0
        ma_aligned_down = 0 < ma_s < ma_m < ma_l

        if trend_strength_ok and ma_aligned_up:
            regime = MarketRegime.TRENDING_UP
        elif trend_strength_ok and ma_aligned_down:
            regime = MarketRegime.TRENDING_DOWN
        elif adx < max(20, self.adx_threshold - 5):
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.TRANSITIONAL

        # Volatility
        if bb_pct >= self.vol_high_pct:
            vol = Volatility.HIGH
        elif bb_pct <= self.vol_low_pct:
            vol = Volatility.LOW
        else:
            vol = Volatility.NORMAL

        return MarketState(
            regime=regime, volatility=vol,
            adx=round(adx, 2),
            ma20=round(ma_s, 2), ma50=round(ma_m, 2), ma200=round(ma_l, 2),
            bb_width_pct=round(bb_pct, 1),
        )
