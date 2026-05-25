"""StrategyEnsemble — market-regime-aware weighted voting across strategies.

Weights adapt to the current market state (classified from a proxy like SPY):

- TRENDING_UP:   trend × 0.7  +  mean_reversion × 0.3
- TRENDING_DOWN: trend × 0.7  +  mean_reversion × 0.3
- RANGING:       mean_reversion × 0.6  +  trend × 0.4
- HIGH_VOL:      trend × 0.6  +  exit signals priority
- TRANSITIONAL:  equal weight
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy, StrategyParams
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility


@dataclass(frozen=True)
class EnsembleParams(StrategyParams):
    long_bias_threshold: float = 0.3   # score ≥ +threshold → long
    short_bias_threshold: float = 0.3  # score ≤ -threshold → short
    min_agreement: int = 1             # minimum strategies agreeing for entry

    def validate(self):
        if self.long_bias_threshold < 0:
            raise ValueError("long_bias_threshold must be >= 0")
        if self.short_bias_threshold < 0:
            raise ValueError("short_bias_threshold must be >= 0")


# Default weights per regime
_DEFAULT_WEIGHTS = {
    MarketRegime.TRENDING_UP:   {"trend": 0.7, "mean_reversion": 0.3},
    MarketRegime.TRENDING_DOWN: {"trend": 0.7, "mean_reversion": 0.3},
    MarketRegime.RANGING:       {"trend": 0.4, "mean_reversion": 0.6},
    MarketRegime.TRANSITIONAL:  {"trend": 0.5, "mean_reversion": 0.5},
}


class StrategyEnsemble(BaseStrategy):
    """Combine multiple strategies via regime-weighted voting.

    Parameters
    ----------
    members : list of (strategy, regime_label)
        e.g. ``[(TurtleTrading(), "trend"), (BollingerMeanReversion(), "mean_reversion")]``
    proxy_df : pd.DataFrame
        OHLCV of the market proxy (e.g. SPY) for regime classification.
    weights : dict, optional
        Regime → {regime_label: weight} mapping.  Falls back to defaults.
    """

    regime = None  # mixed — all types
    long_only = False  # ensemble can go both ways

    params: EnsembleParams

    def __init__(
        self,
        members: List[Tuple[BaseStrategy, str]],
        proxy_df: pd.DataFrame,
        weights: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(EnsembleParams(**kwargs))
        self._members = members
        self._classifier = MarketStateClassifier(proxy_df)
        self._classifier.calculate()
        self._weights = weights or _DEFAULT_WEIGHTS
        self._regime_map = {s.regime: r for s, r in members}

    @property
    def min_bars(self) -> int:
        return max(s.min_bars for s, _ in self._members)

    # -- indicators ---------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame, df_weekly=None) -> pd.DataFrame:
        """Run each member and combine signals via weighted vote."""
        state = self._classifier.classify()
        weights = self._weights.get(state.regime, {"trend": 0.5, "mean_reversion": 0.5})

        # High-vol: increase trend weight slightly, favour exit
        vol_bonus = 0.0
        if state.volatility == Volatility.HIGH:
            vol_bonus = 0.1

        # Collect each member's signal series
        member_signals: List[pd.Series] = []
        member_regimes: List[str] = []
        for strat, regime_label in self._members:
            try:
                df_sig = strat.calculate_indicators(df, df_weekly=df_weekly)
            except TypeError:
                df_sig = strat.calculate_indicators(df)
            member_signals.append(df_sig["Signal"])
            member_regimes.append(regime_label)

        # Weighted score
        score = pd.Series(0.0, index=df.index[:len(member_signals[0])])
        for sig_series, regime_label in zip(member_signals, member_regimes):
            w = weights.get(regime_label, 0.33) + (vol_bonus if regime_label == "trend" else 0)
            score = score + sig_series.astype(float) * w

        # Consensus count (how many members agree on direction)
        long_votes = sum((s > 0).astype(int) for s in member_signals)
        short_votes = sum((s < 0).astype(int) for s in member_signals)

        # Build output DataFrame on the primary daily index
        result = df.copy()
        result["Signal"] = 0
        result.loc[long_votes >= self.params.min_agreement, "Signal"] = (
            (score >= self.params.long_bias_threshold).astype(int)
        )
        result.loc[short_votes >= self.params.min_agreement, "Signal"] = (
            (-1 * (score <= -self.params.short_bias_threshold)).astype(int)
        )
        # Resolve conflict: if both long and short signals on same bar, pick
        # the one with higher absolute score
        conflict = (result["Signal"] > 0) & (result["Signal"] < 0)  # never true, guard
        if conflict.any():
            result.loc[conflict, "Signal"] = np.sign(score[conflict]).astype(int)

        # Carry forward ATR from first member that has it
        for strat, _ in self._members:
            try:
                df_sig = strat.calculate_indicators(df, df_weekly=df_weekly)
            except TypeError:
                df_sig = strat.calculate_indicators(df)
            if "ATR" in df_sig.columns:
                result["ATR"] = df_sig["ATR"]
                break
        else:
            from strategy.base import compute_atr
            result["ATR"] = compute_atr(result, 14)

        return result

    # -- sizing -------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr, 0.02, 2.0, 0.95)
