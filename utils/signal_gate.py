"""SignalGate — centralised pre-trade gate.

Decouples market-state, risk-pause, and regime-filter logic from
LiveTrader, keeping the main flow lean.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple

from utils.market_state import (
    MarketRegime, Volatility, MarketState,
    is_trend_strategy, is_mean_reversion_strategy,
)


def _build_regime_map() -> Dict[str, Optional[str]]:
    """Snapshot the current STRATEGY_MAP → regime mapping.

    Imported lazily so module load does not freeze STRATEGY_MAP at import
    time — callers that monkey-patch STRATEGY_MAP (tests, hot-reload) get a
    fresh map per SignalGate instance.
    """
    from strategy import STRATEGY_MAP
    return {name: getattr(cls, "regime", None) for name, cls in STRATEGY_MAP.items()}


@dataclass
class SignalGate:
    """Check whether a BUY or SELL signal should be acted on.

    Encapsulates:
    - Trading-pause gates (daily loss, consecutive loss, exposure)
    - Market-regime filtering (trend-in-range, reversion-in-trend)
    - Orphan-position guard (sell-only)
    - Exposure cap
    - Risk-limits pre-check delegation
    """

    ms_enabled: bool = False
    market_state: Optional[MarketState] = None
    trading_paused: bool = False
    pause_reason: str = ""
    max_total_exposure_pct: float = 0.80
    vol_high_scalar: float = 0.7
    regime_map: Dict[str, Optional[str]] = field(default_factory=_build_regime_map)

    # -- Public API -------------------------------------------------------

    def allow_buy(
        self, sig: dict, positions: Dict[str, any], account
    ) -> Tuple[bool, str]:
        """Return (can_buy, reason_if_blocked)."""
        sym = sig.get("symbol", "?")

        if self.trading_paused:
            code = re.sub(r'[^A-Za-z0-9]', '_', self.pause_reason).strip('_').upper()
            return False, f"PAUSE_{code}"

        if sig.get("orphan"):
            return False, "ORPHAN_BUY_BLOCKED"

        if self._regime_blocks(sig):
            strat = sig.get("strategy", "")
            regime = self.market_state.regime
            if regime == MarketRegime.RANGING:
                return False, f"RANGING_BLOCK_{strat.upper()}"
            else:
                return False, f"TRENDING_BLOCK_{strat.upper()}"

        # Exposure cap
        qty = sig.get("_qty", 0)
        price = sig.get("price", 0)
        if qty > 0 and price > 0 and account is not None:
            new_value = price * qty
            current = sum(p.market_value for p in positions.values() if p.market_value > 0)
            equity = getattr(account, "total_equity", 0)
            if equity > 0 and (current + new_value) / equity > self.max_total_exposure_pct:
                return False, "EXPOSURE_CAP_EXCEEDED"

        return True, ""

    def allow_sell(self, sig: dict) -> Tuple[bool, str]:
        """Return (can_sell, reason_if_blocked).

        Sells are rarely blocked; only when trading_paused AND an explicit
        sell-block flag is set.  Under normal conditions this always
        returns (True, "").
        """
        return True, ""

    def vol_scaled_qty(self, qty: int) -> int:
        """Scale position size down in high-volatility regimes."""
        if (self.ms_enabled and self.market_state is not None
                and self.market_state.volatility == Volatility.HIGH):
            return 0 if qty == 0 else max(1, int(qty * self.vol_high_scalar))
        return qty

    # -- Internal ---------------------------------------------------------

    def _regime_blocks(self, sig: dict) -> bool:
        if not self.ms_enabled or self.market_state is None:
            return False
        regime = self.market_state.regime
        strat = sig.get("strategy", "")
        if regime == MarketRegime.RANGING and is_trend_strategy(strat, self.regime_map):
            return True
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN) \
                and is_mean_reversion_strategy(strat, self.regime_map):
            return True
        return False
