"""Risk parameters and runtime state — shared by LiveTrader and tests."""

from dataclasses import dataclass, fields


@dataclass
class RiskLimits:
    """Safety limits — checked before every order submission."""

    max_position_pct: float = 0.30
    max_total_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.05
    min_order_value: float = 500.0
    max_slippage_pct: float = 0.02
    max_consecutive_losses: int = 3
    max_daily_trades: int = 5
    base_risk_pct: float = 0.02
    vol_sensitivity: float = 5.0
    min_vol_scalar: float = 0.3
    max_total_drawdown_pct: float = 0.30

    # -- runtime state --
    _day_start_equity: float = 0.0
    _peak_equity: float = 0.0
    _date: str = ""
    _consecutive_losses: int = 0
    _daily_trade_count: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "RiskLimits":
        """Build from watchlist.toml [risk] section.  Missing keys
        inherit dataclass field defaults — no duplicated fallback values."""
        rc = config.get("risk", {})
        param_names = {f.name for f in fields(cls) if not f.name.startswith("_")}
        kwargs = {k: rc[k] for k in param_names if k in rc}
        return cls(**kwargs)
