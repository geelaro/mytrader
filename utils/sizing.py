"""Canonical risk-budget position sizing — shared by backtest engine and live trading.

Both ``BacktestEngine._calc_risk_budget_qty`` and ``RiskController.calc_position_size``
MUST delegate to :func:`calc_risk_budget_qty` to keep backtest results transferable to live.
"""

from typing import Optional

import numpy as np


def calc_risk_budget_qty(
    capital: float,
    price: float,
    atr: float,
    risk_pct: float,
    stop_atr_mult: float = 2.0,
    vol_sensitivity: float = 0,
    min_vol_scalar: float = 1.0,
) -> int:
    """Compute raw risk-budget quantity *before* external caps / floor.

    Core formula::

        risk_dollar = capital × risk_pct
        stop_distance = atr × stop_atr_mult
        vol_scalar   = 1 / (1 + vol_ratio × vol_sensitivity)  (if vol_sensitivity > 0)
        qty = risk_dollar / stop_distance × vol_scalar

    Callers should apply their own floor / ceiling (e.g. available cash cap,
    position-concentration cap, minimum 1 share).

    Parameters
    ----------
    capital : float
        Available cash (or reference capital for concentration caps).
    price : float
        Current price per share.
    atr : float
        Average True Range; NaN/zero → 2% of price as fallback.
    risk_pct : float
        Fraction of capital at risk per trade (0.01 = 1%).
    stop_atr_mult : float
        Stop-loss distance in ATR multiples (default 2.0).
    vol_sensitivity : float
        Volatility scaling coefficient; 0 disables (backtest default).
    min_vol_scalar : float
        Floor for volatility scaler; only relevant when vol_sensitivity > 0.
    """
    if price <= 0:
        return 0
    if atr is None or np.isnan(atr) or atr <= 0:
        atr = price * 0.02

    risk_dollar = capital * risk_pct
    stop_distance = atr * stop_atr_mult
    if stop_distance <= 0:
        return 0

    if vol_sensitivity > 0 and atr > 0:
        vol_ratio = atr / price
        vol_scalar = 1.0 / (1.0 + vol_ratio * vol_sensitivity)
        vol_scalar = max(vol_scalar, min_vol_scalar)
    else:
        vol_scalar = 1.0

    return int(risk_dollar / stop_distance * vol_scalar)
