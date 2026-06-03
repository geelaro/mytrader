"""Marginal / Component VaR and risk-budget decomposition.

VaR tells you the portfolio-level tail loss.  This module decomposes that
loss to per-position contributions so the user can answer "trim which
position to reduce VaR most?".

Two views of the same math:

- :func:`marginal_var` — ∂VaR/∂w_i.  "If I add a tiny bit more of i,
  the portfolio VaR goes up by MVaR_i × Δw_i."  Sensitivity to weight.

- :func:`component_var` — w_i × MVaR_i.  Per-position dollar share of
  the portfolio VaR.  By Euler's theorem on the parametric VaR formula,
  Σᵢ component_var_i = total VaR exactly.  This is the workhorse metric.

Risk budgeting:

- :func:`risk_contribution_pct` — component VaR / total VaR × 100.
  "Position i carries X% of the portfolio's risk."

- :func:`risk_parity_weights` — solve for the weight vector that
  equalises every position's risk contribution.

Method
------
Parametric (Gaussian).  Assumes returns are jointly multivariate normal:
- σ_p = √(wᵀ Σ w)
- VaR_α = -(μ_p + z_α × σ_p), where z_α = Φ⁻¹(1−α)

Closed-form, fast, sufficient for typical equity portfolios.  Captures
correlation effects but understates fat tails — pair with historical
:func:`analysis.var.historical_var` for the headline number.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Internal: prepare aligned returns / cov / weights
# ---------------------------------------------------------------------------


def _prepare(prices: pd.DataFrame, weights: Mapping[str, float]):
    """Return (returns, w_series, mu_vec, cov, sigma_p, mu_p) or None.

    Filters to symbols present in BOTH prices and weights with weight > 0,
    normalises weights to sum to 1, computes covariance from full-period
    daily returns.  Requires at least 30 obs.
    """
    if prices is None or prices.empty or not weights:
        return None
    used = {s: w for s, w in weights.items() if s in prices.columns and w > 0}
    if not used:
        return None
    total = sum(used.values())
    if total <= 0:
        return None
    normed = {s: w / total for s, w in used.items()}
    syms = list(normed.keys())
    rets = prices[syms].pct_change().dropna()
    if rets.empty or len(rets) < 30:
        return None
    w = pd.Series(normed)[syms].values  # 1-D ndarray aligned with syms
    cov = rets.cov().values
    mu = rets.mean().values
    sigma_p = float(np.sqrt(w @ cov @ w))
    mu_p = float(w @ mu)
    return rets, w, mu, cov, sigma_p, mu_p, syms


# ---------------------------------------------------------------------------
# Marginal / Component VaR
# ---------------------------------------------------------------------------


def parametric_portfolio_var(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> float:
    """Total Gaussian VaR of the portfolio (positive loss, decimal).

    Match-point for component_var: Σ component_var ≈ this value.
    Returns 0.0 if data is insufficient or σ_p = 0.
    """
    pkg = _prepare(prices, weights)
    if pkg is None:
        return 0.0
    _, _, _, _, sigma_p, mu_p, _ = pkg
    if sigma_p == 0:
        return 0.0
    z = stats.norm.ppf(1 - confidence)
    return float(max(-(mu_p + z * sigma_p), 0.0))


def _marginal_var_from_pkg(pkg, confidence: float) -> pd.Series:
    """Compute marginal VaR from an already-prepared pkg tuple.

    Internal helper shared by marginal_var / component_var to avoid
    re-running _prepare twice when both are called in sequence.
    """
    _, w, mu, cov, sigma_p, _, syms = pkg
    if sigma_p == 0:
        return pd.Series(0.0, index=syms)
    z = stats.norm.ppf(1 - confidence)
    sigma_w = cov @ w
    mvar = -(mu + z * sigma_w / sigma_p)
    return pd.Series(mvar, index=syms)


def marginal_var(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> pd.Series:
    """∂VaR/∂w_i for each symbol (in decimal return units per unit weight).

    Returns a pd.Series indexed by symbol.  Empty if data insufficient.
    """
    pkg = _prepare(prices, weights)
    if pkg is None:
        return pd.Series(dtype=float)
    return _marginal_var_from_pkg(pkg, confidence)


def component_var(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> pd.Series:
    """Per-position contribution to portfolio VaR.

    By Euler's theorem, Σ component_var_i = parametric_portfolio_var.
    Returns can be negative if a position is a *hedge* (reduces portfolio
    risk on net) — interpret negative cvar as risk-reducing.
    """
    pkg = _prepare(prices, weights)
    if pkg is None:
        return pd.Series(dtype=float)
    _, w, _, _, _, _, syms = pkg
    mvar = _marginal_var_from_pkg(pkg, confidence)
    if mvar.empty:
        return mvar
    return pd.Series(w, index=syms) * mvar


def risk_contribution_pct(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> pd.Series:
    """Each position's % share of the portfolio VaR.

    Sums to ≈ 100% (within float).  An equally-weighted portfolio of low-
    correlation positions will show ~equal contributions; concentration
    in either weight or correlation pulls one bar up.
    """
    cvar = component_var(prices, weights, confidence)
    if cvar.empty:
        return cvar
    total = cvar.sum()
    if total == 0:
        return cvar * 0.0
    return cvar / total * 100


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


def risk_parity_weights(
    prices: pd.DataFrame,
    symbols: Optional[list] = None,
    max_iter: int = 500,
) -> pd.Series:
    """Solve for weights where every position contributes equal risk.

    Uses SLSQP to minimise Σ(rc_i − σ_p/n)² subject to Σw=1, w≥0.
    Falls back to inverse-volatility if optimisation fails.
    Returns weights summing to 1, indexed by symbol.
    """
    import warnings

    from scipy.optimize import minimize

    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    if symbols is not None:
        cols = [s for s in symbols if s in prices.columns]
    else:
        cols = list(prices.columns)
    if not cols:
        return pd.Series(dtype=float)

    rets = prices[cols].pct_change().dropna()
    if rets.empty or len(rets) < 30:
        return pd.Series(dtype=float)

    cov = rets.cov().values
    n = cov.shape[0]
    if n == 1:
        return pd.Series([1.0], index=cols)

    def objective(w: np.ndarray) -> float:
        sigma_p = float(np.sqrt(w @ cov @ w))
        if sigma_p == 0:
            return 1e9
        mrc = (cov @ w) / sigma_p
        rc = w * mrc
        target = sigma_p / n
        return float(((rc - target) ** 2).sum())

    w0 = np.ones(n) / n
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bounds = [(1e-6, 1.0)] * n
    try:
        # SLSQP emits clipping warnings during normal iteration — they're
        # intermediate, not failure signals.  Suppress to avoid noisy output.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module="scipy"
            )
            res = minimize(
                objective, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter, "ftol": 1e-10},
            )
        if res.success:
            w_opt = res.x / res.x.sum()  # re-normalise to guard against drift
            return pd.Series(w_opt, index=cols)
    except Exception:
        pass

    # Fallback: inverse-volatility (a coarse risk parity approximation)
    return inverse_volatility_weights(prices, cols)


def inverse_volatility_weights(
    prices: pd.DataFrame,
    symbols: Optional[list] = None,
) -> pd.Series:
    """Simple risk-parity approximation: w_i ∝ 1/σ_i.

    Equivalent to true risk parity only when correlations are equal.
    Fast and robust; use as a sanity check or fallback.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    cols = symbols if symbols is not None else list(prices.columns)
    cols = [s for s in cols if s in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    rets = prices[cols].pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    vol = rets.std()
    vol = vol[vol > 0]
    if vol.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / vol
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def risk_decomposition_summary(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> dict:
    """All-in-one decomposition for dashboard rendering.

    Returns::

        {
            "total_var_pct":      float,     # portfolio Gaussian VaR (% loss)
            "by_symbol":          DataFrame, # weight / mvar / cvar / pct
            "top_contributor":    str,
            "top_contributor_pct": float,
        }

    Empty/fallback dict if data is insufficient.
    """
    total = parametric_portfolio_var(prices, weights, confidence) * 100
    cvar = component_var(prices, weights, confidence) * 100
    if cvar.empty:
        return {
            "total_var_pct": 0.0,
            "by_symbol": pd.DataFrame(columns=["weight", "mvar_pct", "cvar_pct", "rc_pct"]),
            "top_contributor": None,
            "top_contributor_pct": 0.0,
        }

    mvar = marginal_var(prices, weights, confidence) * 100
    rc = risk_contribution_pct(prices, weights, confidence)

    # Re-derive normalised weights aligned with cvar's index
    used = {s: w for s, w in weights.items() if s in cvar.index and w > 0}
    total_w = sum(used.values())
    w_norm = pd.Series({s: w / total_w for s, w in used.items()})

    df = pd.DataFrame({
        "weight": w_norm,
        "mvar_pct": mvar,
        "cvar_pct": cvar,
        "rc_pct": rc,
    }).sort_values("cvar_pct", ascending=False)

    top = df.index[0] if not df.empty else None
    return {
        "total_var_pct": total,
        "by_symbol": df,
        "top_contributor": top,
        "top_contributor_pct": float(df.iloc[0]["rc_pct"]) if not df.empty else 0.0,
    }
