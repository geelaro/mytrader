"""Extreme Value Theory tail estimator.

Historical VaR at 99% from 5y daily data uses only ~12 observations.
At 99.9% it's down to 1–2.  Those estimates are statistically unstable.
EVT fits a parametric tail (Generalized Pareto Distribution) to the
extreme losses and extrapolates analytically, giving much more reliable
high-confidence VaR.

Method: Peaks-Over-Threshold (POT)
----------------------------------
1. Pick threshold u (e.g. 95th percentile of losses).
2. Collect exceedances: losses_i − u for losses_i > u.
3. Fit GPD with shape ξ and scale β to those exceedances.
4. Extrapolate::

    VaR_α = u + (β/ξ) × [((n/N_u) × (1−α))^(−ξ) − 1]    (ξ ≠ 0)
    VaR_α = u + β × ln(n/N_u × (1/(1−α)))                (ξ = 0)
    ES_α  = (VaR_α + β − ξ × u) / (1 − ξ)                (for ξ < 1)

Where ``n`` is the total sample size and ``N_u`` is the number of
exceedances.

Heavy-tailed assets (ξ > 0): expect EVT > historical for the same α.
Bounded distributions (ξ < 0): EVT < historical.

Convention
----------
- Input: returns Series (decimal form).  Losses = −returns.
- Output: positive numbers (loss magnitudes), decimal form.
- All losses below the threshold are dropped from the fit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# GPD fit
# ---------------------------------------------------------------------------


def fit_gpd(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
) -> Optional[dict]:
    """Fit GPD to loss exceedances above a threshold quantile.

    Returns dict::

        {
            "xi":         float,   # shape parameter ξ (heavy tails: ξ > 0)
            "beta":       float,   # scale parameter β (> 0)
            "threshold":  float,   # the loss threshold u
            "n_exceed":   int,     # number of observations above u
            "n_total":    int,     # full sample size after cleaning
        }

    Returns ``None`` if there are fewer than 20 exceedances (insufficient
    for a stable fit) or if the data is degenerate.
    """
    r = _clean(returns)
    if r.empty:
        return None
    losses = -r  # positive losses
    if not 0 < threshold_quantile < 1:
        raise ValueError(f"threshold_quantile must be in (0,1), got {threshold_quantile}")
    u = losses.quantile(threshold_quantile)
    exceedances = losses[losses > u] - u
    if len(exceedances) < 20:
        return None

    try:
        # scipy parameterisation: shape=ξ, loc, scale=β.
        # We fix loc=0 since we already subtracted the threshold.
        xi, _, beta = stats.genpareto.fit(exceedances.values, floc=0)
    except Exception:
        return None

    if beta <= 0 or not np.isfinite(xi) or not np.isfinite(beta):
        return None

    return {
        "xi": float(xi),
        "beta": float(beta),
        "threshold": float(u),
        "n_exceed": int(len(exceedances)),
        "n_total": int(len(losses)),
    }


# ---------------------------------------------------------------------------
# EVT VaR / ES
# ---------------------------------------------------------------------------


def evt_var(
    returns: pd.Series,
    confidence: float = 0.95,
    threshold_quantile: float = 0.95,
    fit: Optional[dict] = None,
) -> float:
    """Tail-extrapolated VaR using POT/GPD.

    Pass ``fit`` to reuse a previous :func:`fit_gpd` result; otherwise
    fits internally.  Returns 0.0 if the fit is unavailable or the
    requested quantile is *below* the threshold quantile (handled by the
    main historical estimator instead).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    if confidence <= threshold_quantile:
        # Below threshold → EVT extrapolation doesn't apply, return 0.
        return 0.0
    if fit is None:
        fit = fit_gpd(returns, threshold_quantile)
    if fit is None:
        return 0.0

    u = fit["threshold"]
    xi = fit["xi"]
    beta = fit["beta"]
    n = fit["n_total"]
    nu = fit["n_exceed"]

    # P(loss > x) = (Nu/n) × [1 + ξ(x-u)/β]^(-1/ξ)   for x > u
    # Inverting for the (1-α) tail:
    #   1 - α = (Nu/n) × [1 + ξ(VaR-u)/β]^(-1/ξ)
    # → VaR = u + (β/ξ) × [((n/Nu)(1-α))^(-ξ) - 1]
    tail = (n / nu) * (1 - confidence)
    if tail <= 0:
        return 0.0
    if abs(xi) < 1e-8:
        var = u + beta * np.log(1 / tail)
    else:
        var = u + (beta / xi) * (tail ** (-xi) - 1)
    return float(max(var, 0.0))


def evt_es(
    returns: pd.Series,
    confidence: float = 0.95,
    threshold_quantile: float = 0.95,
    fit: Optional[dict] = None,
) -> float:
    """Expected Shortfall (CVaR) extrapolated via GPD.

    ES_α = (VaR_α + β − ξ×u) / (1 − ξ)   when ξ < 1; otherwise the
    expected shortfall is undefined (infinite tail mean) and we return
    +inf.  Returns 0.0 if VaR itself is unavailable.
    """
    if fit is None:
        fit = fit_gpd(returns, threshold_quantile)
    if fit is None:
        return 0.0
    var = evt_var(returns, confidence, threshold_quantile, fit=fit)
    if var == 0.0:
        return 0.0
    xi = fit["xi"]
    beta = fit["beta"]
    u = fit["threshold"]
    if xi >= 1:
        return float("inf")
    return float((var + beta - xi * u) / (1 - xi))


# ---------------------------------------------------------------------------
# Summary — direct comparison against historical
# ---------------------------------------------------------------------------


def evt_summary(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
    confidences: Optional[tuple] = None,
) -> dict:
    """Side-by-side EVT vs historical at multiple confidences.

    Default confidences: 95%, 99%, 99.5%, 99.9%.  At 95% the EVT
    estimate may be missing (it's at/below the threshold); the historical
    figure remains available.  At 99.9% historical is rarely meaningful
    while EVT extrapolates analytically.

    Returns::

        {
            "fit":  {xi, beta, threshold, n_exceed, n_total}  or  None,
            "comparison": [
                {"confidence": "99%", "historical": float, "evt": float,
                 "evt_es": float},
                ...
            ],
            "warning": str | None
        }
    """
    from analysis.var import historical_var, conditional_var
    if confidences is None:
        confidences = (0.95, 0.99, 0.995, 0.999)
    r = _clean(returns)
    fit = fit_gpd(r, threshold_quantile)
    rows = []
    for c in confidences:
        rows.append({
            "confidence": f"{c * 100:g}%",
            "historical": historical_var(r, c),
            "historical_es": conditional_var(r, c),
            "evt": evt_var(r, c, threshold_quantile, fit=fit) if fit else 0.0,
            "evt_es": evt_es(r, c, threshold_quantile, fit=fit) if fit else 0.0,
        })

    warning = None
    if fit is None:
        warning = (
            "EVT 拟合失败 — 阈值之上观测样本不足 20 个, "
            "或数据本身退化. 仅展示历史 VaR."
        )
    elif fit["xi"] > 0.5:
        warning = (
            f"GPD ξ = {fit['xi']:.2f} (> 0.5): 极重尾, "
            "ES 估计极敏感. 99%+ VaR 仍可信, 但需要结合压力测试."
        )

    return {
        "fit": fit,
        "comparison": rows,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r
