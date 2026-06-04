"""VaR backtest — does the model actually hit its target rate?

Why
---
A 95% 1-day VaR claims "loss exceeds this number ≤5% of days".  Without
testing, you don't know if your model holds up.  This module runs a
rolling backtest and applies the standard regulatory coverage tests:

1. **Kupiec POF** (Proportion Of Failures, 1995)
   - H0: observed violation rate = (1 − confidence)
   - LR_POF ~ χ²(1) under H0
   - Rejects = model badly miscalibrated (too many or too few hits)

2. **Christoffersen independence** (1998)
   - H0: violations are serially independent (not clustered)
   - LR_ind ~ χ²(1)
   - Rejects = vol-clustering not captured (typical for non-conditional
     models on real data — exactly the case forward_var aims to fix)

3. **Conditional coverage** = LR_POF + LR_ind ~ χ²(2)
   - Joint test: right rate AND independent.  This is the headline number.

Methods supported
-----------------
- ``historical``   — rolling window historical VaR (the lagging benchmark)
- ``ewma``         — forward_var with method='ewma'
- ``gjr``          — forward_var with method='gjr' (slow: refits every
  ``refit_every`` steps, reuses params between refits)

Speed note
----------
EWMA is ~free per step.  GJR-GARCH MLE takes ~0.5s per fit on 1000 obs,
so for a 5-year backtest with refit_every=1 you'd wait many minutes.
Default ``refit_every=20`` (~monthly refit) trades fidelity for speed —
the volatility dynamics inside the 20-day window are still updated via
the closed-form GARCH recursion using fixed parameters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from analysis.garch import (
    RISKMETRICS_LAMBDA,
    ewma_volatility,
    fit_gjr_garch,
    forecast_volatility,
)
from analysis.var import historical_var


def coverage_backtest(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "ewma",
    window: int = 250,
    refit_every: int = 20,
) -> dict:
    """Rolling 1-day VaR backtest + Kupiec / Christoffersen / CC tests.

    Parameters
    ----------
    returns : pd.Series
        Decimal daily returns.  Index doesn't matter (only ordering).
    confidence : float in (0, 1)
        VaR confidence (0.95 → tests 5% tail).
    method : str
        One of ``"historical"``, ``"ewma"``, ``"gjr"``.
    window : int
        Training window size used at each step.  Must be ≥ ~60.
    refit_every : int
        (gjr only) Refit MLE every N steps; between refits use the
        recursion with frozen params.

    Returns
    -------
    dict — see module docstring.  ``violation_series`` is a 0/1
    pd.Series aligned to the *out-of-sample* return tail.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if method not in {"historical", "ewma", "gjr"}:
        raise ValueError(
            f"method must be one of 'historical'/'ewma'/'gjr', got {method!r}")
    if window < 60:
        raise ValueError(f"window too small ({window}); use ≥ 60")

    r = _clean(returns)
    if len(r) <= window:
        return _empty_result(confidence, method,
                             reason=f"need > {window} obs, got {len(r)}")

    r_arr = r.values
    n_total = len(r_arr)
    n_oos = n_total - window
    z = stats.norm.ppf(1 - confidence)

    var_forecasts = np.empty(n_oos)
    # ── GJR shared model state (refit every N steps) ────────────
    gjr_model: Optional[dict] = None
    last_refit_at = -refit_every - 1

    for i in range(n_oos):
        t = window + i  # absolute index into r_arr
        train = r_arr[:t]

        if method == "historical":
            # Empirical (1-c) percentile of the in-sample window
            q = float(np.quantile(train, 1 - confidence))
            var_forecasts[i] = max(-q, 0.0)

        elif method == "ewma":
            sigma_series = ewma_volatility(pd.Series(train))
            last_sigma = float(sigma_series.iloc[-1])
            last_r = float(train[-1])
            var_next = (RISKMETRICS_LAMBDA * last_sigma ** 2
                        + (1 - RISKMETRICS_LAMBDA) * last_r ** 2)
            sigma_forecast = np.sqrt(max(var_next, 1e-12))
            mu = float(np.mean(train))
            var_forecasts[i] = max(-(mu + z * sigma_forecast), 0.0)

        elif method == "gjr":
            # Refit only every refit_every steps
            if gjr_model is None or (i - last_refit_at) >= refit_every:
                gjr_model = fit_gjr_garch(pd.Series(train))
                last_refit_at = i

            # If MLE failed to converge, fall back to EWMA for this step
            if not gjr_model.get("fitted"):
                sigma_series = ewma_volatility(pd.Series(train))
                last_sigma = float(sigma_series.iloc[-1])
                last_r = float(train[-1])
                var_next = (RISKMETRICS_LAMBDA * last_sigma ** 2
                            + (1 - RISKMETRICS_LAMBDA) * last_r ** 2)
                sigma_forecast = np.sqrt(max(var_next, 1e-12))
            else:
                # Iterate σ_t forward to current t using frozen params
                sigma_forecast = _gjr_iterate_forward(
                    gjr_model, train, since_idx=last_refit_at + window,
                )
            mu = float(np.mean(train))
            var_forecasts[i] = max(-(mu + z * sigma_forecast), 0.0)

    # ── Violations ─────────────────────────────────────────────────
    realised = r_arr[window:]
    violations = (realised < -var_forecasts).astype(int)
    n_viol = int(violations.sum())
    observed_rate = n_viol / n_oos
    expected_rate = 1 - confidence

    # ── Tests ──────────────────────────────────────────────────────
    kupiec = _kupiec_pof(n_viol, n_oos, expected_rate)
    indep = _christoffersen_independence(violations)
    # Conditional coverage = sum of LRs ~ χ²(2)
    cc_lr = kupiec["lr"] + indep["lr"]
    cc_p = 1 - stats.chi2.cdf(cc_lr, df=2) if np.isfinite(cc_lr) else np.nan
    conditional = {
        "lr": cc_lr,
        "p_value": float(cc_p) if np.isfinite(cc_p) else float("nan"),
        "reject_at_5pct": bool(np.isfinite(cc_p) and cc_p < 0.05),
    }

    # Re-attach pandas index to the violation series for plotting
    oos_index = r.index[window:]
    viol_series = pd.Series(violations, index=oos_index, name="violation")
    var_series = pd.Series(var_forecasts, index=oos_index, name="var")

    return {
        "confidence": confidence,
        "method": method,
        "n_oos": int(n_oos),
        "n_violations": n_viol,
        "observed_rate": float(observed_rate),
        "expected_rate": float(expected_rate),
        "kupiec": kupiec,
        "christoffersen": indep,
        "conditional_coverage": conditional,
        "violation_series": viol_series,
        "var_series": var_series,
        "window": window,
    }


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


def _kupiec_pof(n_viol: int, n_total: int, expected_rate: float) -> dict:
    """Kupiec POF test: LR statistic + chi²(1) p-value."""
    if n_total <= 0:
        return _empty_test()
    p = expected_rate
    p_hat = n_viol / n_total
    # LR = -2 ln[(p/p̂)^N · ((1-p)/(1-p̂))^(T-N)]
    if n_viol == 0:
        # Edge: p_hat=0 → log(0) — degenerate.  LR = -2(T-N)·ln(1-p)
        lr = -2 * (n_total * np.log(1 - p))
    elif n_viol == n_total:
        lr = -2 * (n_total * np.log(p))
    else:
        lr = -2 * (n_viol * np.log(p / p_hat)
                   + (n_total - n_viol) * np.log((1 - p) / (1 - p_hat)))
    lr = max(lr, 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "lr": float(lr),
        "p_value": float(p_value),
        "reject_at_5pct": bool(p_value < 0.05),
    }


def _christoffersen_independence(violations: np.ndarray) -> dict:
    """Test serial independence of violations using a 2-state Markov chain."""
    if len(violations) < 2:
        return _empty_test()

    v = violations.astype(int)
    # Transition counts n_ij = times we went from state i to state j
    n_00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n_01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n_10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n_11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    n0 = n_00 + n_01
    n1 = n_10 + n_11
    n = n0 + n1
    n_viol = n_01 + n_11

    # Degenerate: no violations at all → independence is uninformative
    if n_viol == 0 or n_viol == n:
        return {"lr": 0.0, "p_value": 1.0, "reject_at_5pct": False}

    pi_01 = n_01 / n0 if n0 > 0 else 0.0
    pi_11 = n_11 / n1 if n1 > 0 else 0.0
    pi = n_viol / n

    # Restricted likelihood: p(violation) independent of prior state
    # Unrestricted: π_01 ≠ π_11
    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    log_l_restricted = (
        (n_00 + n_10) * _safe_log(1 - pi)
        + (n_01 + n_11) * _safe_log(pi)
    )
    log_l_unrestricted = (
        n_00 * _safe_log(1 - pi_01) + n_01 * _safe_log(pi_01)
        + n_10 * _safe_log(1 - pi_11) + n_11 * _safe_log(pi_11)
    )
    lr = -2 * (log_l_restricted - log_l_unrestricted)
    lr = max(lr, 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "lr": float(lr),
        "p_value": float(p_value),
        "reject_at_5pct": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gjr_iterate_forward(model: dict, train: np.ndarray, since_idx: int) -> float:
    """Apply frozen GJR-GARCH params to advance σ from σ at refit time.

    The fitted model's ``sigma`` series is for the data at refit time.
    Between refits, we extend it forward using the recursion with the
    same ω/α/γ/β so the volatility prediction stays responsive to new
    shocks without paying the MLE cost each day.
    """
    omega, alpha, gamma, beta = (
        model["omega"], model["alpha"], model["gamma"], model["beta"]
    )
    sigma_series = model["sigma"]
    # Last σ from the refit
    last_var = float(sigma_series.iloc[-1]) ** 2
    # Advance through any returns that came in since the refit
    # (since_idx might be < window if just refitted; clip safely)
    n_train = len(train)
    n_advance = max(0, n_train - len(sigma_series))
    var = last_var
    for j in range(n_advance):
        r_prev = float(train[len(sigma_series) + j - 1]) if (
            len(sigma_series) + j > 0) else 0.0
        leverage = 1.0 if r_prev < 0 else 0.0
        var = (omega + alpha * r_prev ** 2
               + gamma * leverage * r_prev ** 2
               + beta * var)
    # Now one more step ahead from t-1 using last training return
    r_prev = float(train[-1])
    leverage = 1.0 if r_prev < 0 else 0.0
    var_next = (omega + alpha * r_prev ** 2
                + gamma * leverage * r_prev ** 2
                + beta * var)
    return float(np.sqrt(max(var_next, 1e-12)))


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def _empty_test() -> dict:
    return {"lr": 0.0, "p_value": float("nan"), "reject_at_5pct": False}


def _empty_result(confidence: float, method: str, reason: str = "") -> dict:
    return {
        "confidence": confidence,
        "method": method,
        "n_oos": 0,
        "n_violations": 0,
        "observed_rate": 0.0,
        "expected_rate": 1 - confidence,
        "kupiec": _empty_test(),
        "christoffersen": _empty_test(),
        "conditional_coverage": {"lr": 0.0, "p_value": float("nan"),
                                 "reject_at_5pct": False},
        "violation_series": pd.Series(dtype=int),
        "var_series": pd.Series(dtype=float),
        "window": 0,
        "reason": reason,
    }
