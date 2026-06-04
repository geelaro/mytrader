"""Forward-looking conditional volatility — EWMA and GJR-GARCH(1,1).

Why
---
The historical / parametric VaR in :mod:`analysis.var` uses the *sample*
standard deviation of past returns.  That number lags reality:

- Before a vol regime change, sample σ understates risk for weeks.
- Right after a crisis, sample σ stays elevated long after the storm.

Conditional-volatility models give a forecast of *tomorrow's* σ based on
the most recent shock, so they catch turning points earlier.

Two estimators
--------------
1. **EWMA** (RiskMetrics 1996, λ=0.94 daily standard)
       σ²_t = λ · σ²_{t−1} + (1 − λ) · r²_{t−1}
   Tiny model, no fitting, robust.  Used by JPM RiskMetrics and most
   bank VaR systems.  Drawback: σ² has no mean-reversion — a one-time
   crisis bleeds out at fixed half-life.

2. **GJR-GARCH(1,1)** (Glosten–Jagannathan–Runkle 1993)
       σ²_t = ω + α · r²_{t−1} + γ · I_{t−1} · r²_{t−1} + β · σ²_{t−1}
       where  I_{t−1} = 1 if r_{t−1} < 0 else 0
   Captures the **leverage effect**: negative shocks raise vol more than
   positive shocks of the same magnitude (well-documented in equity).
   γ > 0 quantifies the asymmetry.  Mean-reverts to ω/(1−α−γ/2−β).

API
---
- :func:`ewma_volatility` — full σ series (RiskMetrics).
- :func:`fit_gjr_garch` — MLE fit, returns params + in-sample σ series.
- :func:`forecast_volatility` — h-step-ahead vol from a fitted/EWMA state.
- :func:`forward_var` — combine forecasted σ with normal quantile → VaR.
- :func:`forward_var_summary` — both methods × multiple confidences for
  dashboard rendering.

Convention: returns input is decimal (0.01 = +1%).  Output VaR is a
*positive* loss number, same as :mod:`analysis.var`.

Caveats
-------
- Forecast assumes returns are conditionally Gaussian.  Heavy-tailed
  assets understate tails — combine with :mod:`analysis.evt` for stress
  scenarios.
- GJR-GARCH MLE needs ≥ ~250 observations to be stable; otherwise this
  module falls back to EWMA (and reports ``method='ewma_fallback'``).
- Stationarity requires ``α + γ/2 + β < 1``; the optimiser is constrained.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


RISKMETRICS_LAMBDA = 0.94  # RiskMetrics 1996 standard for daily data


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = RISKMETRICS_LAMBDA,
) -> pd.Series:
    """RiskMetrics EWMA conditional std time series.

    σ²_t = λ · σ²_{t−1} + (1 − λ) · r²_{t−1}

    Initialises σ²_0 to the sample variance of the first 30 returns
    (or the full series if shorter).  Returns σ_t (not σ²_t).
    """
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)

    r_arr = r.values
    n = len(r_arr)
    var = np.empty(n)
    # Seed: variance over the initial warm-up window
    warmup = min(30, n)
    var[0] = float(np.var(r_arr[:warmup], ddof=1)) if warmup > 1 else float(r_arr[0] ** 2)
    for t in range(1, n):
        var[t] = lambda_ * var[t - 1] + (1 - lambda_) * r_arr[t - 1] ** 2

    sigma = np.sqrt(np.maximum(var, 1e-12))
    return pd.Series(sigma, index=r.index)


# ---------------------------------------------------------------------------
# GJR-GARCH(1,1)
# ---------------------------------------------------------------------------


_GJR_MIN_OBS = 250  # below this, MLE is unstable → fall back to EWMA


def fit_gjr_garch(
    returns: pd.Series,
    init_params: Optional[tuple] = None,
) -> dict:
    """Fit GJR-GARCH(1,1) by maximum likelihood.

    Returns
    -------
    dict::

        {
            "fitted":          bool,      # False if not enough data
            "method":          "gjr" | "ewma_fallback",
            "omega":           float,
            "alpha":           float,
            "gamma":           float,
            "beta":            float,
            "persistence":     float,     # α + γ/2 + β
            "long_run_var":    float,     # ω / (1 − persistence)
            "sigma":           pd.Series, # in-sample σ_t time series
            "loglik":          float,
            "n_obs":           int,
        }

    On failure (insufficient data or optimisation didn't converge) falls
    back to EWMA σ and sets ``method="ewma_fallback"``.
    """
    r = _clean(returns)
    n = len(r)

    def _ewma_fallback():
        sigma = ewma_volatility(r) if not r.empty else pd.Series(dtype=float)
        return {
            "fitted": False,
            "method": "ewma_fallback",
            "omega": float("nan"),
            "alpha": float("nan"),
            "gamma": float("nan"),
            "beta": float("nan"),
            "persistence": float("nan"),
            "long_run_var": float("nan"),
            "sigma": sigma,
            "loglik": float("nan"),
            "n_obs": n,
        }

    if n < _GJR_MIN_OBS:
        return _ewma_fallback()

    r_arr = r.values.astype(float)
    r2 = r_arr ** 2
    neg = (r_arr < 0).astype(float)

    sample_var = float(np.var(r_arr, ddof=1))
    if sample_var <= 0:
        return _ewma_fallback()

    # Initial guess: typical equity GARCH — α=0.05, γ=0.10, β=0.80
    # ω chosen so unconditional variance ≈ sample_var.
    init_persistence = 0.05 + 0.10 / 2 + 0.80  # = 0.90
    x0 = init_params or (sample_var * (1 - init_persistence), 0.05, 0.10, 0.80)

    def _neg_loglik(params):
        omega, alpha, gamma, beta = params
        # Bounds enforced via bounds= argument; reject infeasible numerics here.
        if omega <= 1e-15:
            return 1e12
        persistence_p = alpha + gamma / 2 + beta
        if persistence_p >= 0.999:
            # Strong penalty toward stationarity instead of hard constraint
            return 1e10 + 1e6 * (persistence_p - 0.999)
        var = np.empty(n)
        var[0] = sample_var
        for t in range(1, n):
            var[t] = (omega
                      + alpha * r2[t - 1]
                      + gamma * neg[t - 1] * r2[t - 1]
                      + beta * var[t - 1])
        var = np.maximum(var, 1e-12)
        # Gaussian log-likelihood (drop constant 0.5*log(2π))
        ll = -0.5 * np.sum(np.log(var) + r2 / var)
        return -ll  # minimise the negative

    # L-BFGS-B with just bounds — handles flat likelihood surfaces (IID
    # returns) gracefully where SLSQP can fail.  Stationarity checked
    # post-hoc.
    bounds = [
        (1e-10, sample_var * 10),  # ω
        (0.0, 0.4),                 # α
        (0.0, 0.5),                 # γ
        (0.0, 0.999),               # β
    ]
    try:
        result = optimize.minimize(
            _neg_loglik, x0=x0, method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )
    except Exception:
        return _ewma_fallback()

    # L-BFGS-B may return success=False but still give usable params for
    # flat regions; accept as long as the objective is finite and bounds hold.
    if not np.isfinite(result.fun) or result.fun > 1e9:
        return _ewma_fallback()

    omega, alpha, gamma, beta = result.x
    persistence = alpha + gamma / 2 + beta
    if persistence >= 1.0 or omega <= 0:
        return _ewma_fallback()

    # Recompute σ series with the fitted parameters
    var = np.empty(n)
    var[0] = sample_var
    for t in range(1, n):
        var[t] = (omega
                  + alpha * r2[t - 1]
                  + gamma * neg[t - 1] * r2[t - 1]
                  + beta * var[t - 1])
    var = np.maximum(var, 1e-12)
    sigma = pd.Series(np.sqrt(var), index=r.index)

    return {
        "fitted": True,
        "method": "gjr",
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "persistence": float(persistence),
        "long_run_var": float(omega / (1 - persistence)),
        "sigma": sigma,
        "loglik": float(-result.fun),
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------


def forecast_volatility(
    model: dict,
    last_return: Optional[float] = None,
    horizon: int = 1,
) -> float:
    """h-step-ahead σ forecast from a fitted GARCH or EWMA model.

    Parameters
    ----------
    model : dict from :func:`fit_gjr_garch` (or an EWMA fallback).
    last_return : float, optional
        The most recent observed return.  If omitted, the model's last
        in-sample σ_t is assumed to already incorporate it (i.e., the
        function returns σ_T for h=1, the long-run mean for h→∞).
    horizon : int
        How many steps ahead.  h=1 → tomorrow.

    Notes
    -----
    For GJR-GARCH, multi-step variance forecasts iterate using the
    *unconditional* leverage indicator (we assume future shocks are
    50/50 positive/negative — so γ contributes γ/2 on average).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")

    sigma_series = model.get("sigma", pd.Series(dtype=float))
    if sigma_series is None or sigma_series.empty:
        return 0.0
    last_sigma = float(sigma_series.iloc[-1])

    if not model.get("fitted"):
        # EWMA fallback: variance has no mean-reversion, σ persists.
        # If last_return given, advance one step then hold flat for h>1.
        if last_return is not None:
            var_next = (RISKMETRICS_LAMBDA * last_sigma ** 2
                        + (1 - RISKMETRICS_LAMBDA) * last_return ** 2)
            return float(np.sqrt(max(var_next, 1e-12)))
        return last_sigma

    omega = model["omega"]
    alpha = model["alpha"]
    gamma = model["gamma"]
    beta = model["beta"]
    persistence = alpha + gamma / 2 + beta
    long_run_var = model["long_run_var"]

    # h=1 conditional variance given last_return (or last_sigma)
    if last_return is not None:
        leverage = 1.0 if last_return < 0 else 0.0
        var_1 = (omega
                 + alpha * last_return ** 2
                 + gamma * leverage * last_return ** 2
                 + beta * last_sigma ** 2)
    else:
        var_1 = last_sigma ** 2

    if horizon == 1:
        return float(np.sqrt(max(var_1, 1e-12)))

    # h-step: var_h → long_run_var as h grows, geometric decay
    # var_h = long_run_var + persistence^(h-1) * (var_1 - long_run_var)
    var_h = long_run_var + (persistence ** (horizon - 1)) * (var_1 - long_run_var)
    return float(np.sqrt(max(var_h, 1e-12)))


# ---------------------------------------------------------------------------
# Forward VaR
# ---------------------------------------------------------------------------


def forward_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "gjr",
    horizon: int = 1,
) -> float:
    """Forecast h-day VaR using conditional volatility.

    VaR = −(μ + z · σ_forecast) where z is the (1−c) standard-normal quantile.

    Parameters
    ----------
    returns : pd.Series of decimal daily returns.
    confidence : 0.95 / 0.99 etc.
    method : "gjr" (fits GJR-GARCH, falls back to EWMA if unfittable)
             or "ewma" (always RiskMetrics EWMA).
    horizon : forecast horizon in days.  VaR scales by sqrt(h) under
              the IID-normal-shock assumption.

    Returns a positive loss number (e.g., 0.024 = 2.4% loss).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    r = _clean(returns)
    if r.empty or r.std() == 0:
        return 0.0

    if method == "ewma":
        sigma = ewma_volatility(r)
        model = {
            "fitted": False, "method": "ewma_fallback",
            "sigma": sigma,
        }
    elif method == "gjr":
        model = fit_gjr_garch(r)
    else:
        raise ValueError(f"unknown method '{method}'; expected 'gjr' or 'ewma'")

    last_return = float(r.iloc[-1])
    sigma_forecast = forecast_volatility(model, last_return=last_return, horizon=1)
    if horizon > 1:
        sigma_forecast = sigma_forecast * np.sqrt(horizon)

    z = stats.norm.ppf(1 - confidence)
    mu = float(r.mean())
    var = -(mu * horizon + z * sigma_forecast)
    return float(max(var, 0.0))


def forward_var_summary(
    returns: pd.Series,
    confidences: Optional[tuple] = None,
    horizon: int = 1,
) -> dict:
    """Both EWMA and GJR-GARCH forward-VaR at multiple confidence levels.

    Returns dict containing fitted-model diagnostics plus per-confidence
    forecasts, mirroring :func:`analysis.var.var_summary` for dashboard
    parallelism.
    """
    if confidences is None:
        confidences = (0.95, 0.99)
    r = _clean(returns)

    gjr_model = fit_gjr_garch(r)
    ewma_sigma = ewma_volatility(r)
    last_return = float(r.iloc[-1]) if not r.empty else 0.0

    sigma_gjr = forecast_volatility(gjr_model, last_return=last_return, horizon=1)
    sigma_ewma = (forecast_volatility(
        {"fitted": False, "sigma": ewma_sigma},
        last_return=last_return,
    ) if not ewma_sigma.empty else 0.0)

    if horizon > 1:
        sigma_gjr *= np.sqrt(horizon)
        sigma_ewma *= np.sqrt(horizon)

    mu = float(r.mean()) if not r.empty else 0.0

    out: dict = {
        "n_obs": int(len(r)),
        "horizon_days": horizon,
        "sigma_forecast": {
            "ewma": sigma_ewma,
            "gjr":  sigma_gjr,
        },
        "gjr_params": {
            "fitted":      gjr_model.get("fitted", False),
            "method":      gjr_model.get("method", "ewma_fallback"),
            "omega":       gjr_model.get("omega"),
            "alpha":       gjr_model.get("alpha"),
            "gamma":       gjr_model.get("gamma"),
            "beta":        gjr_model.get("beta"),
            "persistence": gjr_model.get("persistence"),
        },
    }
    for c in confidences:
        key = f"{int(c * 100)}%"
        z = stats.norm.ppf(1 - c)
        out[key] = {
            "ewma": float(max(-(mu * horizon + z * sigma_ewma), 0.0)),
            "gjr":  float(max(-(mu * horizon + z * sigma_gjr), 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    return r.replace([np.inf, -np.inf], np.nan).dropna()
