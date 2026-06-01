"""Value-at-Risk and Expected Shortfall.

Three estimators, all returning a *positive* number (the loss):

- :func:`historical_var` — empirical percentile of past returns.
  Non-parametric.  Captures fat tails seen in actual data; needs ≥ ~250
  observations for the 95% percentile to be stable.

- :func:`parametric_var` — μ + z·σ assuming returns are normal.
  Closed-form, fast, but understates tail risk for fat-tailed assets.

- :func:`conditional_var` (Expected Shortfall / CVaR) — average loss
  *given* the loss exceeds the VaR threshold.  Captures tail severity
  that VaR misses (VaR tells you *where* the cliff is; ES tells you *how
  deep* it falls).

All three accept any pd.Series of single-period returns (daily, weekly).
Convert from prices via ``df["Close"].pct_change().dropna()``.

Portfolio aggregation: :func:`portfolio_returns` weights individual symbol
returns into a portfolio return series; pass the result back into the
single-series functions.

Convention
----------
All VaR/ES values are **positive losses**.  ``historical_var = 0.024`` means
a 2.4% loss.  Returns input should be in decimal form (0.01 = +1%).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical VaR: the (1−c)-th percentile of the return distribution.

    Returns the magnitude of the loss as a positive number.  Returns 0.0
    if the requested percentile is positive (no loss at that confidence).
    """
    _validate_confidence(confidence)
    r = _clean(returns)
    if r.empty:
        return 0.0
    q = r.quantile(1 - confidence)
    return float(max(-q, 0.0))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR: −(μ + z·σ) where z is the (1−c) normal quantile."""
    _validate_confidence(confidence)
    r = _clean(returns)
    if r.empty or r.std() == 0:
        return 0.0
    z = stats.norm.ppf(1 - confidence)
    var = -(r.mean() + z * r.std(ddof=1))
    return float(max(var, 0.0))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall: mean of returns at or below the VaR threshold.

    A.k.a. CVaR / Average VaR.  Always ≥ historical_var (in magnitude).
    Returns 0.0 if no returns fall in the tail.
    """
    _validate_confidence(confidence)
    r = _clean(returns)
    if r.empty:
        return 0.0
    threshold = r.quantile(1 - confidence)
    tail = r[r <= threshold]
    if tail.empty:
        return 0.0
    return float(max(-tail.mean(), 0.0))


def portfolio_returns(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Convert a price panel + weights into a portfolio return series.

    Parameters
    ----------
    prices : DataFrame indexed by date, columns = symbols, values = price.
    weights : dict {symbol: weight}.  Weights are normalised internally
        (must sum to a positive number).  Symbols missing from prices are
        ignored; symbols missing from weights contribute zero.

    Returns
    -------
    pd.Series of portfolio returns aligned to the price index, NaNs dropped.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    used = {s: w for s, w in weights.items() if s in prices.columns and w > 0}
    total = sum(used.values())
    if total <= 0:
        return pd.Series(dtype=float)
    normed = {s: w / total for s, w in used.items()}

    rets = prices[list(normed.keys())].pct_change().dropna(how="all")
    if rets.empty:
        return pd.Series(dtype=float)
    w_vec = pd.Series(normed)
    pf = (rets * w_vec).sum(axis=1)
    return pf.dropna()


def var_summary(
    returns: pd.Series,
    confidences: Optional[tuple] = None,
) -> dict:
    """One-stop summary: dict[confidence → {historical, parametric, cvar}].

    Convenient for dashboard rendering.  Default confidences: 95% and 99%.
    Also reports ``n_obs`` and ``mean`` / ``std`` of the input series for
    sanity checks.
    """
    if confidences is None:
        confidences = (0.95, 0.99)
    r = _clean(returns)
    out: dict = {
        "n_obs": int(len(r)),
        "mean": float(r.mean()) if not r.empty else 0.0,
        "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
    }
    for c in confidences:
        key = f"{int(c * 100)}%"
        out[key] = {
            "historical": historical_var(r, c),
            "parametric": parametric_var(r, c),
            "cvar": conditional_var(r, c),
        }
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaN/inf, ensure float dtype.  Robust to caller passing dirty data."""
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _validate_confidence(c: float):
    if not 0 < c < 1:
        raise ValueError(f"confidence must be in (0, 1), got {c}")
