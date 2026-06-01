"""Risk-adjusted return metrics beyond Sharpe.

Sharpe penalises upside volatility just as much as downside — for a risk
management tool that's the wrong signal.  This module adds:

- :func:`sortino_ratio` — only the *downside* deviation in the denominator.
  Two strategies with the same Sharpe but different downside profiles
  diverge sharply here.

- :func:`calmar_ratio` — CAGR / |MaxDD|.  "How much do I earn per unit of
  pain at the worst point."  Simple, robust, well-suited to discretionary
  comparison ("Is 1.5 acceptable for the strategies I'm running?").

- :func:`mar_ratio` — synonym of Calmar with explicit lookback.  Some
  shops use 3y CAGR / 3y MaxDD, others use full history; this lets the
  caller specify.

- :func:`omega_ratio` — gains-above-threshold / losses-below-threshold,
  in expectation.  Captures full distribution shape (skew, kurtosis)
  that Sharpe misses.  Threshold 0 ≈ "win/loss ratio in expectation".

- :func:`information_ratio` — alpha / tracking error vs a benchmark.
  Requires a benchmark return series.

- :func:`pain_index` — average drawdown depth over the full period.
  Lower is better.  Less spike-prone than MaxDD; smooths "long shallow"
  vs "short deep" episodes.

- :func:`pain_ratio` — annualised return / pain_index.  Calmar's gentler
  cousin (uses average DD instead of max).

Convention
----------
All functions accept ``returns: pd.Series`` of single-period returns
(daily by default).  Pass ``periods=252`` for daily, ``52`` for weekly.
Risk-free rate is in *period* units, not annualised (rf=0.0001 for
2.5% annual ÷ 252).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Default periods-per-year for annualisation.  Override per-call as needed.
PERIODS_DAILY = 252
PERIODS_WEEKLY = 52
PERIODS_MONTHLY = 12


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = PERIODS_DAILY,
    target: Optional[float] = None,
) -> float:
    """Sortino = (mean − target) × √periods / downside_deviation.

    ``target`` is the minimum acceptable return (MAR); defaults to ``rf``.
    Downside deviation = √(mean of squared deviations *below* target).
    Returns 0.0 if no returns fall below target.
    """
    if target is None:
        target = rf
    r = _clean(returns)
    if r.empty:
        return 0.0
    excess = r - target
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    # Population (ddof=0) by Sortino convention: scale by N over downside subset
    dd_std = float(np.sqrt((downside ** 2).mean()))
    if dd_std == 0:
        return 0.0
    return float(excess.mean() * np.sqrt(periods) / dd_std)


# ---------------------------------------------------------------------------
# Calmar / MAR
# ---------------------------------------------------------------------------


def _max_drawdown_pct(returns: pd.Series) -> float:
    """Return MaxDD as a *positive* number in percent (e.g. 25.0 for -25%)."""
    r = _clean(returns)
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1)
    return float(-dd.min() * 100)


def _annualised_return_pct(returns: pd.Series, periods: int) -> float:
    """Geometric CAGR in percent."""
    r = _clean(returns)
    if r.empty:
        return 0.0
    n = len(r)
    if n == 0:
        return 0.0
    cumulative = float((1 + r).prod())
    if cumulative <= 0:
        return -100.0
    years = n / periods
    if years <= 0:
        return 0.0
    return (cumulative ** (1 / years) - 1) * 100


def calmar_ratio(returns: pd.Series, periods: int = PERIODS_DAILY) -> float:
    """CAGR / MaxDD (both in percent, result dimensionless).

    Returns 0.0 if MaxDD is 0 (no downside) — could be infinite but 0 is
    safer for sort/aggregation than inf.
    """
    cagr = _annualised_return_pct(returns, periods)
    mdd = _max_drawdown_pct(returns)
    if mdd <= 0:
        return 0.0
    return cagr / mdd


def mar_ratio(
    returns: pd.Series,
    lookback_years: Optional[float] = None,
    periods: int = PERIODS_DAILY,
) -> float:
    """MAR = CAGR / MaxDD, optionally restricted to last ``lookback_years``.

    If ``lookback_years`` is None, identical to :func:`calmar_ratio`.
    """
    r = _clean(returns)
    if r.empty:
        return 0.0
    if lookback_years is not None and lookback_years > 0:
        cutoff = int(lookback_years * periods)
        r = r.iloc[-cutoff:] if cutoff < len(r) else r
    return calmar_ratio(r, periods=periods)


# ---------------------------------------------------------------------------
# Omega
# ---------------------------------------------------------------------------


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega = E[max(r − τ, 0)] / E[max(τ − r, 0)].

    >1 means expected gains above τ outweigh expected losses below.
    Captures distribution shape Sharpe ignores (tail skew, fat tails).
    Returns inf if there are no returns below the threshold.
    """
    r = _clean(returns)
    if r.empty:
        return 0.0
    gains = (r - threshold).clip(lower=0).sum()
    losses = (threshold - r).clip(lower=0).sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Information ratio
# ---------------------------------------------------------------------------


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: int = PERIODS_DAILY,
) -> float:
    """IR = mean(active) × √periods / std(active).

    ``active = returns − benchmark_returns`` aligned on index.  Returns
    0.0 if alignment yields no overlap.
    """
    r = _clean(returns)
    b = _clean(benchmark_returns)
    if r.empty or b.empty:
        return 0.0
    aligned = pd.concat({"r": r, "b": b}, axis=1).dropna()
    if aligned.empty:
        return 0.0
    active = aligned["r"] - aligned["b"]
    if active.std(ddof=1) == 0:
        return 0.0
    return float(active.mean() * np.sqrt(periods) / active.std(ddof=1))


# ---------------------------------------------------------------------------
# Pain index / Pain ratio
# ---------------------------------------------------------------------------


def pain_index(returns: pd.Series) -> float:
    """Mean absolute drawdown over the full period (% units).

    A portfolio that spends 50% of its days at −10% drawdown has a
    pain index of 5.0 (in percent).  Less spike-prone than MaxDD.
    """
    r = _clean(returns)
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100  # in percent, ≤ 0
    return float(-dd.mean())  # flip to positive


def pain_ratio(returns: pd.Series, periods: int = PERIODS_DAILY) -> float:
    """CAGR / pain_index.  Calmar's smoother cousin (uses avg DD not max)."""
    cagr = _annualised_return_pct(returns, periods)
    pi = pain_index(returns)
    if pi <= 0:
        return 0.0
    return cagr / pi


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def risk_adjusted_summary(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf: float = 0.0,
    periods: int = PERIODS_DAILY,
) -> dict:
    """One-stop dict of all metrics — for dashboard rendering.

    Sharpe is included alongside Sortino/Calmar/etc. so callers can
    compare side-by-side without two API surfaces.
    """
    r = _clean(returns)
    out: dict = {
        "n_obs": int(len(r)),
        "annual_return_pct": _annualised_return_pct(r, periods),
        "max_drawdown_pct": _max_drawdown_pct(r),
        "sharpe": _sharpe_ratio(r, rf, periods),
        "sortino": sortino_ratio(r, rf, periods),
        "calmar": calmar_ratio(r, periods),
        "omega": omega_ratio(r, threshold=rf),
        "pain_index_pct": pain_index(r),
        "pain_ratio": pain_ratio(r, periods),
    }
    if benchmark_returns is not None:
        out["information_ratio"] = information_ratio(r, benchmark_returns, periods)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sharpe_ratio(returns: pd.Series, rf: float, periods: int) -> float:
    r = _clean(returns)
    if r.empty or r.std(ddof=1) == 0:
        return 0.0
    excess = r - rf
    return float(excess.mean() * np.sqrt(periods) / excess.std(ddof=1))


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r
