"""Portfolio concentration metrics.

Tells you *how concentrated* a portfolio is — beyond just listing weights.
Used to spot "I thought I was diversified but I'm really long one factor".

Metrics
-------
- :func:`hhi` — Herfindahl-Hirschman Index, the standard concentration
  measure.  Scaled 0–10000.  Conventional bands (US DOJ for industries,
  reused widely in finance):

    HHI < 1500   : unconcentrated
    1500–2500    : moderately concentrated
    > 2500       : highly concentrated

  An equal-weight portfolio of N names has HHI = 10000 / N
  (so 4 equally-weighted ≈ 2500, the "highly concentrated" floor).

- :func:`effective_n` — 1 / Σw_i², the "equivalent number of equally-
  weighted holdings".  10 names with one taking 50% → effective_n ≈ 3.7,
  not 10.

- :func:`top_n_weight` — share held by the largest N.

- :func:`sector_exposure` — aggregated weight by sector.

- :func:`sector_hhi` — HHI computed on sector weights instead of symbol
  weights.  This is the metric that flags "11 names but all in Tech".

- :func:`correlation_hhi` — concentration adjusted for pairwise
  correlation.  Two perfectly correlated 50/50 positions have effective
  HHI of one 100% position (not 5000).

Conventions
-----------
Weights are in decimal form (sum to 1.0 if normalised).  All functions
normalise internally so callers can pass un-normalised inputs.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Concentration bands (US DOJ industry concentration convention)
# ---------------------------------------------------------------------------

HHI_UNCONCENTRATED = 1500
HHI_MODERATE = 2500


def hhi_label(hhi_value: float) -> str:
    """Human-readable band for an HHI value."""
    if hhi_value < HHI_UNCONCENTRATED:
        return "分散"
    if hhi_value < HHI_MODERATE:
        return "中等集中"
    return "高度集中"


# ---------------------------------------------------------------------------
# Symbol-level concentration
# ---------------------------------------------------------------------------


def hhi(weights) -> float:
    """Herfindahl-Hirschman Index of weights (0–10000)."""
    w = _normalised_weights(weights)
    if w.empty:
        return 0.0
    return float((w ** 2).sum() * 10000)


def effective_n(weights) -> float:
    """Equivalent number of equally-weighted holdings (= 1 / Σw²)."""
    w = _normalised_weights(weights)
    if w.empty:
        return 0.0
    ss = float((w ** 2).sum())
    return 1.0 / ss if ss > 0 else 0.0


def top_n_weight(weights, n: int = 3) -> float:
    """Sum of the top N weights (as a fraction in [0, 1])."""
    if n <= 0:
        return 0.0
    w = _normalised_weights(weights)
    if w.empty:
        return 0.0
    return float(w.nlargest(n).sum())


# ---------------------------------------------------------------------------
# Sector aggregation
# ---------------------------------------------------------------------------


def sector_exposure(
    weights,
    sector_map: Mapping[str, str],
    unknown_label: str = "Unknown",
) -> dict:
    """Aggregate weights by sector.

    Symbols missing from ``sector_map`` are bucketed into ``unknown_label``.
    Returns dict {sector: weight} sorted by weight descending.
    """
    w = _normalised_weights(weights)
    if w.empty:
        return {}
    by_sector: dict = {}
    for sym, weight in w.items():
        sector = sector_map.get(sym, unknown_label)
        by_sector[sector] = by_sector.get(sector, 0.0) + float(weight)
    # Sort by weight descending
    return dict(sorted(by_sector.items(), key=lambda kv: -kv[1]))


def sector_hhi(
    weights,
    sector_map: Mapping[str, str],
    unknown_label: str = "Unknown",
) -> float:
    """HHI on sector-level weights — the "all-in-one-factor" detector.

    A watchlist of 11 names all in Technology will have low symbol HHI
    (well-diversified by name) but sector_hhi near 10000.
    """
    exposure = sector_exposure(weights, sector_map, unknown_label)
    if not exposure:
        return 0.0
    return float(sum(v ** 2 for v in exposure.values()) * 10000)


# ---------------------------------------------------------------------------
# Correlation-adjusted concentration
# ---------------------------------------------------------------------------


def correlation_hhi(
    weights,
    correlation_matrix: Optional[pd.DataFrame] = None,
) -> float:
    """HHI adjusted for return correlation: wᵀ · ρ · w · 10000.

    Two 50/50 perfectly-correlated positions → corr_hhi = 10000 (same
    as one 100% position), even though plain HHI = 5000.

    If ``correlation_matrix`` is None, falls back to plain :func:`hhi`.
    Symbols missing from the matrix get correlation 1 with themselves
    (their plain weight²) and 0 with others.
    """
    w = _normalised_weights(weights)
    if w.empty:
        return 0.0
    if correlation_matrix is None or correlation_matrix.empty:
        return hhi(weights)

    # Align: only use symbols present in both
    syms = [s for s in w.index if s in correlation_matrix.columns and s in correlation_matrix.index]
    if not syms:
        return hhi(weights)

    w_arr = w.loc[syms].values
    rho = correlation_matrix.loc[syms, syms].values
    # Symmetric quadratic form; clip negative correlations don't reduce
    # below 0 mathematically here (full rho matrix), no clip needed.
    quadratic = float(w_arr @ rho @ w_arr)
    return max(0.0, quadratic * 10000)


# ---------------------------------------------------------------------------
# One-stop summary
# ---------------------------------------------------------------------------


def concentration_summary(
    weights,
    sector_map: Optional[Mapping[str, str]] = None,
    correlation_matrix: Optional[pd.DataFrame] = None,
    top_n: int = 3,
) -> dict:
    """All-in-one diagnostic for a portfolio.

    Returns::

        {
            "n_holdings":     int,        # active positions
            "hhi":            float,      # symbol HHI 0-10000
            "hhi_label":      str,        # "分散" / "中等集中" / "高度集中"
            "effective_n":    float,
            "top_3_weight":   float,      # 0-1 fraction
            "sector_exposure": {sector: weight} or None,
            "sector_hhi":     float or None,
            "sector_hhi_label": str or None,
            "correlation_hhi": float or None,  # only if corr provided
        }
    """
    w = _normalised_weights(weights)
    n = int((w > 0).sum())
    h = hhi(weights)
    out = {
        "n_holdings": n,
        "hhi": h,
        "hhi_label": hhi_label(h),
        "effective_n": effective_n(weights),
        f"top_{top_n}_weight": top_n_weight(weights, top_n),
    }
    if sector_map is not None:
        out["sector_exposure"] = sector_exposure(weights, sector_map)
        sh = sector_hhi(weights, sector_map)
        out["sector_hhi"] = sh
        out["sector_hhi_label"] = hhi_label(sh)
    if correlation_matrix is not None:
        out["correlation_hhi"] = correlation_hhi(weights, correlation_matrix)
    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _normalised_weights(weights) -> pd.Series:
    """Coerce dict/Series to a normalised non-negative pd.Series."""
    if weights is None:
        return pd.Series(dtype=float)
    if isinstance(weights, dict):
        s = pd.Series(weights, dtype=float)
    elif isinstance(weights, pd.Series):
        s = weights.astype(float)
    else:
        return pd.Series(dtype=float)
    s = s[s > 0]
    if s.empty:
        return s
    total = s.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return s / total
