"""Correlation-based concentration analysis.

Beyond HHI on weights and sectors: how many *independent* bets does the
portfolio actually hold?  Two perfectly correlated 50/50 positions are
mathematically one bet, not two.  Sector HHI catches the egregious case
("all Tech") but misses high-correlation pairs within or across sectors.

Two complementary lenses:

- :func:`effective_bets` — PCA-based.  Σλᵢ explains correlation matrix
  variance; concentration of eigenvalues tells you the effective number
  of independent factor directions.

- :func:`correlation_clusters` — hierarchical clustering.  Groups
  symbols with average pairwise correlation above ``threshold``; each
  cluster is "one bet".

- :func:`max_pairwise_correlation` — fastest red-flag detector.  If any
  pair correlation > 0.85, you essentially have one position.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------


def correlation_matrix(prices: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Pairwise correlation of daily returns.  Requires ≥ ``min_obs`` rows."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    rets = prices.pct_change().dropna()
    if len(rets) < min_obs:
        return pd.DataFrame()
    return rets.corr()


def max_pairwise_correlation(
    prices: pd.DataFrame,
    min_obs: int = 30,
) -> Optional[dict]:
    """Largest off-diagonal correlation in the panel.

    Returns ``{"symbols": (a, b), "correlation": float}`` or None if data
    is insufficient or only one symbol is present.
    """
    corr = correlation_matrix(prices, min_obs)
    if corr.empty or len(corr) < 2:
        return None
    # Mask diagonal
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    if c.isna().all().all():
        return None
    # Find argmax over flattened upper-triangle
    upper = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
    val = upper.max().max()
    if pd.isna(val):
        return None
    # Locate the cell
    idx = upper.stack().idxmax()
    return {"symbols": tuple(idx), "correlation": float(val)}


# ---------------------------------------------------------------------------
# Effective Bets (PCA-based)
# ---------------------------------------------------------------------------


def effective_bets(
    prices: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    min_obs: int = 30,
) -> dict:
    """Number of independent directions of risk in the portfolio.

    Method
    ------
    Two flavours:

    1. **Unweighted** (when ``weights`` is None): N_eff = 1 / Σ(λᵢ/Σλ)²
       where λᵢ are eigenvalues of the correlation matrix.  This is the
       inverse Herfindahl of eigenvalue-explained variance — interprets
       the matrix as a probability distribution over directions.

    2. **Weighted by portfolio variance**: project portfolio variance
       onto principal components and compute the same inverse-Herfindahl
       on those contributions.  More accurate for portfolios with skewed
       weights.

    Returns dict::

        {
            "n_symbols":           int,
            "effective_n":         float,  # ~equal to n_symbols if independent
            "concentration_ratio": float,  # effective_n / n_symbols, ≤ 1
            "top_eigenvalue_pct":  float,  # % variance in largest eigenvalue
        }

    Empty dict if insufficient data.
    """
    corr = correlation_matrix(prices, min_obs)
    n = len(corr)
    if n == 0:
        return {}
    if n == 1:
        return {
            "n_symbols": 1,
            "effective_n": 1.0,
            "concentration_ratio": 1.0,
            "top_eigenvalue_pct": 100.0,
        }

    # Eigendecomposition of the correlation matrix (symmetric → eigh)
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = eigvals[eigvals > 0]  # numerical noise: drop tiny negatives
    if eigvals.size == 0:
        return {}

    if weights is None:
        # Unweighted: inverse Herfindahl of eigenvalue shares
        shares = eigvals / eigvals.sum()
        n_eff = 1.0 / float((shares ** 2).sum())
        top_pct = float(shares.max()) * 100
    else:
        # Weighted: project portfolio onto PCs
        # σ²_p = Σᵢ (qᵀ_i w)² λᵢ, where q_i is the i-th eigenvector
        used = {s: w for s, w in weights.items() if s in corr.columns and w > 0}
        total_w = sum(used.values())
        if total_w <= 0:
            return {}
        w_vec = np.array([used.get(s, 0) / total_w for s in corr.columns])
        eigvals_full, eigvecs = np.linalg.eigh(corr.values)
        # Component variance contributions: (qᵀ w)² × λ
        proj = eigvecs.T @ w_vec
        contributions = (proj ** 2) * eigvals_full
        contributions = contributions[contributions > 0]
        if contributions.size == 0:
            return {}
        shares = contributions / contributions.sum()
        n_eff = 1.0 / float((shares ** 2).sum())
        top_pct = float(shares.max()) * 100

    return {
        "n_symbols": int(n),
        "effective_n": float(n_eff),
        "concentration_ratio": float(n_eff / n),
        "top_eigenvalue_pct": top_pct,
    }


# ---------------------------------------------------------------------------
# Hierarchical correlation clustering
# ---------------------------------------------------------------------------


def correlation_clusters(
    prices: pd.DataFrame,
    distance_threshold: float = 0.3,
    method: str = "average",
    min_obs: int = 30,
) -> dict:
    """Hierarchical clustering on the correlation distance matrix.

    Distance d = 1 − |ρ|.  Symbols with d < ``distance_threshold`` (i.e.
    |ρ| > 1 − threshold = 0.7 at default) end up in the same cluster.

    Returns dict::

        {
            "clusters":      {cluster_id: [symbols]},  # id starts at 1
            "n_clusters":    int,
            "n_symbols":     int,
            "linkage_method": str,
            "distance_threshold": float,
        }

    Empty dict if data insufficient.
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    corr = correlation_matrix(prices, min_obs)
    n = len(corr)
    if n < 2:
        return {}

    # Distance matrix
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0)
    # squareform needs a strictly non-negative symmetric matrix; clip tiny
    # negatives from float noise.
    dist_arr = np.clip(dist.values, 0, None)
    np.fill_diagonal(dist_arr, 0)
    try:
        condensed = squareform(dist_arr, checks=False)
    except Exception:
        return {}

    try:
        linkage = hierarchy.linkage(condensed, method=method)
        labels = hierarchy.fcluster(
            linkage, t=distance_threshold, criterion="distance",
        )
    except Exception:
        return {}

    clusters: dict = {}
    for sym, label in zip(corr.columns, labels):
        clusters.setdefault(int(label), []).append(str(sym))
    return {
        "clusters": clusters,
        "n_clusters": len(clusters),
        "n_symbols": n,
        "linkage_method": method,
        "distance_threshold": distance_threshold,
    }


# ---------------------------------------------------------------------------
# All-in-one summary
# ---------------------------------------------------------------------------


def correlation_summary(
    prices: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    cluster_distance: float = 0.3,
) -> dict:
    """One-stop dict combining all three lenses for dashboard rendering."""
    out: dict = {
        "max_pair": max_pairwise_correlation(prices),
        "effective_bets": effective_bets(prices, weights=weights),
        "clusters": correlation_clusters(prices, distance_threshold=cluster_distance),
    }
    return out
