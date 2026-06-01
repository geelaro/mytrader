"""What-If portfolio rebalance preview.

Lets the user explore "if I trim X and add Y, what happens to my VaR,
HHI, sector tilt?" — without actually trading.  This is the workhorse
interaction for risk-decision support: combine with marginal VaR ("which
to trim") and the user has a closed-loop tool ("how much, and what
happens").

Two primitives:

- :func:`apply_rebalance` — pure dict math.  Add/subtract from current
  weights, drop anything that goes to zero or negative.

- :func:`compare_portfolios` — run the existing analytics
  (parametric VaR, concentration, sector) on both before and after,
  returning side-by-side + deltas so callers can render diff tables.
"""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from analysis.concentration import concentration_summary, hhi_label
from analysis.risk_decomposition import parametric_portfolio_var


def apply_rebalance(
    weights: Mapping[str, float],
    deltas: Mapping[str, float],
) -> dict:
    """Return a new weights dict = ``weights + deltas`` (positional).

    - Existing symbols see ``new = old + delta`` (delta can be negative).
    - New symbols (in deltas but not weights) are added with weight=delta.
    - Any resulting weight ≤ 0 is dropped (position closed).
    - Output is *not* normalised — caller decides whether to re-normalise.
      For risk-decomposition / concentration, normalisation is internal
      and the absolute scale is irrelevant, so this is the natural fit.
    """
    new = dict(weights)
    for sym, delta in deltas.items():
        new[sym] = new.get(sym, 0.0) + float(delta)
    return {s: w for s, w in new.items() if w > 0}


def compare_portfolios(
    prices: pd.DataFrame,
    before_weights: Mapping[str, float],
    after_weights: Mapping[str, float],
    sector_map: Optional[Mapping[str, str]] = None,
    confidence: float = 0.95,
) -> dict:
    """Snapshot risk + concentration metrics for two portfolios.

    Returns::

        {
            "before": {var_pct, hhi, hhi_label, effective_n,
                       top_3_weight, sector_hhi?, sector_exposure?},
            "after":  same shape,
            "deltas": {var_pct, hhi, effective_n, top_3_weight, sector_hhi?},
            "summary_text": "VaR ↓ 0.32pp / HHI ↓ 850 / 行业 HHI ↓ 1200"
        }

    Sectors fields only present when ``sector_map`` is provided.
    """
    before = _snapshot(prices, before_weights, sector_map, confidence)
    after = _snapshot(prices, after_weights, sector_map, confidence)

    delta_keys = ["var_pct", "hhi", "effective_n", "top_3_weight"]
    if sector_map is not None:
        delta_keys.append("sector_hhi")
    deltas = {
        k: after.get(k, 0.0) - before.get(k, 0.0)
        for k in delta_keys
        if k in before and k in after
    }
    return {
        "before": before,
        "after": after,
        "deltas": deltas,
        "summary_text": _format_summary(deltas),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _snapshot(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    sector_map: Optional[Mapping[str, str]],
    confidence: float,
) -> dict:
    """Compute a single portfolio's risk + concentration snapshot."""
    out: dict = {
        "var_pct": parametric_portfolio_var(prices, weights, confidence) * 100,
    }
    out.update(concentration_summary(weights, sector_map=sector_map))
    return out


def _format_summary(deltas: dict) -> str:
    """Human-readable one-liner of the most useful deltas.

    Arrow conventions: ↓ = decrease (good for risk metrics), ↑ = increase.
    """
    parts = []
    if "var_pct" in deltas:
        d = deltas["var_pct"]
        arrow = "↓" if d < 0 else "↑"
        parts.append(f"VaR {arrow} {abs(d):.2f}pp")
    if "hhi" in deltas:
        d = deltas["hhi"]
        arrow = "↓" if d < 0 else "↑"
        parts.append(f"HHI {arrow} {abs(d):.0f}")
    if "sector_hhi" in deltas:
        d = deltas["sector_hhi"]
        arrow = "↓" if d < 0 else "↑"
        parts.append(f"行业 HHI {arrow} {abs(d):.0f}")
    if "effective_n" in deltas:
        d = deltas["effective_n"]
        arrow = "↑" if d > 0 else "↓"  # more effective N is better
        parts.append(f"有效 N {arrow} {abs(d):.2f}")
    return " / ".join(parts) if parts else "无变化"
