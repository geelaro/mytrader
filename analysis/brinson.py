"""Brinson performance attribution.

Decomposes portfolio active return (vs benchmark) into three effects per sector:

- **Allocation** — gain/loss from over- or under-weighting a sector vs benchmark.
  Formula: ``(w_p − w_b) × r_b``.
  A positive allocation effect means you overweighted a sector that did well
  (or underweighted one that did poorly).

- **Selection** — gain/loss from picking better-than-sector stocks within
  each sector.  Formula: ``w_b × (r_p − r_b)``.
  Isolates the stock-picking skill, scaled by *benchmark* weight (the
  "what would I have earned at neutral weight" baseline).

- **Interaction** — the cross-term ``(w_p − w_b) × (r_p − r_b)``.
  Rewards consistency between allocation and selection: overweighted a
  sector AND beat it.  Often reported but rarely the dominant effect.

By construction, ``Σᵢ (alloc + sel + int) = portfolio_return − benchmark_return``
(Brinson–Hood–Beebower 1986).

Conventions
-----------
- Weights sum to 1 (normalised internally).  Long-only.
- Returns are total returns over the period in decimal form (0.05 = +5%).
- Sectors keyed by string name; missing keys treated as zero weight / zero return.

Benchmark sector returns
------------------------
For US equities we use the SPDR Select Sector ETFs as sector proxies
(:data:`SECTOR_ETF`).  Their daily history is fetched via the existing
:class:`DataProvider` chain.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sector → SPDR ETF map
# ---------------------------------------------------------------------------

SECTOR_ETF: dict[str, str] = {
    "Technology":    "XLK",
    "Financial":     "XLF",
    "Consumer":      "XLY",   # Consumer Discretionary (most "Consumer" labels)
    "Automotive":    "XLY",   # Auto sits inside Consumer Discretionary
    "Healthcare":    "XLV",
    "Energy":        "XLE",
    "Industrial":    "XLI",
    "Utilities":     "XLU",
    "Materials":     "XLB",
    "RealEstate":    "XLRE",
    "Communication": "XLC",
}

# Equal-weight 11-sector benchmark — simple baseline.  Real SPY sector
# weights are typically dominated by Tech (~30%) and would require a
# separate data feed to keep updated; equal-weight is the conventional
# "what if I had no view" reference.
EQUAL_WEIGHT_BENCHMARK: dict[str, float] = {sym: 1 / 11 for sym in (
    "Technology", "Financial", "Consumer", "Healthcare", "Energy",
    "Industrial", "Utilities", "Materials", "RealEstate",
    "Communication", "Cash",  # 11th bucket
)}


# ---------------------------------------------------------------------------
# Core BHB calculation
# ---------------------------------------------------------------------------


def brinson_attribution(
    portfolio_weights: Mapping[str, float],
    portfolio_returns: Mapping[str, float],
    benchmark_weights: Mapping[str, float],
    benchmark_returns: Mapping[str, float],
) -> dict:
    """Decompose portfolio active return into allocation / selection / interaction.

    Parameters
    ----------
    portfolio_weights, benchmark_weights : dict[sector, weight]
        Weights in each sector.  Normalised internally.
    portfolio_returns, benchmark_returns : dict[sector, return]
        Total return per sector over the period, decimal form.

    Returns
    -------
    dict::

        {
            "by_sector": DataFrame indexed by sector with columns
                w_p, w_b, r_p, r_b, allocation, selection, interaction, total
            "totals": {
                "portfolio_return", "benchmark_return", "active_return",
                "allocation", "selection", "interaction",
            }
        }

    Missing sector keys are treated as zero weight / zero return on that side.
    Sectors with zero weight in both portfolio and benchmark are dropped.
    """
    w_p = _normalise(portfolio_weights)
    w_b = _normalise(benchmark_weights)

    sectors = sorted(set(w_p) | set(w_b))
    rows = []
    for sec in sectors:
        wp_i = float(w_p.get(sec, 0.0))
        wb_i = float(w_b.get(sec, 0.0))
        if wp_i == 0 and wb_i == 0:
            continue
        rp_i = float(portfolio_returns.get(sec, 0.0))
        rb_i = float(benchmark_returns.get(sec, 0.0))

        alloc = (wp_i - wb_i) * rb_i
        sel = wb_i * (rp_i - rb_i)
        inter = (wp_i - wb_i) * (rp_i - rb_i)
        rows.append({
            "sector": sec,
            "w_p": wp_i,
            "w_b": wb_i,
            "r_p": rp_i,
            "r_b": rb_i,
            "allocation": alloc,
            "selection": sel,
            "interaction": inter,
            "total": alloc + sel + inter,
        })

    df = pd.DataFrame(rows).set_index("sector") if rows else _empty_sector_df()

    pr = sum(w_p.get(s, 0) * portfolio_returns.get(s, 0) for s in sectors)
    br = sum(w_b.get(s, 0) * benchmark_returns.get(s, 0) for s in sectors)
    totals = {
        "portfolio_return": float(pr),
        "benchmark_return": float(br),
        "active_return": float(pr - br),
        "allocation": float(df["allocation"].sum()) if not df.empty else 0.0,
        "selection": float(df["selection"].sum()) if not df.empty else 0.0,
        "interaction": float(df["interaction"].sum()) if not df.empty else 0.0,
    }
    return {"by_sector": df, "totals": totals}


# ---------------------------------------------------------------------------
# Data helpers — turn positions + provider into Brinson inputs
# ---------------------------------------------------------------------------


def portfolio_sector_breakdown(
    symbols: Iterable[str],
    sector_map: Mapping[str, str],
    period_returns: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[dict, dict]:
    """Aggregate per-symbol returns into per-sector weights and returns.

    Parameters
    ----------
    symbols : iterable of str
        Portfolio symbols (e.g. from compute_hypothetical_positions).
    sector_map : dict[symbol, sector]
        Symbol → sector classification (use ``utils.sectors.DEFAULT_SECTORS``).
    period_returns : dict[symbol, return]
        Total return per symbol over the period.
    weights : dict[symbol, weight], optional
        Symbol weights (default: equal weight across all symbols).

    Returns
    -------
    (sector_weights, sector_returns) — both dicts keyed by sector.

    Symbols missing from ``sector_map`` go to ``"Unknown"``.  Symbols
    missing from ``period_returns`` are dropped (no data).
    """
    syms = [s for s in symbols if s in period_returns]
    if not syms:
        return {}, {}
    if weights is None:
        weights = {s: 1.0 for s in syms}

    by_sector_w: dict[str, float] = {}
    by_sector_wr: dict[str, float] = {}
    for sym in syms:
        sec = sector_map.get(sym, "Unknown")
        w = float(weights.get(sym, 0))
        if w <= 0:
            continue
        r = float(period_returns[sym])
        by_sector_w[sec] = by_sector_w.get(sec, 0.0) + w
        by_sector_wr[sec] = by_sector_wr.get(sec, 0.0) + w * r

    # Normalise weights and average returns
    total_w = sum(by_sector_w.values())
    if total_w <= 0:
        return {}, {}
    sector_weights = {s: w / total_w for s, w in by_sector_w.items()}
    sector_returns = {
        s: by_sector_wr[s] / w if w > 0 else 0.0
        for s, w in by_sector_w.items()
    }
    return sector_weights, sector_returns


def compute_period_returns(
    prices: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float]:
    """Total return per column over [start, end].

    Skips symbols with insufficient data (need bars on both endpoints, or
    close to them).  Returns a dict mapping symbol → decimal return.
    """
    if prices is None or prices.empty:
        return {}
    window = prices.loc[
        (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    ]
    if window.empty or len(window) < 2:
        return {}

    out: dict[str, float] = {}
    for sym in window.columns:
        series = window[sym].dropna()
        if len(series) < 2:
            continue
        first = float(series.iloc[0])
        last = float(series.iloc[-1])
        if first <= 0:
            continue
        out[sym] = last / first - 1
    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    """Drop non-positive weights, normalise positive ones to sum to 1."""
    pos = {s: float(w) for s, w in weights.items() if float(w) > 0}
    total = sum(pos.values())
    if total <= 0:
        return {}
    return {s: w / total for s, w in pos.items()}


def _empty_sector_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "w_p", "w_b", "r_p", "r_b",
        "allocation", "selection", "interaction", "total",
    ])
