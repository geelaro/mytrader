"""Drawdown attribution — decompose a portfolio drawdown to per-position.

Two attribution flavours
------------------------
1. :func:`attribute_drawdown` — needs a daily per-symbol value DataFrame
   (price × shares), gives accounting-exact dollar attribution.  Best for
   live use where you already have the holdings snapshot.

2. :func:`trade_overlap_attribution` (this module) — uses closed-trade
   records and counts the full PnL of every trade whose lifetime overlaps
   the [peak, trough] window.  Approximate but doesn't need daily prices
   — works directly with backtest output.

3. :func:`historical_drawdown_attribution` — wraps (2) to scan the top-N
   historical drawdown episodes from an equity curve and produce a
   per-episode breakdown table for review.

Companion to :mod:`analysis.drawdown` (which characterises the *shape* of
drawdowns: depth, duration, recovery).  This module answers a different
question: **inside a given drawdown episode, which positions caused it?**

Use case
--------
The portfolio is down 8% from a recent peak.  The risk dashboard says
"in drawdown" — but does not say *because of which holdings*.  This
module turns the aggregate drawdown into a ranked per-symbol attribution:

    本轮 -8.2% 回撤中, NVDA 贡献 -5.1%, TSLA -2.4%, SPY +0.3%

Inputs
------
- ``portfolio_value`` : daily total market value (cash + positions) time series
- ``position_values`` : DataFrame indexed by date, columns are symbols,
  each cell is the position's USD market value that day (price × shares).
  Missing/zero ⇒ position not held that day.

The decomposition is a simple-and-honest dollar bookkeeping:

    contribution_i_usd = position_values[i, trough] - position_values[i, peak]
    contribution_i_pct = contribution_i_usd / portfolio_value[peak] * 100

Symbols not held at the peak but opened mid-drawdown contribute their
``trough_value - 0`` (and conversely, symbols closed mid-drawdown
contribute the negative of their peak value).  Any residual between
``sum(contribution_i) - drawdown_pct`` is reported as ``unexplained_pct``
— typically near-zero for buy-and-hold; nonzero when there were big
cash flows (deposits / withdrawals) inside the window.

Use ``attribute_active_drawdown`` to auto-detect the most-recent peak
and current trough (no need to pick dates by hand).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def attribute_drawdown(
    portfolio_value: pd.Series,
    position_values: pd.DataFrame,
    peak_date: Optional[pd.Timestamp] = None,
    trough_date: Optional[pd.Timestamp] = None,
    top_n: Optional[int] = None,
) -> dict:
    """Decompose a peak-to-trough drawdown into per-symbol contributions.

    Parameters
    ----------
    portfolio_value : pd.Series
        Daily total portfolio value (cash + positions).  Index = dates.
    position_values : pd.DataFrame
        Daily per-symbol market value.  Index = dates (same as
        ``portfolio_value``), columns = symbols.  Cell value = that
        symbol's USD value on that date (0 / NaN = not held).
    peak_date : Timestamp, optional
        Start of the drawdown window.  If None, picked as the running
        cummax up to ``trough_date`` (or end of series).
    trough_date : Timestamp, optional
        End of the drawdown window.  If None, the date of the lowest
        portfolio value after ``peak_date`` (or last date if not in DD).
    top_n : int, optional
        If set, only return the top-N contributors by ``|contribution_usd|``.

    Returns
    -------
    dict::

        {
            "peak_date":        Timestamp,
            "trough_date":      Timestamp,
            "peak_value":       float,
            "trough_value":     float,
            "depth_usd":        float,   # negative
            "depth_pct":        float,   # negative, e.g. -8.2
            "by_symbol":        list[dict],
            "unexplained_usd":  float,   # residual (deposits/withdrawals)
            "unexplained_pct":  float,
        }

    ``by_symbol`` entries are dicts with: ``symbol``, ``peak_value``,
    ``trough_value``, ``contribution_usd``, ``contribution_pct``,
    sorted by ``|contribution_usd|`` descending.

    Edge cases
    ----------
    - Empty inputs or no drawdown ⇒ all-zeros summary, ``by_symbol=[]``.
    - peak_date == trough_date ⇒ 0% depth, no contributions.
    - Symbols opened mid-window: peak_value=0, contribution = +trough_value.
    """
    pv = _clean_series(portfolio_value)
    if pv.empty:
        return _empty_summary()

    pos = position_values if isinstance(position_values, pd.DataFrame) else pd.DataFrame()
    pos = pos.reindex(pv.index).fillna(0.0) if not pos.empty else pd.DataFrame(index=pv.index)

    # ---- Resolve window --------------------------------------------------
    if peak_date is None and trough_date is None:
        # Find the most-painful peak→trough in the whole series.
        cummax = pv.cummax()
        underwater = pv / cummax - 1
        trough_idx = underwater.idxmin()
        # Peak = the cummax-setting date at or before trough.
        peak_idx = pv.loc[:trough_idx].idxmax()
    elif peak_date is None:
        trough_idx = _normalise_date(trough_date, pv.index)
        peak_idx = pv.loc[:trough_idx].idxmax()
    elif trough_date is None:
        peak_idx = _normalise_date(peak_date, pv.index)
        trough_idx = pv.loc[peak_idx:].idxmin()
    else:
        peak_idx = _normalise_date(peak_date, pv.index)
        trough_idx = _normalise_date(trough_date, pv.index)

    if peak_idx is None or trough_idx is None or trough_idx < peak_idx:
        return _empty_summary()

    peak_v = float(pv.loc[peak_idx])
    trough_v = float(pv.loc[trough_idx])
    depth_usd = trough_v - peak_v
    depth_pct = (depth_usd / peak_v * 100) if peak_v != 0 else 0.0

    # ---- Per-symbol contributions ---------------------------------------
    rows: list[dict] = []
    explained_usd = 0.0
    if not pos.empty:
        peak_row = pos.loc[peak_idx] if peak_idx in pos.index else pd.Series(dtype=float)
        trough_row = pos.loc[trough_idx] if trough_idx in pos.index else pd.Series(dtype=float)
        symbols = sorted(set(pos.columns))
        for sym in symbols:
            p = float(peak_row.get(sym, 0) or 0)
            t = float(trough_row.get(sym, 0) or 0)
            contrib_usd = t - p
            if contrib_usd == 0 and p == 0 and t == 0:
                continue  # never held during the window
            contrib_pct = (contrib_usd / peak_v * 100) if peak_v != 0 else 0.0
            rows.append({
                "symbol": sym,
                "peak_value": p,
                "trough_value": t,
                "contribution_usd": contrib_usd,
                "contribution_pct": contrib_pct,
            })
            explained_usd += contrib_usd

    rows.sort(key=lambda r: -abs(r["contribution_usd"]))
    if top_n is not None and top_n > 0:
        rows = rows[:top_n]

    unexplained_usd = depth_usd - explained_usd
    unexplained_pct = (unexplained_usd / peak_v * 100) if peak_v != 0 else 0.0

    return {
        "peak_date": peak_idx,
        "trough_date": trough_idx,
        "peak_value": peak_v,
        "trough_value": trough_v,
        "depth_usd": depth_usd,
        "depth_pct": depth_pct,
        "by_symbol": rows,
        "unexplained_usd": unexplained_usd,
        "unexplained_pct": unexplained_pct,
    }


def attribute_active_drawdown(
    portfolio_value: pd.Series,
    position_values: pd.DataFrame,
    top_n: Optional[int] = None,
) -> dict:
    """Convenience: attribute the *currently ongoing* drawdown.

    Finds the most-recent peak (running cummax) and uses the latest
    date as the trough.  If the portfolio is at a new all-time high
    (not in drawdown), returns a zero-depth summary.
    """
    pv = _clean_series(portfolio_value)
    if pv.empty:
        return _empty_summary()

    cummax = pv.cummax()
    latest_idx = pv.index[-1]
    # If at ATH, no active drawdown.
    if pv.iloc[-1] >= cummax.iloc[-1]:
        return _empty_summary(latest=latest_idx)

    # Peak = the most recent date where cummax was set, looking only at
    # dates <= latest. We find the LAST date where pv == cummax before now.
    at_peak = pv == cummax
    peak_candidates = at_peak[at_peak].index
    if len(peak_candidates) == 0:
        peak_idx = pv.index[0]
    else:
        peak_idx = peak_candidates[-1]

    return attribute_drawdown(
        portfolio_value=pv,
        position_values=position_values,
        peak_date=peak_idx,
        trough_date=latest_idx,
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# Trade-overlap attribution (no daily prices needed)
# ---------------------------------------------------------------------------


def trade_overlap_attribution(
    trades,
    peak_date: pd.Timestamp,
    trough_date: pd.Timestamp,
    depth_usd: float = 0.0,
) -> dict:
    """Attribute a drawdown to per-symbol trade PnL via lifetime overlap.

    For each trade whose ``[entry_time, exit_time]`` intersects
    ``[peak_date, trough_date]``, attribute its **whole** PnL to its
    symbol.  Sum by symbol → ranked contributions.

    Approximation: a trade open for 200 days that crosses a 30-day
    drawdown gets its full lifetime PnL attributed to that window.  In
    practice this works well because trades only enter the count when
    they overlap the window, and across many trades the over/under
    counting tends to wash out.  Use :func:`attribute_drawdown` for
    accounting-exact dollar bookkeeping.

    Parameters
    ----------
    trades : iterable
        Iterable of objects with attributes ``entry_time``, ``exit_time``,
        ``symbol``, ``pnl``.  Compatible with ``PortfolioTrade``.
    peak_date, trough_date : pd.Timestamp
    depth_usd : float, optional
        Absolute dollar depth of the drawdown.  If provided, the
        ``contribution_pct`` field is filled in for each symbol;
        otherwise it's left as None.

    Returns
    -------
    dict::

        {
            "peak_date":   Timestamp,
            "trough_date": Timestamp,
            "by_symbol":   list[dict],   # sorted by pnl asc (worst first)
            "explained_usd":   float,
            "unexplained_usd": float,    # depth - explained (open-position PnL)
            "n_trades":    int,
        }

    ``by_symbol`` entries: ``{symbol, pnl, n_trades, contribution_pct}``.
    """
    by_symbol: dict[str, dict] = {}
    n_total = 0
    for t in trades or []:
        try:
            entry = pd.Timestamp(t.entry_time) if t.entry_time else None
            exit_ = (pd.Timestamp(t.exit_time) if t.exit_time
                     else trough_date)
        except Exception:
            continue
        if entry is None:
            continue
        if exit_ < peak_date or entry > trough_date:
            continue
        sym = str(t.symbol)
        pnl = float(t.pnl or 0)
        if sym not in by_symbol:
            by_symbol[sym] = {"symbol": sym, "pnl": 0.0, "n_trades": 0,
                              "contribution_pct": None}
        by_symbol[sym]["pnl"] += pnl
        by_symbol[sym]["n_trades"] += 1
        n_total += 1

    rows = sorted(by_symbol.values(), key=lambda r: r["pnl"])
    explained = sum(r["pnl"] for r in rows)

    if depth_usd:
        for r in rows:
            r["contribution_pct"] = (r["pnl"] / abs(depth_usd) * 100)

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "by_symbol": rows,
        "explained_usd": float(explained),
        "unexplained_usd": float(depth_usd - explained) if depth_usd else 0.0,
        "n_trades": n_total,
    }


def historical_drawdown_attribution(
    equity_curve: pd.Series,
    trades,
    top_n_episodes: int = 5,
    min_depth_pct: float = -1.0,
) -> list[dict]:
    """Per-episode drawdown attribution across the historical equity curve.

    Scans the equity curve for the worst N drawdown episodes (via
    :func:`analysis.drawdown.drawdown_episodes`), then applies
    :func:`trade_overlap_attribution` to each.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value.  Index = dates.
    trades : iterable
        Backtest trades (see :func:`trade_overlap_attribution`).
    top_n_episodes : int
        Limit to the N deepest episodes.
    min_depth_pct : float
        Skip shallow episodes (e.g. ``-1.0`` skips < 1%).  Pass 0 to keep all.

    Returns
    -------
    list[dict] — one per episode, sorted by depth ascending (worst first)::

        {
            "peak_date":             Timestamp,
            "trough_date":           Timestamp,
            "recovery_date":         Timestamp | NaT,
            "depth_pct":             float (negative),
            "depth_usd":             float (negative),
            "duration_to_trough":    int (days),
            "duration_to_recovery":  int (days, NaN if open),
            "by_symbol":             list[dict],
            "explained_usd":         float,
            "unexplained_usd":       float,
            "n_trades_in_window":    int,
        }
    """
    # Inline import to avoid circulars at module init
    from analysis.drawdown import drawdown_episodes

    pv = _clean_series(equity_curve)
    if pv.empty:
        return []

    returns = pv.pct_change(fill_method=None).dropna()
    episodes = drawdown_episodes(returns)
    if episodes.empty:
        return []

    out: list[dict] = []
    for _, ep in episodes.head(top_n_episodes).iterrows():
        if ep["depth_pct"] > min_depth_pct:
            continue
        peak_d = ep["peak_date"]
        trough_d = ep["trough_date"]
        peak_v = float(pv.loc[peak_d]) if peak_d in pv.index else 0.0
        trough_v = float(pv.loc[trough_d]) if trough_d in pv.index else 0.0
        depth_usd = trough_v - peak_v

        attr = trade_overlap_attribution(trades, peak_d, trough_d,
                                         depth_usd=depth_usd)
        out.append({
            "peak_date":             peak_d,
            "trough_date":           trough_d,
            "recovery_date":         ep.get("recovery_date"),
            "depth_pct":             float(ep["depth_pct"]),
            "depth_usd":             float(depth_usd),
            "duration_to_trough":    int(ep["duration_to_trough_days"]),
            "duration_to_recovery":  (
                int(ep["duration_to_recovery_days"])
                if pd.notna(ep["duration_to_recovery_days"]) else None
            ),
            "by_symbol":             attr["by_symbol"],
            "explained_usd":         attr["explained_usd"],
            "unexplained_usd":       attr["unexplained_usd"],
            "n_trades_in_window":    attr["n_trades"],
        })
    return out


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _clean_series(s: pd.Series) -> pd.Series:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return pd.Series(dtype=float)
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _normalise_date(date, index) -> Optional[pd.Timestamp]:
    """Coerce a date-like to an index member (nearest if not exact)."""
    if date is None or len(index) == 0:
        return None
    try:
        ts = pd.Timestamp(date)
    except Exception:
        return None
    if ts in index:
        return ts
    # Snap to nearest index member.
    pos = index.get_indexer([ts], method="nearest")[0]
    return index[pos] if pos >= 0 else None


def _empty_summary(latest: Optional[pd.Timestamp] = None) -> dict:
    return {
        "peak_date": latest,
        "trough_date": latest,
        "peak_value": 0.0,
        "trough_value": 0.0,
        "depth_usd": 0.0,
        "depth_pct": 0.0,
        "by_symbol": [],
        "unexplained_usd": 0.0,
        "unexplained_pct": 0.0,
    }
