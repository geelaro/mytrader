"""Decision attribution — link decisions to outcomes.

What this answers
-----------------
"Of my last 90 days of trade decisions, how did the ones made under RED
risk-light perform vs GREEN?"

Approach
--------
:class:`live.decision_logger.DecisionLogger` writes one row to
``decision_history`` per actionable moment, capturing the risk context.
:class:`live.order_manager.OrderManager` writes one row to ``trade_pnl``
per fully-closed round-trip, keyed by ``order_id`` (the closing order's
id).  This module joins the two on order_id and computes hit-rate stats
grouped by any context field.

Caveats
-------
- Entry-side decisions (trade_buy) cannot be joined directly — the buy
  ``order_id`` doesn't match the close.  We match by **symbol** and
  next-after-entry close, which gives the right answer for FIFO
  one-position-per-symbol strategies.  Multi-leg / overlapping positions
  are out of scope for v1.
- Decisions with no matching exit yet (open positions, kill-switch
  events without close fills) come back with ``pnl=None`` and are
  excluded from hit-rate aggregates by default.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional


def join_decision_pnl(
    decisions: Iterable[dict],
    trade_pnl_rows: Iterable[dict],
) -> list[dict]:
    """Attach realised PnL to each decision via order_id (close) or symbol (entry).

    Parameters
    ----------
    decisions : iterable of dicts from :meth:`StateStore.query_decisions`.
    trade_pnl_rows : iterable of dicts from :meth:`StateStore.query_trade_pnl`.

    Returns a NEW list of dict copies, each augmented with::

        {
            "pnl":           float | None,
            "pnl_pct":       float | None,
            "exit_date":     str | None,
            "matched_by":    "order_id" | "symbol" | None,
        }

    Match priority:
    1. ``decision.payload.order_id`` == ``trade_pnl.order_id`` (close-side decisions)
    2. ``decision.symbol`` + decision is a ``trade_buy`` and there exists a
       trade_pnl row with same symbol exited AFTER the decision time —
       attribute the next close to this entry.
    """
    trades = list(trade_pnl_rows or [])
    decisions = list(decisions or [])

    # ── Index trade_pnl ───────────────────────────────────────────────
    by_order_id: dict[str, dict] = {}
    by_symbol_exits: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        oid = t.get("order_id")
        if oid:
            by_order_id[str(oid)] = t
        sym = t.get("symbol")
        if sym:
            by_symbol_exits[sym].append(t)
    # Sort each symbol's exits by exit_date ascending so "next exit after
    # decision time" is a binary search.
    for sym in by_symbol_exits:
        by_symbol_exits[sym].sort(key=lambda r: r.get("exit_date") or "")

    out: list[dict] = []
    for d in decisions:
        row = dict(d)
        row["pnl"] = None
        row["pnl_pct"] = None
        row["exit_date"] = None
        row["matched_by"] = None

        payload = d.get("payload") or {}
        oid = payload.get("order_id") if isinstance(payload, dict) else None

        # Priority 1: exact order_id match (close-side)
        if oid and str(oid) in by_order_id:
            t = by_order_id[str(oid)]
            row["pnl"] = float(t.get("pnl") or 0)
            row["pnl_pct"] = float(t.get("pnl_pct") or 0)
            row["exit_date"] = t.get("exit_date")
            row["matched_by"] = "order_id"
            out.append(row)
            continue

        # Priority 2: trade_buy + next-after-entry close on same symbol
        if d.get("decision_type") == "trade_buy" and d.get("symbol"):
            decision_ts = d.get("ts") or ""
            candidates = by_symbol_exits.get(d["symbol"], [])
            for t in candidates:
                if (t.get("exit_date") or "") > decision_ts:
                    row["pnl"] = float(t.get("pnl") or 0)
                    row["pnl_pct"] = float(t.get("pnl_pct") or 0)
                    row["exit_date"] = t.get("exit_date")
                    row["matched_by"] = "symbol"
                    break

        out.append(row)

    return out


def hit_rate_by_group(
    decisions_with_pnl: Iterable[dict],
    group_by: str = "risk_light",
    include_open: bool = False,
) -> dict:
    """Aggregate win-rate, average PnL, total PnL grouped by ``group_by``.

    Parameters
    ----------
    decisions_with_pnl : output of :func:`join_decision_pnl`.
    group_by : str
        Field to group on — typically ``risk_light``, ``symbol``, or
        ``decision_type``.  Rows with None in this field are bucketed
        as ``"(unknown)"``.
    include_open : bool
        If False (default), decisions without a matched pnl are skipped.

    Returns
    -------
    dict::

        {
            "groups": {
                group_key: {
                    "n":          int,
                    "n_wins":     int,
                    "n_losses":   int,
                    "win_rate":   float,  # [0, 1]
                    "total_pnl":  float,
                    "avg_pnl":    float,
                    "median_pnl": float,
                },
                ...
            },
            "n_matched":     int,
            "n_unmatched":   int,
        }
    """
    rows = list(decisions_with_pnl or [])
    matched = [r for r in rows if r.get("pnl") is not None]
    unmatched = [r for r in rows if r.get("pnl") is None]

    rows_for_agg = rows if include_open else matched

    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows_for_agg:
        key = r.get(group_by)
        if key is None or key == "":
            key = "(unknown)"
        pnl = r.get("pnl")
        if pnl is None:
            continue
        buckets[key].append(float(pnl))

    groups: dict[str, dict] = {}
    for key, pnls in buckets.items():
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        sorted_pnls = sorted(pnls)
        n = len(pnls)
        median = (sorted_pnls[n // 2] if n % 2 == 1
                  else (sorted_pnls[n // 2 - 1] + sorted_pnls[n // 2]) / 2
                  if n > 0 else 0.0)
        groups[key] = {
            "n": n,
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": len(wins) / n if n else 0.0,
            "total_pnl": float(sum(pnls)),
            "avg_pnl": float(sum(pnls) / n) if n else 0.0,
            "median_pnl": float(median),
        }
    return {
        "groups": groups,
        "n_matched": len(matched),
        "n_unmatched": len(unmatched),
    }


def decision_attribution_summary(
    decisions: Iterable[dict],
    trade_pnl_rows: Iterable[dict],
    group_by: str = "risk_light",
) -> dict:
    """One-stop join + aggregate for dashboard rendering."""
    joined = join_decision_pnl(decisions, trade_pnl_rows)
    aggregated = hit_rate_by_group(joined, group_by=group_by)
    return {
        "decisions_with_pnl": joined,
        **aggregated,
        "group_by": group_by,
    }
