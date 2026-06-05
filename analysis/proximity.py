"""Proximity-to-trigger detection — surface "接近触发" signals.

The strategy classes only emit discrete signals (-1/0/+1) when their
hard conditions are fully met.  Edge cases — MACD just about to cross,
KDJ in extreme overbought, price testing a Donchian band — produce
``Signal=0`` and disappear from the "今日信号" view.

This module re-examines a scanner result and surfaces those edge
patterns as soft warnings, so the dashboard can show "no signal but
NVDA's daily MACD will likely cross bearish tomorrow" type info.

API
---
- :func:`proximity_warnings(result)` — per-scan detection (one symbol × one strategy)
- :func:`proximity_summary(results)`  — aggregate + rank across all scans

Warning levels
--------------
- ``"alert"`` — at the threshold; very likely to trigger within 1-2 bars
- ``"warn"``  — approaching; trigger within 3-5 bars likely if trend holds
- ``"info"``  — noteworthy state (e.g., KDJ J > 100) but no immediate trigger

Patterns detected (generic across strategies)
---------------------------------------------
- ``macd_near_cross``: ``|MACD − signal|`` small relative to magnitude
- ``kdj_kd_near_cross``: ``|K − D|`` < 3 (near KDJ golden/death cross)
- ``kdj_overbought_extreme``: ``J > 100`` (likely top reversal)
- ``kdj_oversold_extreme``: ``J < 0`` (likely bottom bounce)
- ``donchian_near_upper``: ``Close / Donchian_upper > 0.98``
- ``donchian_near_lower``: ``Close / Donchian_lower < 1.02``
- ``ma_near_breakout``: ``|Close/MA − 1| < 0.02 AND Close > MA``
- ``ma_near_breakdown``: ``|Close/MA − 1| < 0.02 AND Close < MA``
- ``near_n_day_high``: ``Close / N_day_high > 0.98``

Patterns are derived from indicator names produced by current strategies
(MACD/MACD_signal/K/D/J/Donchian_*/MA/N_day_high).  Strategies that
expose different indicators won't trigger anything — silent, not error.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# Pattern thresholds — tuned for daily US-equity indicators
_MACD_REL_THRESHOLD_ALERT = 0.05   # within 5% of magnitude
_MACD_REL_THRESHOLD_WARN = 0.15
_MACD_ABS_FLOOR = 0.10              # ignore tiny noise (penny stock-ish)

_KDJ_KD_GAP_ALERT = 1.0             # |K-D| < 1
_KDJ_KD_GAP_WARN = 3.0
_KDJ_J_OVERBOUGHT = 100.0
_KDJ_J_EXTREME_OVERBOUGHT = 110.0
_KDJ_J_OVERSOLD = 0.0
_KDJ_J_EXTREME_OVERSOLD = -10.0

_DONCHIAN_UPPER_ALERT = 0.99        # Close / Donchian_upper > 0.99
_DONCHIAN_UPPER_WARN = 0.98
_DONCHIAN_LOWER_ALERT = 1.01
_DONCHIAN_LOWER_WARN = 1.02

_MA_NEAR_ALERT = 0.01               # |Close/MA - 1| < 1%
_MA_NEAR_WARN = 0.02

_N_DAY_HIGH_ALERT = 0.99
_N_DAY_HIGH_WARN = 0.98


def proximity_warnings(result: dict) -> List[dict]:
    """Detect proximity-to-trigger patterns in one scan result.

    Parameters
    ----------
    result : dict
        One entry from :meth:`SignalScanner.scan` — must contain
        ``signal``, ``price``, and ``indicators`` (dict of numeric).

    Returns
    -------
    list of dict — empty if nothing detected.  Each entry::

        {
            "level":     "info" | "warn" | "alert",
            "direction": "bullish" | "bearish" | "neutral",
            "type":      pattern code,
            "message":   short human-readable description,
            "metric":    the numerical value that triggered the rule,
        }

    Already-signalling results (``signal != 0``) still get checked —
    callers can filter as they see fit.
    """
    out: List[dict] = []
    ind = result.get("indicators") or {}
    price = float(result.get("price") or 0)

    # ── MACD near cross ────────────────────────────────────────────
    macd = _num(ind.get("MACD"))
    macd_sig = _num(ind.get("MACD_signal"))
    if macd is not None and macd_sig is not None:
        gap = macd - macd_sig
        magnitude = max(abs(macd), abs(macd_sig), _MACD_ABS_FLOOR)
        rel = abs(gap) / magnitude
        if rel < _MACD_REL_THRESHOLD_WARN:
            level = "alert" if rel < _MACD_REL_THRESHOLD_ALERT else "warn"
            # When gap > 0 currently bullish → approaching bearish cross
            direction = "bearish" if gap > 0 else "bullish"
            cross = "死叉" if direction == "bearish" else "金叉"
            out.append({
                "level": level,
                "direction": direction,
                "type": "macd_near_cross",
                "message": (f"MACD 接近{cross} — "
                            f"diff={gap:+.2f}, 距 0 仅 {rel * 100:.1f}%"),
                "metric": float(gap),
            })

    # ── KDJ K-D near cross ─────────────────────────────────────────
    k = _num(ind.get("K"))
    d = _num(ind.get("D"))
    j = _num(ind.get("J"))
    if k is not None and d is not None:
        kd_gap = k - d
        if abs(kd_gap) < _KDJ_KD_GAP_WARN:
            level = "alert" if abs(kd_gap) < _KDJ_KD_GAP_ALERT else "warn"
            direction = "bearish" if kd_gap > 0 else "bullish"
            cross = "死叉" if direction == "bearish" else "金叉"
            out.append({
                "level": level,
                "direction": direction,
                "type": "kdj_kd_near_cross",
                "message": (f"KDJ 接近{cross} — K={k:.1f}, D={d:.1f}, "
                            f"间距 {kd_gap:+.1f}"),
                "metric": float(kd_gap),
            })

    # ── KDJ J extreme ──────────────────────────────────────────────
    if j is not None:
        if j > _KDJ_J_EXTREME_OVERBOUGHT:
            out.append({
                "level": "alert", "direction": "bearish",
                "type": "kdj_overbought_extreme",
                "message": f"KDJ-J={j:.1f} 极度超买 (>110) — 高概率回调",
                "metric": float(j),
            })
        elif j > _KDJ_J_OVERBOUGHT:
            out.append({
                "level": "warn", "direction": "bearish",
                "type": "kdj_overbought_extreme",
                "message": f"KDJ-J={j:.1f} 超买 (>100)",
                "metric": float(j),
            })
        elif j < _KDJ_J_EXTREME_OVERSOLD:
            out.append({
                "level": "alert", "direction": "bullish",
                "type": "kdj_oversold_extreme",
                "message": f"KDJ-J={j:.1f} 极度超卖 (<-10) — 高概率反弹",
                "metric": float(j),
            })
        elif j < _KDJ_J_OVERSOLD:
            out.append({
                "level": "warn", "direction": "bullish",
                "type": "kdj_oversold_extreme",
                "message": f"KDJ-J={j:.1f} 超卖 (<0)",
                "metric": float(j),
            })

    # ── Donchian breakout proximity ────────────────────────────────
    upper = _num(ind.get("Donchian_upper"))
    lower = _num(ind.get("Donchian_lower"))
    if price > 0 and upper is not None and upper > 0:
        ratio = price / upper
        if ratio > _DONCHIAN_UPPER_WARN:
            level = "alert" if ratio > _DONCHIAN_UPPER_ALERT else "warn"
            out.append({
                "level": level, "direction": "bullish",
                "type": "donchian_near_upper",
                "message": (f"Close ${price:.2f} 接近 Donchian 上轨 "
                            f"${upper:.2f} (距 {(1 - ratio) * 100:+.2f}%)"),
                "metric": float(ratio),
            })
    if price > 0 and lower is not None and lower > 0:
        ratio = price / lower
        if ratio < _DONCHIAN_LOWER_WARN:
            level = "alert" if ratio < _DONCHIAN_LOWER_ALERT else "warn"
            out.append({
                "level": level, "direction": "bearish",
                "type": "donchian_near_lower",
                "message": (f"Close ${price:.2f} 接近 Donchian 下轨 "
                            f"${lower:.2f} (距 {(1 - ratio) * 100:+.2f}%)"),
                "metric": float(ratio),
            })

    # ── MA breakout proximity ──────────────────────────────────────
    ma = _num(ind.get("MA"))
    if price > 0 and ma is not None and ma > 0:
        rel_dist = (price / ma) - 1
        if abs(rel_dist) < _MA_NEAR_WARN:
            level = "alert" if abs(rel_dist) < _MA_NEAR_ALERT else "warn"
            if rel_dist > 0:
                out.append({
                    "level": level, "direction": "bearish",
                    "type": "ma_near_breakdown",
                    "message": (f"Close ${price:.2f} 紧贴 MA ${ma:.2f} 上方 "
                                f"(差 {rel_dist * 100:+.2f}%) — 跌破即转空"),
                    "metric": float(rel_dist),
                })
            else:
                out.append({
                    "level": level, "direction": "bullish",
                    "type": "ma_near_breakout",
                    "message": (f"Close ${price:.2f} 紧贴 MA ${ma:.2f} 下方 "
                                f"(差 {rel_dist * 100:+.2f}%) — 突破即转多"),
                    "metric": float(rel_dist),
                })

    # ── N-day high proximity ───────────────────────────────────────
    n_high = _num(ind.get("N_day_high"))
    if price > 0 and n_high is not None and n_high > 0:
        ratio = price / n_high
        if ratio > _N_DAY_HIGH_WARN:
            level = "alert" if ratio > _N_DAY_HIGH_ALERT else "warn"
            out.append({
                "level": level, "direction": "bullish",
                "type": "near_n_day_high",
                "message": (f"Close ${price:.2f} 接近 N 日高 ${n_high:.2f} "
                            f"(距 {(1 - ratio) * 100:+.2f}%)"),
                "metric": float(ratio),
            })

    return out


def proximity_summary(scan_results: Iterable[dict]) -> List[dict]:
    """Run :func:`proximity_warnings` across a full scan, ranked by urgency.

    Parameters
    ----------
    scan_results : iterable of scan-result dicts (output of SignalScanner.scan).

    Returns
    -------
    list of dict — one entry per (symbol, strategy) that produced at least
    one warning.  Sorted by urgency:
      1. has any "alert" → top
      2. has any "warn" → next
      3. info-only → bottom
    Within the same level, more warnings come first.

    Each entry::

        {
            "symbol":      str,
            "strategy":    str,
            "max_level":   "alert" | "warn" | "info",
            "warnings":    list of warning dicts,
            "n_warnings":  int,
        }
    """
    _level_rank = {"alert": 3, "warn": 2, "info": 1}
    rows: List[dict] = []
    for r in scan_results or []:
        warnings = proximity_warnings(r)
        if not warnings:
            continue
        max_level = max((w["level"] for w in warnings),
                        key=lambda lv: _level_rank.get(lv, 0))
        rows.append({
            "symbol": r.get("symbol", ""),
            "strategy": r.get("strategy", ""),
            "max_level": max_level,
            "warnings": warnings,
            "n_warnings": len(warnings),
        })
    rows.sort(
        key=lambda x: (-_level_rank.get(x["max_level"], 0), -x["n_warnings"]),
    )
    return rows


def proximity_summary_by_symbol(scan_results: Iterable[dict]) -> List[dict]:
    """Like :func:`proximity_summary` but collapse all strategies on the
    same symbol into one row.

    Each strategy's warnings remain accessible via ``by_strategy`` so the
    renderer can still show "AAPL → weekly_macd_kdj: ..., spy_ma_breakout: ...".

    Sort: alerts first, then warns, then by total ``n_warnings`` desc.

    Returns list of dict, one per symbol with at least one warning::

        {
            "symbol":       str,
            "max_level":    "alert" | "warn" | "info",
            "n_warnings":   int,            # total across all strategies
            "n_strategies": int,            # number of strategies contributing
            "by_strategy":  dict[str, list[dict]],   # {strategy_name: [warning, ...]}
        }
    """
    _level_rank = {"alert": 3, "warn": 2, "info": 1}

    by_sym: dict[str, dict] = {}
    for r in scan_results or []:
        warnings = proximity_warnings(r)
        if not warnings:
            continue
        sym = r.get("symbol", "")
        strat = r.get("strategy", "")
        slot = by_sym.setdefault(sym, {
            "symbol": sym,
            "max_level": "info",
            "n_warnings": 0,
            "by_strategy": {},
        })
        slot["by_strategy"].setdefault(strat, []).extend(warnings)
        slot["n_warnings"] += len(warnings)
        # Promote max_level if any warning is more severe
        for w in warnings:
            if _level_rank.get(w["level"], 0) > _level_rank.get(slot["max_level"], 0):
                slot["max_level"] = w["level"]

    rows = list(by_sym.values())
    for r in rows:
        r["n_strategies"] = len(r["by_strategy"])
    rows.sort(
        key=lambda x: (-_level_rank.get(x["max_level"], 0), -x["n_warnings"]),
    )
    return rows


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _num(x) -> Optional[float]:
    """Coerce to finite float, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v == float("inf") or v == float("-inf"):
        return None
    return v
