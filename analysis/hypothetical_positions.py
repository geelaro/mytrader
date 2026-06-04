"""Compute hypothetical positions with Chandelier trailing stops.

Pure compute — given a config + a DataProvider, runs each watchlist
symbol's active single-strategy across history and identifies the most
recent open simulated buy.  For each open buy, computes:

- entry / current price + PnL%
- Chandelier-style trailing stop (highest-since-entry minus N · ATR)
- distance to stop, days held

The output schema matches what
:func:`live.risk_alerts.RiskAlerter.check_positions` consumes, so the
same calculation drives both the dashboard position-watch table and the
daemon's stop-proximity alert state machine.

Why this lives in ``analysis/``
-------------------------------
The function takes ``(config, target_date, provider)`` and returns a
list of dicts.  No broker calls, no DB writes, no Streamlit, no network
of its own.  Originally lived in ``live/position_stops.py`` because the
first consumer was the live alerter, but it's conceptually a research
/ read-only computation and belongs in the pure-compute analysis layer.

A re-export shim remains at ``live.position_stops.compute_hypothetical_positions``
for backward compatibility with the 7 existing callers; new code should
import directly from ``analysis.hypothetical_positions`` (or via the
top-level ``analysis`` package).

Position-dict schema::

    {
        "symbol":         str,
        "strategy":       str,
        "entry_date":     "YYYY-MM-DD",
        "entry_price":    float,
        "current_price":  float,
        "pnl_pct":        float,
        "stop_price":     float,
        "distance_pct":   float,
        "days_held":      int,
    }
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import pandas as pd

from strategy import STRATEGY_MAP

logger = logging.getLogger(__name__)


def _find_open_simulated_trade(df_sig: pd.DataFrame) -> Optional[int]:
    """Locate the most recent unclosed buy signal in ``df_sig``.

    Scans the ``Signal`` column from the end backward.  A buy (1) with no
    sell (-1) after it is treated as an open simulated position.  Returns
    the bar index of the entry, or None if no open trade.
    """
    if "Signal" not in df_sig.columns:
        return None
    signals = df_sig["Signal"].values
    for i in range(len(signals) - 1, -1, -1):
        if signals[i] == -1:
            return None
        if signals[i] == 1:
            return i
    return None


def compute_hypothetical_positions(
    config: dict,
    target_date,
    provider,
    symbols: Optional[Iterable[str]] = None,
) -> List[dict]:
    """For each watchlist symbol with an active single-strategy, return a
    hypothetical-position dict including Chandelier trailing stop.

    Parameters
    ----------
    config : dict
        Parsed watchlist.toml.  Must contain ``watchlist`` list; optional
        ``scanner.lookback_years`` (default 3) and per-strategy ``strategy``
        param overrides.
    target_date : datetime.date | pd.Timestamp
        The "as-of" date.  Data is fetched up to this date.
    provider : DataProvider
        Anything exposing ``get_daily(symbol, start, end) -> DataFrame``.
    symbols : iterable of str, optional
        Restrict to this subset of watchlist symbols.  Defaults to all.

    Returns
    -------
    list of dict
        Sorted by ``distance_pct`` ascending (closest to stop first).
        Symbols without an open simulated trade, with missing ATR, or with
        zero/negative prices are skipped silently.
    """
    lookback_years = config.get("scanner", {}).get("lookback_years", 3)
    start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    # target_date may be a date object or string — normalise via Timestamp.
    end = pd.Timestamp(target_date).date().isoformat()

    sym_filter = set(symbols) if symbols is not None else None
    rows: List[dict] = []

    for item in config.get("watchlist", []):
        symbol = item.get("symbol")
        if not symbol:
            continue
        if sym_filter is not None and symbol not in sym_filter:
            continue
        strat_name = item.get("active", "")
        # Skip ensembles (list-typed active) or unknown strategies — we need
        # a single concrete strategy to estimate the Chandelier stop.
        if not isinstance(strat_name, str) or strat_name not in STRATEGY_MAP:
            continue

        try:
            df = provider.get_daily(symbol, start=start, end=end)
        except Exception as exc:
            logger.debug("position stops fetch failed for %s: %s", symbol, exc)
            continue
        if df is None or df.empty:
            continue

        params = config.get("strategy", {}).get(strat_name, {})
        try:
            strategy = STRATEGY_MAP[strat_name](**params)
            df_sig = strategy.calculate_indicators(df)
        except Exception as exc:
            logger.debug("position stops indicators failed for %s/%s: %s",
                         symbol, strat_name, exc)
            continue

        idx_entry = _find_open_simulated_trade(df_sig)
        if idx_entry is None:
            continue

        try:
            entry_date = df_sig.index[idx_entry]
            entry_price = float(df_sig["Close"].iloc[idx_entry])
            current_price = float(df_sig["Close"].iloc[-1])
            atr = float(df_sig["ATR"].iloc[-1]) if "ATR" in df_sig.columns else 0.0
        except (KeyError, IndexError):
            continue
        if atr <= 0 or entry_price <= 0:
            continue

        track_col = "High" if "High" in df_sig.columns else "Close"
        try:
            highest = float(df_sig[track_col].iloc[idx_entry:].max())
        except (KeyError, ValueError):
            continue

        trail_mult = float(getattr(strategy.params, "trail_atr_mult", 2.5))
        stop_price = highest - trail_mult * atr

        pnl_pct = (current_price / entry_price - 1) * 100
        dist_pct = (current_price - stop_price) / current_price * 100 if current_price > 0 else 0
        days_held = (df_sig.index[-1] - entry_date).days

        rows.append({
            "symbol": symbol,
            "strategy": strat_name,
            "entry_date": entry_date.date().isoformat(),
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "stop_price": stop_price,
            "distance_pct": dist_pct,
            "days_held": days_held,
        })

    rows.sort(key=lambda r: r["distance_pct"])
    return rows
