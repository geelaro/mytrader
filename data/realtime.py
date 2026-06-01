"""Real-time quote endpoints — bypass the cache for live values.

The persistent cache stores EOD bars (one row per trading day).  Some
decisions need *now* values, especially the VIX-spike alert: a daemon
tick at 22:00 BJT shouldn't fire on yesterday's stale VIX when SPY is
clearly crashing intraday.

Currently only VIX is supported, via Yahoo Finance chart v8 with 1-minute
interval.  Yahoo's quote is delayed ~15 minutes for free users, which is
still good enough for our 5-minute daemon tick alerts.

Failure mode
------------
Any error (network, parse, missing fields) returns ``None``.  Callers
must fall back to the cached EOD value.  The cache layer is the source
of truth; real-time is a best-effort overlay.

Caching
-------
Each successful fetch is cached in-process for ``ttl`` seconds (default
60).  Prevents hammering Yahoo when multiple callers within the same
daemon tick (alerter + dashboard + risk_monitor) ask for VIX.  Reset by
calling :func:`reset_cache` (mainly for tests).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import requests

from data.sources import _yahoo_session

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 60.0
_DEFAULT_TIMEOUT = 5.0

# Module-level cache: {key: (timestamp, value)}.  Lock-protected for
# multi-threaded daemons that may call from notify worker + main tick.
_cache: dict = {}
_cache_lock = threading.Lock()


def get_realtime_vix(
    timeout: float = _DEFAULT_TIMEOUT,
    ttl: float = _DEFAULT_TTL,
) -> Optional[float]:
    """Fetch current VIX value from Yahoo (≈15-minute delayed for free).

    Tries two Yahoo endpoints in order — the lighter ``spark`` endpoint
    first (rarely rate-limited), then the heavier ``chart/v8`` endpoint
    that the rest of the codebase already uses.  The first success wins.

    Returns float in decimal points (e.g. 18.45 for VIX=18.45) or
    ``None`` on total failure.  Callers should treat None as "fall back
    to cached EOD VIX".
    """
    now = time.time()
    with _cache_lock:
        cached = _cache.get("vix")
        if cached and (now - cached[0]) < ttl:
            return cached[1]

    value = _try_spark() or _try_chart(timeout)
    if value is not None:
        with _cache_lock:
            _cache["vix"] = (now, value)
    return value


def _try_spark(timeout: float = _DEFAULT_TIMEOUT) -> Optional[float]:
    """Yahoo `query1/v8/finance/spark` — lightweight chart endpoint.

    Response shape:
        {"^VIX": {"timestamp": [...], "close": [...], "previousClose": x,
                  "chartPreviousClose": y, ...}}
    """
    url = "https://query1.finance.yahoo.com/v8/finance/spark"
    params = {"symbols": "^VIX", "range": "1d", "interval": "1m"}
    try:
        s = _yahoo_session()
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug("Realtime VIX spark fetch failed: %s", e)
        return None
    try:
        block = data.get("^VIX") or {}
        closes = block.get("close") or []
    except AttributeError:
        return None
    for c in reversed(closes):
        if isinstance(c, (int, float)) and c > 0:
            return float(c)
    # Fall back to previousClose / chartPreviousClose if intraday is empty
    for key in ("regularMarketPrice", "previousClose", "chartPreviousClose"):
        v = block.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _try_chart(timeout: float = _DEFAULT_TIMEOUT) -> Optional[float]:
    """Yahoo `query2/v8/finance/chart/^VIX` — heavier endpoint (fallback)."""
    url = "https://query2.finance.yahoo.com/v8/finance/chart/^VIX"
    params = {"interval": "1m", "range": "1d"}
    try:
        s = _yahoo_session()
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug("Realtime VIX chart fetch failed: %s", e)
        return None
    return _extract_latest_quote(data)


def reset_cache() -> None:
    """Clear the in-process realtime cache.  Used by tests."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _extract_latest_quote(data: dict) -> Optional[float]:
    """Parse Yahoo chart v8 response and return the latest valid close.

    Preference order:
      1. ``meta.regularMarketPrice`` — Yahoo's authoritative "last trade".
      2. Last non-null entry in ``indicators.quote[0].close`` — fallback
         for slightly older quote responses.
    """
    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(result, dict):
        return None

    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)

    try:
        closes = result["indicators"]["quote"][0].get("close", [])
    except (KeyError, IndexError, TypeError):
        return None
    for c in reversed(closes):
        if isinstance(c, (int, float)) and c > 0:
            return float(c)
    return None
