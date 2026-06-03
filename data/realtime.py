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

logger = logging.getLogger(__name__)


# Realtime quote uses its OWN session, NOT shared with data.sources._yahoo_session.
# Why: _yahoo_session warms with a fc.yahoo.com cookie request needed by Yahoo's
# historical chart endpoint.  That cookie marks the session for tighter rate
# limits on the realtime endpoints — empirically spark/chart 429 within seconds
# of acquiring the cookie.  Realtime endpoints work fine WITHOUT cookies, so we
# build a minimal session here and keep it isolated.
_REALTIME_SESSION: Optional[requests.Session] = None
_REALTIME_SESSION_LOCK = threading.Lock()


def _realtime_session() -> requests.Session:
    """Lightweight Yahoo session for realtime endpoints — no fc.yahoo.com cookie."""
    global _REALTIME_SESSION
    if _REALTIME_SESSION is not None:
        return _REALTIME_SESSION
    with _REALTIME_SESSION_LOCK:
        if _REALTIME_SESSION is not None:
            return _REALTIME_SESSION
        s = requests.Session()
        s.trust_env = True
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        _REALTIME_SESSION = s
        return s


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

    Tries two JSON endpoints in order, first success wins:

    1. ``query1/v8/finance/spark`` — lightest, sometimes rate-limited.
    2. ``query2/v8/finance/chart`` — same backend, often 429s the worst.

    On total failure returns ``None`` — caller should fall back to the
    cached EOD VIX (CBOE).  Yahoo's HTML page is intentionally NOT used
    as a fallback because the CDN serves a stale snapshot from the last
    market close, so returning it as "realtime" actively misleads:
    falling back to the CBOE EOD value with a honest "yesterday's close"
    label is preferable.
    """
    now = time.time()
    with _cache_lock:
        cached = _cache.get("vix")
        if cached and (now - cached[0]) < ttl:
            _incr_metric("realtime_vix_cache_hits_total")
            return cached[1]

    _incr_metric("realtime_vix_cache_misses_total")
    value = _try_spark(timeout) or _try_chart(timeout)
    if value is not None:
        with _cache_lock:
            _cache["vix"] = (now, value)
    return value


def _incr_metric(name: str) -> None:
    """Best-effort metric increment — never raises."""
    try:
        from utils import metrics_server
        metrics_server.incr(name)
    except Exception:
        pass


def _try_spark(timeout: float = _DEFAULT_TIMEOUT) -> Optional[float]:
    """Yahoo `query1/v8/finance/spark` — lightweight chart endpoint.

    Response shape:
        {"^VIX": {"timestamp": [...], "close": [...], "previousClose": x,
                  "chartPreviousClose": y, ...}}
    """
    url = "https://query1.finance.yahoo.com/v8/finance/spark"
    params = {"symbols": "^VIX", "range": "1d", "interval": "1m"}
    try:
        s = _realtime_session()
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
        s = _realtime_session()
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
