"""Unified DataProvider — gateway for all market-data requests.

Responsibilities
----------------
1. Route a symbol to the correct data source(s).
2. Check local cache; identify gaps.
3. Fetch only missing date ranges from external sources (fallback chain).
4. Merge + cache + return a complete DataFrame.
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd

from .cache import CacheManager
from .protocol import (
    OHLCV_COLUMNS,
    SOURCE_PRIORITY,
    DataSource,
    classify_symbol,
    CN_SYMBOLS,
)
from .sources import (
    YFinanceSource,
    TencentSource,
    SinaSource,
    AKShareSource,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trading-calendar helpers (used for gap detection)
# ---------------------------------------------------------------------------

# Approximate US holidays for gap-skipping
_US_HOLIDAYS: set[str] = set()
# US market holidays 2018-2030 (New Year, MLK, Presidents, Good Fri, Memorial,
# Juneteenth, Independence, Labor, Thanksgiving, Christmas)
_US_HOLIDAY_SKELETON = [
    # New Year's Day
    (1, 1),  # Jan 1
    # Martin Luther King Jr. Day — 3rd Monday of Jan
    # Presidents' Day — 3rd Monday of Feb
    # Memorial Day — last Monday of May
    # Juneteenth — Jun 19
    (6, 19),
    # Independence Day — Jul 4
    (7, 4),
    # Labor Day — 1st Monday of Sep
    # Thanksgiving — 4th Thursday of Nov
    # Christmas — Dec 25
    (12, 25),
]


def _is_weekend(d: pd.Timestamp) -> bool:
    return d.weekday() >= 5  # Saturday=5, Sunday=6


def _count_trading_gap(
    cached_end: Optional[str], req_start: str, max_gap_business_days: int = 30
) -> bool:
    """If the gap between *cached_end* and *req_start* is small enough,
    consider it bridged (e.g. a few weekends + minor holidays)."""
    if cached_end is None:
        return False
    ce = pd.Timestamp(cached_end)
    rs = pd.Timestamp(req_start)
    gap_days = (rs - ce).days
    # Loose heuristic: allow ~7 calendar days (≈5 business days) gap
    return gap_days <= 7


# ---------------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------------


class DataProvider:
    """Unified market-data access point.

    .. code-block:: python

        from data import DataProvider

        dp = DataProvider()
        df = dp.get_daily("AAPL",  start="2022-01-01", end="2024-12-31")
        df = dp.get_daily("510300", start="2022-01-01", end="2024-12-31")
    """

    def __init__(
        self,
        cache: Optional[CacheManager] = None,
        sources: Optional[List[DataSource]] = None,
    ):
        self.cache = cache or CacheManager()
        self._sources: List[DataSource] = sources or [
            TencentSource(),
            SinaSource(),
            AKShareSource(),
            YFinanceSource(),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return daily OHLCV for *symbol*.

        On first call, fetches from external sources and caches locally.
        Subsequent calls return cached data (incremental updates).

        Parameters
        ----------
        symbol : str
            Ticker (e.g. "AAPL", "510300", "sh510300").
        start : str | None
        end : str | None
        force_refresh : bool
            If True, skip cache and re-fetch everything.
        """
        sym = symbol.upper().strip()
        # Resolve CN aliases
        if sym in CN_SYMBOLS:
            sym = CN_SYMBOLS[sym].upper()

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # If no start, try to return whatever the cache has
        if start is None:
            cached_start, _ = self.cache.date_range(sym)
            start = cached_start or "2015-01-01"

        if not force_refresh:
            df = self._load_from_cache(sym, start, end)
            if self._is_complete(df, start, end):
                return df

        # Identify gaps and fetch
        gaps = self._find_gaps(sym, start, end, force_refresh)
        for gap_start, gap_end in gaps:
            fetched = self._fetch_from_sources(sym, gap_start, gap_end)
            if fetched is not None and not fetched.empty:
                source_name = self._resolve_source(sym).name
                self.cache.save(sym, fetched, source=source_name)

        return self._load_from_cache(sym, start, end)

    def cached_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (earliest, latest) cached dates for *symbol*."""
        return self.cache.date_range(symbol.upper())

    def list_sources(self, symbol: str) -> List[str]:
        """Return ordered list of source names that can serve *symbol*."""
        market = classify_symbol(symbol)
        sources = SOURCE_PRIORITY.get(market, SOURCE_PRIORITY["default"])
        return sources

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_source(self, symbol: str) -> DataSource:
        """Return the first source that supports *symbol*."""
        for src in self._sources:
            if src.supports(symbol):
                return src
        return self._sources[-1]  # YFinance as last resort

    def _load_from_cache(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        return self.cache.load(symbol, start, end)

    def _find_gaps(
        self, symbol: str, start: str, end: str, force: bool
    ) -> List[Tuple[str, str]]:
        """Determine date ranges that need fetching."""
        if force:
            return [(start, end)]
        return self.cache.missing_ranges(symbol, start, end)

    def _fetch_from_sources(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Try sources in priority order; return first successful fetch."""
        market = classify_symbol(symbol)
        priorities = SOURCE_PRIORITY.get(market, SOURCE_PRIORITY["default"])

        for source_name in priorities:
            src = self._find_source_by_name(source_name)
            if src is None or not src.supports(symbol):
                continue
            logger.info("Fetching %s from %s (%s → %s)", symbol, source_name, start, end)
            try:
                df = src.fetch(symbol, start, end)
                if df is not None and not df.empty:
                    logger.info("  → got %d bars from %s", len(df), source_name)
                    return df
            except Exception:
                logger.exception("Source %s failed for %s", source_name, symbol)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    def _find_source_by_name(self, name: str) -> Optional[DataSource]:
        for src in self._sources:
            if src.name == name:
                return src
        return None

    @staticmethod
    def _is_complete(df: pd.DataFrame, start: str, end: str) -> bool:
        """Heuristic: does *df* cover the entire requested range?"""
        if df is None or df.empty:
            return False
        first = df.index[0]
        last = df.index[-1]
        expected_first = pd.Timestamp(start)
        expected_last = pd.Timestamp(end)
        # Allow ~3 business days of slack (weekends / minor holidays)
        slack = pd.Timedelta(days=5)
        return (first <= expected_first + slack) and (last >= expected_last - slack)
