"""Shared signal-scanning engine — used by daily.py and live_trader.py.

Extracted from the ~70% duplicated logic that both entry points previously
reimplemented: fetch bar data per symbol, run each strategy on it, and
collect the last-bar signal/price/ATR/indicators snapshot.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP
from utils import get_logger

logger = get_logger("scanner")


class SignalScanner:
    """Run configured strategies across watchlist symbols for a target date.

    Parameters
    ----------
    provider : DataProvider
    cache : CacheManager, optional
        If provided, signals are persisted to the signal_history table.
    lookback_years : int
        How many years of bar data to fetch before scanning.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[CacheManager] = None,
        lookback_years: int = 3,
    ):
        self.provider = provider
        self.cache = cache
        self.lookback_years = lookback_years
        self._data_cache: Dict[str, pd.DataFrame] = {}  # symbol → daily df
        self._weekly_cache: Dict[str, pd.DataFrame] = {}  # symbol → weekly df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        config: dict,
        target_date: Optional[str] = None,
        orphan_positions: Optional[List[dict]] = None,
        monitors: bool = True,
    ) -> List[dict]:
        """Run strategies on every watchlist (and optional orphan) symbol.

        Returns
        -------
        list[dict]
            Each dict: symbol, name, strategy, signal, price, atr,
            bar_date, indicators, orphan (bool).
        """
        if target_date is None:
            target_date = date.today().isoformat()

        start = (
            pd.Timestamp(target_date) - pd.DateOffset(years=self.lookback_years)
        ).strftime("%Y-%m-%d")

        results: List[dict] = []
        self._data_cache.clear()
        self._weekly_cache.clear()

        # --- watchlist symbols -------------------------------------------------
        for item in config.get("watchlist", []):
            results.extend(
                self._scan_symbol(item, start, target_date, active_first=monitors)
            )

        # --- orphan positions --------------------------------------------------
        if orphan_positions:
            for op in orphan_positions:
                if op["symbol"] in {r["symbol"] for r in results}:
                    continue  # already scanned via watchlist
                item = {
                    "symbol": op["symbol"],
                    "name": op.get("name", op["symbol"]),
                    "active": op["strategy"],
                    "_orphan": True,
                }
                results.extend(
                    self._scan_symbol(item, start, target_date, active_first=False)
                )

        self._data_cache.clear()
        self._weekly_cache.clear()
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan_symbol(
        self,
        item: dict,
        start: str,
        target_date: str,
        active_first: bool = True,
    ) -> List[dict]:
        """Scan one symbol across its active + monitor strategies."""
        symbol = item["symbol"]
        name = item.get("name", symbol)
        is_orphan = item.get("_orphan", False)
        active_strat = item.get("active", "")
        monitor_list = item.get("monitor", []) if active_first else []

        strategy_names = [active_strat] + monitor_list if active_strat else monitor_list
        if not strategy_names:
            return []

        df = self._fetch_data(symbol, start, target_date)
        if df is None or df.empty:
            return []

        df_weekly = self._fetch_weekly(symbol, start, target_date)

        bar_date = (
            target_date if target_date in df.index.strftime("%Y-%m-%d")
            else df.index[-1].strftime("%Y-%m-%d")
        )

        results: List[dict] = []
        for strat_name in strategy_names:
            if strat_name not in STRATEGY_MAP:
                continue
            strat_params = item.get("params", {})
            strategy = STRATEGY_MAP[strat_name](**strat_params)
            try:
                try:
                    df_sig = strategy.calculate_indicators(df, df_weekly=df_weekly)
                except TypeError:
                    df_sig = strategy.calculate_indicators(df)
            except Exception:
                logger.exception("策略计算失败: %s %s", symbol, strat_name)
                continue

            last_idx = -1
            signal = int(df_sig["Signal"].iloc[last_idx])
            price = float(df_sig["Close"].iloc[last_idx])
            atr = float(df_sig["ATR"].iloc[last_idx]) if "ATR" in df_sig.columns else 0

            indicators: Dict[str, float] = {}
            for col in df_sig.columns:
                if col not in ("Open", "High", "Low", "Close", "Volume", "Signal"):
                    val = df_sig[col].iloc[last_idx]
                    if isinstance(val, (float, int)) and not pd.isna(val):
                        indicators[col] = round(float(val), 4)

            # Persist to cache if available
            if self.cache is not None:
                self.cache.save_signal(
                    scan_date=target_date,
                    symbol=symbol,
                    strategy=strat_name,
                    bar_date=bar_date,
                    signal=signal,
                    price=price,
                    atr=atr,
                    indicators=json.dumps(indicators, ensure_ascii=False),
                )

            results.append({
                "symbol": symbol,
                "name": name,
                "strategy": strat_name,
                "signal": signal,
                "price": price,
                "atr": atr,
                "bar_date": bar_date,
                "indicators": indicators,
                "orphan": is_orphan,
            })

        return results

    def _fetch_data(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV for *symbol*, caching per-symbol within a scan pass."""
        if symbol not in self._data_cache:
            self._data_cache[symbol] = self.provider.get_daily(symbol, start=start, end=end)
        df = self._data_cache[symbol]
        if df is None or df.empty:
            return None
        return df

    def _fetch_weekly(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch weekly OHLCV for *symbol* (resampled from daily, cached in-scan)."""
        if symbol not in self._weekly_cache:
            daily = self._fetch_data(symbol, start, end)
            if daily is not None and not daily.empty:
                self._weekly_cache[symbol] = (
                    daily.resample("W-FRI")
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna()
                )
            else:
                self._weekly_cache[symbol] = pd.DataFrame()
        df = self._weekly_cache[symbol]
        if df is None or df.empty:
            return None
        return df


# ---------------------------------------------------------------------------
# Helper: merge strategy params from watchlist.toml into scan item
# ---------------------------------------------------------------------------

def enrich_scan_items(config: dict) -> List[dict]:
    """Return watchlist items with strategy params baked in."""
    items: List[dict] = []
    for item in config.get("watchlist", []):
        active = item.get("active", "")
        merged = dict(item)
        merged["params"] = config.get("strategy", {}).get(active, {})
        items.append(merged)
    return items
