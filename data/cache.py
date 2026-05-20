"""SQLite-based OHLCV cache — local storage with incremental-update support."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pandas import Timestamp

from .protocol import OHLCV_COLUMNS


class CacheManager:
    """SQLite cache for daily OHLCV bars.

    Schema
    ------
    ohlcv_daily(symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume INTEGER, source TEXT,
                PRIMARY KEY (symbol, date))
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("MYTRADER_DB", "trading_data.db")
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol   TEXT    NOT NULL,
                date     TEXT    NOT NULL,
                open     REAL,
                high     REAL,
                low      REAL,
                close    REAL,
                volume   INTEGER,
                source   TEXT,
                PRIMARY KEY (symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_daily(symbol);
            CREATE INDEX IF NOT EXISTS idx_ohlcv_date  ON ohlcv_daily(date);

            CREATE TABLE IF NOT EXISTS signal_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT    NOT NULL,
                symbol    TEXT    NOT NULL,
                strategy  TEXT    NOT NULL,
                bar_date  TEXT    NOT NULL,
                signal    INTEGER NOT NULL,
                price     REAL,
                atr       REAL,
                indicators TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_signal_scan ON signal_history(scan_date);
            CREATE INDEX IF NOT EXISTS idx_signal_sym  ON signal_history(symbol);

            CREATE TABLE IF NOT EXISTS risk_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entry_prices (
                symbol     TEXT PRIMARY KEY,
                price      REAL NOT NULL,
                entry_date TEXT NOT NULL
            );

            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = NORMAL;
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Return cached bars for *symbol*, or empty DataFrame."""
        self.init_schema()
        query = "SELECT date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = ?"
        params = [symbol.upper()]

        if start:
            query += " AND date >= ?"
            params.append(str(pd.Timestamp(start).date()))
        if end:
            query += " AND date <= ?"
            params.append(str(pd.Timestamp(end).date()))

        query += " ORDER BY date ASC"
        df = pd.read_sql_query(
            query, self.conn, params=params, index_col="date", parse_dates=["date"]
        )
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # DB stores lowercase — map to canonical Title case
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }, inplace=True)

        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        return df[OHLCV_COLUMNS]

    def date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (earliest_date, latest_date) cached for *symbol*."""
        self.init_schema()
        cur = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM ohlcv_daily WHERE symbol = ?",
            [symbol.upper()],
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None, None
        return row[0], row[1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self, symbol: str, df: pd.DataFrame, source: str = ""
    ) -> int:
        """Insert or replace bars.  Returns number of rows written."""
        if df is None or df.empty:
            return 0

        self.init_schema()
        df = df.copy()

        # Ensure we have a date column
        if df.index.name != "date" and "date" not in df.columns:
            raise ValueError("DataFrame must have a 'date' index or column")
        if "date" not in df.columns:
            df["date"] = df.index.map(lambda x: str(pd.Timestamp(x).date()))

        df["symbol"] = symbol.upper()
        df["source"] = source
        df["date"] = df["date"].map(lambda x: str(pd.Timestamp(x).date()))

        # Normalise case for DB columns (use lowercase)
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        db_df = pd.DataFrame()
        db_df["symbol"] = df["symbol"]
        db_df["date"] = df["date"]
        for cap, low in col_map.items():
            db_df[low] = df[cap] if cap in df.columns else 0
        db_df["source"] = df["source"]

        # Upsert within an explicit transaction for atomicity
        dates = [str(d) for d in db_df["date"].unique()]
        if not dates:
            return 0
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                f"DELETE FROM ohlcv_daily WHERE symbol = ? AND date IN ({','.join('?' * len(dates))})",
                [symbol.upper()] + dates,
            )
            db_df.to_sql(
                "ohlcv_daily", self.conn, if_exists="append", index=False,
                method="multi",
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(db_df)

    # ------------------------------------------------------------------
    # Signal history
    # ------------------------------------------------------------------

    def save_signal(
        self, scan_date: str, symbol: str, strategy: str,
        bar_date: str, signal: int, price: float, atr: float,
        indicators: str = "",
    ):
        self.init_schema()
        # Upsert — remove previous scan for this date+symbol+strategy
        self.conn.execute(
            "DELETE FROM signal_history WHERE scan_date = ? AND symbol = ? AND strategy = ?",
            [scan_date, symbol.upper(), strategy],
        )
        self.conn.execute(
            "INSERT INTO signal_history (scan_date, symbol, strategy, bar_date, signal, price, atr, indicators) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [scan_date, symbol.upper(), strategy, bar_date, signal, price, atr, indicators],
        )
        self.conn.commit()

    def query_signals(self, scan_date: str | None = None, symbol: str | None = None) -> list[dict]:
        self.init_schema()
        query = "SELECT * FROM signal_history WHERE 1=1"
        params = []
        if scan_date:
            query += " AND scan_date >= ?"
            params.append(scan_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY scan_date DESC, symbol ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(zip(["id", "scan_date", "symbol", "strategy", "bar_date",
                          "signal", "price", "atr", "indicators"], row)) for row in rows]

    # ------------------------------------------------------------------
    # Risk state persistence
    # ------------------------------------------------------------------

    def save_risk_state(self, key: str, value: str):
        self.init_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO risk_state (key, value) VALUES (?, ?)",
            [key, value],
        )
        self.conn.commit()

    def load_risk_state(self, key: str) -> Optional[str]:
        self.init_schema()
        row = self.conn.execute(
            "SELECT value FROM risk_state WHERE key = ?", [key]
        ).fetchone()
        return row[0] if row else None

    def save_entry_price(self, symbol: str, price: float, entry_date: str):
        self.init_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO entry_prices (symbol, price, entry_date) VALUES (?, ?, ?)",
            [symbol.upper(), price, entry_date],
        )
        self.conn.commit()

    def load_entry_price(self, symbol: str) -> Optional[tuple]:
        self.init_schema()
        row = self.conn.execute(
            "SELECT price, entry_date FROM entry_prices WHERE symbol = ?",
            [symbol.upper()],
        ).fetchone()
        return (row[0], row[1]) if row else None

    def load_all_entry_prices(self) -> dict:
        self.init_schema()
        rows = self.conn.execute(
            "SELECT symbol, price, entry_date FROM entry_prices"
        ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def delete_entry_price(self, symbol: str):
        self.init_schema()
        self.conn.execute(
            "DELETE FROM entry_prices WHERE symbol = ?", [symbol.upper()]
        )
        self.conn.commit()

    def save_trade_pnl(self, symbol: str, side: str, qty: int,
                       entry_price: float, exit_price: float,
                       exit_date: str, order_id: str):
        """Record a completed round-trip trade PnL."""
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price / entry_price - 1) * 100
        self.init_schema()
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_pnl ("
            "  symbol TEXT, side TEXT, qty INTEGER,"
            "  entry_price REAL, exit_price REAL,"
            "  pnl REAL, pnl_pct REAL, exit_date TEXT,"
            "  order_id TEXT PRIMARY KEY"
            ")"
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO trade_pnl VALUES (?,?,?,?,?,?,?,?,?)",
            [symbol.upper(), side, qty, entry_price, exit_price,
             round(pnl, 2), round(pnl_pct, 2), exit_date, order_id],
        )
        self.conn.commit()

    def query_trade_pnl(self, symbol: str = None, limit: int = 50) -> list[dict]:
        """Return recent trade PnL records."""
        self.init_schema()
        query = "SELECT * FROM trade_pnl"
        params = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY exit_date DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(zip(["symbol","side","qty","entry_price","exit_price",
                         "pnl","pnl_pct","exit_date","order_id"], row)) for row in rows]

    # ------------------------------------------------------------------
    # Incremental helpers
    # ------------------------------------------------------------------

    def missing_ranges(
        self, symbol: str, start: str, end: str
    ) -> List[Tuple[str, str]]:
        """Return list of (start_date, end_date) tuples that need fetching.

        Compares the requested interval against what's already cached.
        If the entire range is cached, returns an empty list.
        """
        cached_start, cached_end = self.date_range(symbol)

        req_start = str(pd.Timestamp(start).date())
        req_end = str(pd.Timestamp(end).date())

        if cached_start is None or cached_end is None:
            return [(req_start, req_end)]

        gaps = []
        if req_start < cached_start:
            gap_end = min(req_end, str(pd.Timestamp(cached_start).date()))
            if req_start <= gap_end:
                gaps.append((req_start, gap_end))

        if req_end > cached_end:
            gap_start = max(req_start, str(pd.Timestamp(cached_end).date()))
            if gap_start <= req_end:
                gaps.append((gap_start, req_end))

        return gaps
