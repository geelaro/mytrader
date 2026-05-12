"""Tests for data/cache.py — CacheManager."""

import pytest
from data.cache import CacheManager


class TestCacheSchema:
    def test_init_creates_tables(self, temp_cache):
        temp_cache.init_schema()
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "ohlcv_daily" in names
        assert "signal_history" in names

    def test_init_is_idempotent(self, temp_cache):
        temp_cache.init_schema()
        temp_cache.init_schema()  # should not raise
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len([r for r in tables if r[0] == "ohlcv_daily"]) == 1


class TestSaveLoadRoundtrip:
    def test_save_and_load(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv, source="test")
        loaded = temp_cache.load("TEST")
        assert len(loaded) == len(ohlcv)
        assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
        # Values should match within float tolerance
        assert abs(loaded.iloc[0]["Close"] - ohlcv.iloc[0]["Close"]) < 0.01

    def test_load_with_date_filter(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv, source="test")
        start = ohlcv.index[50].strftime("%Y-%m-%d")
        end = ohlcv.index[99].strftime("%Y-%m-%d")
        subset = temp_cache.load("TEST", start=start, end=end)
        assert len(subset) == 50
        assert subset.index[0] >= pd.Timestamp(start)
        assert subset.index[-1] <= pd.Timestamp(end)

    def test_load_empty_symbol(self, temp_cache):
        df = temp_cache.load("NOEXIST")
        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_save_upsert(self, temp_cache, ohlcv):
        """Saving same symbol twice should not duplicate rows."""
        temp_cache.save("TEST", ohlcv.iloc[:100], source="a")
        temp_cache.save("TEST", ohlcv.iloc[:100], source="b")
        loaded = temp_cache.load("TEST")
        assert len(loaded) == 100  # not 200


class TestDateRange:
    def test_empty_cache(self, temp_cache):
        s, e = temp_cache.date_range("NOEXIST")
        assert s is None
        assert e is None

    def test_has_data(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv)
        s, e = temp_cache.date_range("TEST")
        assert s is not None
        assert e is not None


class TestMissingRanges:
    def test_all_missing(self, temp_cache):
        gaps = temp_cache.missing_ranges("X", "2020-01-01", "2020-06-01")
        assert len(gaps) == 1
        assert gaps[0] == ("2020-01-01", "2020-06-01")

    def test_partial_gap(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv)  # approx 2020-01-01 → 2021-03
        # Request earlier data
        gaps = temp_cache.missing_ranges("TEST", "2019-06-01", "2021-06-01")
        # Should have a gap before and after cached range
        assert len(gaps) >= 1


class TestSignalHistory:
    def test_save_signal(self, temp_cache):
        temp_cache.save_signal("2025-01-15", "AAPL", "weekly_macd",
                               "2025-01-15", 1, 195.0, 5.0, '{"MACD":1.5}')
        rows = temp_cache.query_signals()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["signal"] == 1

    def test_save_signal_upsert(self, temp_cache):
        """Second save for same scan_date+symbol+strategy replaces first."""
        temp_cache.save_signal("2025-01-15", "AAPL", "w", "2025-01-15", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-15", "AAPL", "w", "2025-01-15", -1, 2, 2, "")
        rows = temp_cache.query_signals()
        assert len(rows) == 1
        assert rows[0]["signal"] == -1

    def test_query_by_date(self, temp_cache):
        temp_cache.save_signal("2025-01-10", "A", "s", "2025-01-10", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-15", "B", "s", "2025-01-15", -1, 2, 2, "")
        rows = temp_cache.query_signals(scan_date="2025-01-13")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "B"

    def test_query_by_symbol(self, temp_cache):
        temp_cache.save_signal("2025-01-10", "AAPL", "s", "2025-01-10", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-10", "NVDA", "s", "2025-01-10", -1, 2, 2, "")
        rows = temp_cache.query_signals(symbol="AAPL")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"


# Need pd for asserts
import pandas as pd
