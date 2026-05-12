"""Tests for data/provider.py and data/sources.py — routing, validation, classify."""

import pandas as pd
import pytest

from data.protocol import (
    DataSource,
    OHLCV_COLUMNS,
    classify_symbol,
    CN_SYMBOLS,
    SOURCE_PRIORITY,
)
from data.sources import YFinanceSource, TencentSource, SinaSource, AKShareSource
from data.provider import DataProvider
from data.cache import CacheManager


# ===================================================================
# classify_symbol
# ===================================================================


class TestClassifySymbol:
    def test_us_stock(self):
        assert classify_symbol("AAPL") == "us"
        assert classify_symbol("NVDA") == "us"

    def test_cn_digit(self):
        assert classify_symbol("510300") == "cn"
        assert classify_symbol("159919") == "cn"

    def test_cn_prefix(self):
        assert classify_symbol("sh510300") == "cn"
        assert classify_symbol("SZ159915") == "cn"

    def test_cn_alias(self):
        assert classify_symbol("510050") == "cn"

    def test_global(self):
        assert classify_symbol("0700.HK") == "global"


# ===================================================================
# SOURCE_PRIORITY
# ===================================================================


class TestSourcePriority:
    def test_us_priority(self):
        assert SOURCE_PRIORITY["us"][0] == "tencent"
        assert SOURCE_PRIORITY["us"][-1] == "yfinance"

    def test_cn_priority(self):
        assert SOURCE_PRIORITY["cn"][0] == "sina"
        assert "tencent" in SOURCE_PRIORITY["cn"]


# ===================================================================
# DataSource.validate
# ===================================================================


class TestDataSourceValidate:
    def test_normalises_columns(self):
        df = pd.DataFrame({
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [1000, 2000],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
        result = DataSource.validate(df, "TEST")
        assert list(result.columns) == OHLCV_COLUMNS

    def test_drops_nan_prices(self):
        df = pd.DataFrame({
            "Open": [100, None], "High": [102, 103], "Low": [99, 100],
            "Close": [None, 102], "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
        result = DataSource.validate(df, "TEST")
        assert len(result) <= 1

    def test_empty_input(self):
        df = pd.DataFrame(columns=OHLCV_COLUMNS)
        result = DataSource.validate(df, "TEST")
        assert result.empty

    def test_sorts_index(self):
        df = pd.DataFrame({
            "Open": [101, 100], "High": [103, 102], "Low": [100, 99],
            "Close": [102, 101], "Volume": [2000, 1000],
        }, index=pd.to_datetime(["2025-01-02", "2025-01-01"]))
        result = DataSource.validate(df, "TEST")
        assert result.index[0] < result.index[1]


# ===================================================================
# Source supports
# ===================================================================


class TestSourceSupports:
    def test_tencent_knows_us(self):
        src = TencentSource()
        assert src.supports("AAPL")
        assert src.supports("QQQ")
        assert not src.supports("510300")

    def test_sina_knows_cn(self):
        src = SinaSource()
        assert src.supports("510300")
        assert src.supports("sh510050")
        assert not src.supports("AAPL")

    def test_yfinance_rejects_cn(self):
        src = YFinanceSource()
        assert not src.supports("510300")
        assert not src.supports("SH510050")

    def test_akshare_knows_cn(self):
        src = AKShareSource()
        assert src.supports("510300")
        assert src.supports("sh510050")


# ===================================================================
# SinaSource symbol normalisation
# ===================================================================


class TestSinaNormalise:
    def test_digit_to_sh(self):
        assert SinaSource._normalise_symbol("510300") == "sh510300"

    def test_already_prefixed(self):
        assert SinaSource._normalise_symbol("sh510050") == "sh510050"

    def test_shenzhen(self):
        assert SinaSource._normalise_symbol("000001") == "sz000001"


# ===================================================================
# DataProvider with mock source
# ===================================================================


class _MockSource(DataSource):
    """A controlled data source for testing provider routing."""

    def __init__(self, name="yfinance", supports_all=True, data=None):
        self._name = name
        self._supports_all = supports_all
        self._data = data or self._make_data()

    @property
    def name(self) -> str:
        return self._name  # matches SOURCE_PRIORITY names for routing

    def supports(self, symbol: str) -> bool:
        return self._supports_all

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self._data

    @staticmethod
    def _make_data() -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame({
            "Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1_000_000,
        }, index=dates)
        df.index.name = "date"
        return df


class TestDataProvider:
    def test_routes_to_mock_source(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        df = provider.get_daily("ANY", start="2025-01-01", end="2025-01-15")
        assert len(df) > 0
        assert "Close" in df.columns

    def test_cache_hit_avoids_refetch(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])

        # First call — should fetch and cache
        df1 = provider.get_daily("TEST", start="2025-01-01", end="2025-01-20")
        assert len(df1) > 0

        # Second call with same range — should hit cache
        mock._supports_all = False  # if provider tries to refetch, this fails
        df2 = provider.get_daily("TEST", start="2025-01-01", end="2025-01-20")
        assert len(df2) >= len(df1)  # cache returns what was stored

    def test_force_refresh(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        df = provider.get_daily("TEST", start="2025-01-01", end="2025-01-10", force_refresh=True)
        assert len(df) > 0

    def test_cached_range(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        provider.get_daily("TEST", start="2025-01-01", end="2025-01-15")
        start, end = provider.cached_range("TEST")
        assert start is not None

    def test_list_sources(self):
        provider = DataProvider()
        sources = provider.list_sources("AAPL")
        assert len(sources) >= 1
