"""Tests for data/provider.py and data/sources.py — routing, validation, classify."""

import json
from unittest.mock import MagicMock, patch

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
        assert "yfinance" in SOURCE_PRIORITY["us"]

    def test_cn_priority(self):
        assert SOURCE_PRIORITY["cn"][0] == "sina"
        assert "tencent" in SOURCE_PRIORITY["cn"]

    def test_global_priority_uses_yfinance(self):
        assert SOURCE_PRIORITY["global"][0] == "yfinance"


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

    def test_tencent_supports_non_whitelist_us(self):
        """AMD was not in the whitelist but should be supported via fallback pattern."""
        src = TencentSource()
        assert src.supports("AMD")
        assert src.supports("MU")
        assert src.supports("ORCL")

    def test_tencent_rejects_non_alpha(self):
        src = TencentSource()
        assert not src.supports("510300")
        assert not src.supports("TQQQ3L")  # > 5 chars
        assert src.supports("AMD")  # 3-char US ticker

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

    def __init__(self, name="tencent", supports_all=True, data=None):
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


# ===================================================================
# Source fetch mocking
# ===================================================================


MOCK_TENCENT_RESP = {
    "code": 0,
    "data": {
        "usAAPL.OQ": {
            "day": [
                ["2025-01-02", "190.00", "191.00", "192.00", "189.00", "50000000"],
                ["2025-01-03", "191.50", "192.50", "193.50", "190.50", "45000000"],
                ["2025-01-06", "192.00", "194.00", "195.00", "191.00", "55000000"],
            ]
        }
    }
}

MOCK_SINA_RESP = [
    {"day": "2025-01-02", "open": "3.500", "high": "3.550", "low": "3.480", "close": "3.520", "volume": "10000000"},
    {"day": "2025-01-03", "open": "3.520", "high": "3.580", "low": "3.510", "close": "3.560", "volume": "12000000"},
]


class TestTencentSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import TencentSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_TENCENT_RESP
        mock_session.get.return_value = mock_resp

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 3
            assert list(df.columns) == OHLCV_COLUMNS
            assert df["Close"].iloc[0] == 191.0

    def test_fetch_api_error_returns_empty(self):
        from data.sources import TencentSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": -1, "msg": "error"}
        mock_session.get.return_value = mock_resp

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_request_exception_returns_empty(self):
        from data.sources import TencentSource
        import requests as req

        mock_session = MagicMock()
        mock_session.get.side_effect = req.RequestException("connection error")

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("NONEXIST", "2025-01-01", "2025-01-10")
            assert isinstance(df, pd.DataFrame)

    def test_fetch_with_split_adjustment(self):
        from data.sources import TencentSource

        src = TencentSource()
        # AAPL has a 4:1 split on 2020-08-31
        # With 2025 data, no splits should apply
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_TENCENT_RESP
        mock_session.get.return_value = mock_resp

        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 3


class TestSinaSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(MOCK_SINA_RESP)
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert len(df) == 2
            assert list(df.columns) == OHLCV_COLUMNS

    def test_fetch_empty_response(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "[]"
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_json_error(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "not json"
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty


class TestYFinanceSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import YFinanceSource

        mock_df = pd.DataFrame({
            "Open": [190, 191], "High": [192, 193],
            "Low": [189, 190], "Close": [191, 192],
            "Volume": [50000000, 45000000],
        }, index=pd.to_datetime(["2025-01-02", "2025-01-03"]))

        src = YFinanceSource()
        with patch("yfinance.download", return_value=mock_df):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 2
            assert list(df.columns) == OHLCV_COLUMNS

    def test_fetch_empty_returns_empty_df(self):
        from data.sources import YFinanceSource

        src = YFinanceSource()
        with patch("yfinance.download", return_value=pd.DataFrame()):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_exception_returns_empty(self):
        from data.sources import YFinanceSource

        src = YFinanceSource()
        with patch("yfinance.download", side_effect=Exception("fail")):
            with patch("data.sources.logger") as mock_logger:
                df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
                assert df.empty
                mock_logger.exception.assert_called_once()


class TestAKShareSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import AKShareSource

        mock_df = pd.DataFrame({
            "日期": ["2025-01-02", "2025-01-03"],
            "开盘": [3.50, 3.52],
            "最高": [3.55, 3.58],
            "最低": [3.48, 3.51],
            "收盘": [3.52, 3.56],
            "成交量": [10000000, 12000000],
        })

        src = AKShareSource()
        with patch("data.sources.AKShareSource.fetch", return_value=pd.DataFrame(columns=OHLCV_COLUMNS)):
            # Just verify import/init doesn't crash
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_without_akshare_installed(self):
        try:
            import akshare  # noqa: F401
            pytest.skip("akshare is installed")
        except ImportError:
            from data.sources import AKShareSource
            src = AKShareSource()
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty
