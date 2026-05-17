"""Tests for utils/sectors.py."""

import pytest
from utils.sectors import get_sector, DEFAULT_SECTORS


class TestGetSector:
    def test_known_tech(self):
        assert get_sector("AAPL") == "Technology"
        assert get_sector("NVDA") == "Technology"
        assert get_sector("GOOGL") == "Technology"

    def test_automotive(self):
        assert get_sector("TSLA") == "Automotive"

    def test_etf(self):
        assert get_sector("SPY") == "Broad Market ETF"
        assert get_sector("QQQ") == "Broad Market ETF"

    def test_china(self):
        assert get_sector("510300") == "China Equity"

    def test_unknown(self):
        assert get_sector("MYSTERY") == "Unknown"

    def test_case_insensitive(self):
        assert get_sector("aapl") == "Technology"
        assert get_sector("Spy") == "Broad Market ETF"

    def test_map_is_non_empty(self):
        assert len(DEFAULT_SECTORS) > 20
