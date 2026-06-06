"""Tests for splits-adjustment uniformity across all US sources.

Regression test for the NVDA 2023-12-26 bug: Sina returned unadjusted
NVDA prices (~$492 pre-split) while Tencent returned adjusted (~$48),
producing a 10× price cliff at the source-boundary date in cache.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import data.sources as sources_mod
from data.sources import (
    SinaUSSource,
    TencentSource,
    YahooChartSource,
    _US_SPLITS,
    apply_us_splits,
)


def _mock_tencent_response(code: str, rows: list[list]) -> MagicMock:
    """Build a mock Tencent fqkline response for *code*.

    Each row in *rows* is [date_str, open, close, high, low, volume].
    """
    body = {"code": 0, "data": {code: {"day": rows}}}
    resp = MagicMock()
    resp.json.return_value = body
    return resp


# ===================================================================
# apply_us_splits — pure helper
# ===================================================================


class TestApplyUsSplits:
    def test_no_op_for_unconfigured_symbol(self):
        df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99],
            "Close": [100], "Volume": [1000],
        }, index=pd.to_datetime(["2020-01-01"]))
        result = apply_us_splits(df.copy(), "NOSUCH")
        pd.testing.assert_frame_equal(result, df)

    def test_no_op_for_empty_frame(self):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = apply_us_splits(empty, "NVDA")
        assert result.empty

    def test_pre_split_prices_divided(self):
        """NVDA 4:1 split on 2021-07-20 → bars before should be /4."""
        df = pd.DataFrame({
            "Open": [200.0, 200.0, 50.0],
            "High": [200.0, 200.0, 50.0],
            "Low": [200.0, 200.0, 50.0],
            "Close": [200.0, 200.0, 50.0],
            "Volume": [1000, 1000, 4000],
        }, index=pd.to_datetime([
            "2021-07-15",  # pre-split
            "2021-07-19",  # pre-split
            "2021-07-21",  # post-split (raw price already small)
        ]))
        result = apply_us_splits(df.copy(), "NVDA")
        # 2024-06-10 10:1 also applies to ALL of these (all are pre-2024)
        # So 2021-07-15: /4 (2021 split) then /10 (2024 split) = /40 → 5.0
        # 2021-07-19: same /40 → 5.0
        # 2021-07-21: only /10 (post-2021 split) → 5.0
        assert result.loc["2021-07-15", "Close"] == pytest.approx(5.0, abs=1e-9)
        assert result.loc["2021-07-19", "Close"] == pytest.approx(5.0, abs=1e-9)
        assert result.loc["2021-07-21", "Close"] == pytest.approx(5.0, abs=1e-9)

    def test_volume_multiplied(self):
        """Volume is multiplied by the inverse — shares outstanding doubled by split."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [100.0], "Low": [100.0],
            "Close": [100.0], "Volume": [1000],
        }, index=pd.to_datetime(["2020-01-01"]))
        # AAPL split 4:1 on 2020-08-31 — this bar is pre-split
        result = apply_us_splits(df.copy(), "AAPL")
        assert result.loc["2020-01-01", "Close"] == pytest.approx(25.0)
        assert result.loc["2020-01-01", "Volume"] == 4000

    def test_post_split_bars_untouched(self):
        """Bars after all splits should be unchanged."""
        df = pd.DataFrame({
            "Open": [150.0], "High": [150.0], "Low": [150.0],
            "Close": [150.0], "Volume": [1000],
        }, index=pd.to_datetime(["2025-01-01"]))  # after all NVDA splits
        result = apply_us_splits(df.copy(), "NVDA")
        assert result.loc["2025-01-01", "Close"] == 150.0
        assert result.loc["2025-01-01", "Volume"] == 1000

    def test_case_insensitive_symbol(self):
        df = pd.DataFrame({
            "Open": [200.0], "High": [200.0], "Low": [200.0],
            "Close": [200.0], "Volume": [1000],
        }, index=pd.to_datetime(["2020-01-01"]))
        # AAPL has 2020-08-31 4:1 split
        result_upper = apply_us_splits(df.copy(), "AAPL")
        result_lower = apply_us_splits(df.copy(), "aapl")
        pd.testing.assert_frame_equal(result_upper, result_lower)


# ===================================================================
# splits.json content — sanity
# ===================================================================


class TestSplitsConfig:
    def test_nvda_has_both_splits(self):
        """NVDA had two splits in the period covered: 2021-07-20 and 2024-06-10."""
        assert "NVDA" in _US_SPLITS
        dates = [d for d, _ in _US_SPLITS["NVDA"]]
        assert "2021-07-20" in dates
        assert "2024-06-10" in dates

    def test_all_ratios_positive_integers(self):
        for sym, entries in _US_SPLITS.items():
            for date_str, ratio in entries:
                assert isinstance(ratio, int), f"{sym} {date_str} ratio not int"
                assert ratio > 1, f"{sym} {date_str} ratio not > 1"


# ===================================================================
# SinaUSSource — regression: must apply splits
# ===================================================================


def _mock_sina_response(rows: list[dict]) -> MagicMock:
    """Build a mock requests response with Sina's JSONP wrapper."""
    body = json.dumps(rows)
    resp = MagicMock()
    resp.text = f"({body})"
    return resp


class TestSinaUSSourceAppliesSplits:
    def test_nvda_pre_split_adjusted(self):
        """Sina returns raw $492 for 2023-12-26 NVDA → must come out as $49.279."""
        raw_rows = [
            {"d": "2023-12-22", "o": 489, "h": 495, "l": 487, "c": 488.30, "v": 25000000},
            {"d": "2023-12-26", "o": 489.68, "h": 496.00, "l": 489.60, "c": 492.79, "v": 24419716},
            {"d": "2023-12-27", "o": 495.11, "h": 496.80, "l": 490.85, "c": 494.17, "v": 23364645},
        ]
        with patch("data.sources._make_session") as mock_make:
            session = MagicMock()
            session.get.return_value = _mock_sina_response(raw_rows)
            mock_make.return_value = session

            src = SinaUSSource()
            df = src.fetch("NVDA", "2023-12-22", "2023-12-27")

        # All bars are pre-2024-06-10 → divided by 10
        assert df.loc["2023-12-26", "Close"] == pytest.approx(49.279, abs=0.001)
        assert df.loc["2023-12-27", "Close"] == pytest.approx(49.417, abs=0.001)


class TestYahooChartSourceAppliesSplits:
    def test_nvda_pre_split_adjusted(self):
        """Yahoo chart v8 also returns raw prices in `quote` — must adjust."""
        timestamps = [
            int(pd.Timestamp("2023-12-26").timestamp()),
            int(pd.Timestamp("2023-12-27").timestamp()),
        ]
        mock_data = {
            "chart": {
                "result": [{
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{
                            "open": [489.68, 495.11],
                            "high": [496.00, 496.80],
                            "low": [489.60, 490.85],
                            "close": [492.79, 494.17],
                            "volume": [24419716, 23364645],
                        }],
                    },
                }],
            },
        }
        resp = MagicMock()
        resp.json.return_value = mock_data
        resp.raise_for_status = MagicMock()
        with patch("data.sources._yahoo_session") as mock_make:
            session = MagicMock()
            session.get.return_value = resp
            mock_make.return_value = session

            src = YahooChartSource()
            df = src.fetch("NVDA", "2023-12-25", "2023-12-28")

        # /10 for 2024-06-10 split
        assert df["Close"].iloc[0] == pytest.approx(49.279, abs=0.01)
        assert df["Close"].iloc[1] == pytest.approx(49.417, abs=0.01)


# ===================================================================
# Cross-source consistency (the actual bug)
# ===================================================================


class TestCrossSourceConsistency:
    """The NVDA 2023-12-26 bug: three sources MUST produce comparable
    prices when fed the same raw input."""

    def test_three_sources_same_scale(self):
        """Same raw price (492.79) through all three sources should yield
        the same adjusted output (~49.28) for an NVDA pre-2024-split bar."""
        # SinaUS
        with patch("data.sources._make_session") as mock_make:
            session = MagicMock()
            session.get.return_value = _mock_sina_response([
                {"d": "2023-12-26", "o": 489.68, "h": 496.00,
                 "l": 489.60, "c": 492.79, "v": 1},
            ])
            mock_make.return_value = session
            sina_df = SinaUSSource().fetch("NVDA", "2023-12-26", "2023-12-26")
        sina_close = sina_df["Close"].iloc[0]

        # YahooChart
        ts = int(pd.Timestamp("2023-12-26").timestamp())
        resp = MagicMock()
        resp.json.return_value = {
            "chart": {"result": [{
                "timestamp": [ts],
                "indicators": {"quote": [{
                    "open": [489.68], "high": [496.00],
                    "low": [489.60], "close": [492.79], "volume": [1],
                }]},
            }]},
        }
        resp.raise_for_status = MagicMock()
        with patch("data.sources._yahoo_session") as mock_make:
            session = MagicMock()
            session.get.return_value = resp
            mock_make.return_value = session
            yahoo_df = YahooChartSource().fetch("NVDA", "2023-12-26", "2023-12-26")
        yahoo_close = yahoo_df["Close"].iloc[0]

        # Two adjusted values from independent paths must match within float
        assert sina_close == pytest.approx(yahoo_close, abs=0.001)
        # And both ~= the expected /10 split adjustment
        assert sina_close == pytest.approx(49.279, abs=0.01)


# ===================================================================
# TencentSource — regression: must drop ET-today (mid-session snapshot)
# ===================================================================


class TestTencentDropsIntradayBar:
    """tencent's day endpoint live-updates the last bar during US market
    hours. The last row's OHLC is a partial-day snapshot, not EOD. Caching
    it pollutes the store until a manual force_refresh.
    """

    def test_drops_et_today_row(self, monkeypatch):
        """Mock ET-today = 2026-06-05; the 6/5 row must be dropped, 6/4 kept."""
        monkeypatch.setattr(
            sources_mod, "_et_today_naive",
            lambda: pd.Timestamp("2026-06-05"),
        )
        # row layout per tencent: [date, open, close, high, low, volume]
        # SPY has no splits, so apply_us_splits is a no-op.
        raw_rows = [
            ["2026-06-03", 758.15, 756.08, 758.80, 756.01, 3394229],
            ["2026-06-04", 752.10, 757.09, 758.31, 751.47, 49873840],
            # This is the mid-session snapshot — close 749.35 vs real EOD 737.55
            ["2026-06-05", 752.31, 749.35, 752.82, 748.23, 12349807],
        ]
        with patch("data.sources._make_session") as mock_make:
            session = MagicMock()
            session.get.return_value = _mock_tencent_response("usSPY.AM", raw_rows)
            mock_make.return_value = session

            df = TencentSource().fetch("SPY", "2026-06-03", "2026-06-05")

        # 6/5 row dropped, 6/3 and 6/4 retained
        assert pd.Timestamp("2026-06-05") not in df.index
        assert pd.Timestamp("2026-06-04") in df.index
        assert pd.Timestamp("2026-06-03") in df.index
        assert df.loc["2026-06-04", "Close"] == pytest.approx(757.09)

    def test_keeps_all_rows_when_today_is_future(self, monkeypatch):
        """When ET-today is after the fetched range, every row is EOD already."""
        monkeypatch.setattr(
            sources_mod, "_et_today_naive",
            lambda: pd.Timestamp("2026-06-08"),  # Monday after the fetched week
        )
        raw_rows = [
            ["2026-06-04", 752.10, 757.09, 758.31, 751.47, 49873840],
            ["2026-06-05", 752.31, 737.55, 752.82, 735.00, 49600000],
        ]
        with patch("data.sources._make_session") as mock_make:
            session = MagicMock()
            session.get.return_value = _mock_tencent_response("usSPY.AM", raw_rows)
            mock_make.return_value = session

            df = TencentSource().fetch("SPY", "2026-06-04", "2026-06-05")

        # Both rows kept — neither is ET-today
        assert pd.Timestamp("2026-06-05") in df.index
        assert pd.Timestamp("2026-06-04") in df.index
        assert df.loc["2026-06-05", "Close"] == pytest.approx(737.55)
