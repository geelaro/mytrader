"""Tests for utils/metrics.py — drawdown_stats + exposure_from_trades."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from utils.metrics import drawdown_stats, exposure_from_trades


# ---------------------------------------------------------------------------
# drawdown_stats
# ---------------------------------------------------------------------------


class TestDrawdownStats:
    def test_flat_curve_zero_drawdown(self):
        curve = pd.Series([100.0, 100.0, 100.0, 100.0], index=pd.date_range("2020-01-01", periods=4))
        current_dd, max_dd, longest = drawdown_stats(curve)
        assert current_dd == 0.0
        assert max_dd == 0.0
        assert longest == 0

    def test_declining_curve(self):
        curve = pd.Series([100.0, 95.0, 90.0, 85.0], index=pd.date_range("2020-01-01", periods=4))
        current_dd, max_dd, longest = drawdown_stats(curve)
        assert current_dd < 0
        assert max_dd < 0
        assert longest == 3  # 3 bars below peak

    def test_recovery_curve(self):
        curve = pd.Series([100.0, 90.0, 95.0, 100.0, 105.0], index=pd.date_range("2020-01-01", periods=5))
        current_dd, max_dd, longest = drawdown_stats(curve)
        assert current_dd == 0.0  # at new high
        assert max_dd < 0  # max dd was at bar 1
        assert longest == 2  # bars 1-2 under water

    def test_single_point(self):
        curve = pd.Series([100.0], index=pd.date_range("2020-01-01", periods=1))
        current_dd, max_dd, longest = drawdown_stats(curve)
        assert current_dd == 0.0
        assert max_dd == 0.0
        assert longest == 0

    def test_v_shape(self):
        curve = pd.Series([100.0, 80.0, 70.0, 80.0, 100.0, 110.0],
                          index=pd.date_range("2020-01-01", periods=6))
        current_dd, max_dd, longest = drawdown_stats(curve)
        assert current_dd == 0.0
        assert max_dd < 0
        assert longest == 3  # bars 1-2-3 under original peak (bar 4 is at new high)


# ---------------------------------------------------------------------------
# exposure_from_trades
# ---------------------------------------------------------------------------


def _make_result(closed_trades):
    result = MagicMock()
    result.closed_trades = closed_trades
    return result


class TestExposureFromTrades:
    def test_empty_trades_returns_empty(self):
        result = _make_result([])
        curve = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert len(net_pct) == 0
        assert last_exp == 0.0
        assert top3 == {}

    def test_no_trades_none_closed(self):
        result = _make_result([])
        curve = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert last_exp == 0.0

    def test_single_trade(self):
        trade = MagicMock()
        trade.symbol = "AAPL"
        trade.entry_price = 100.0
        trade.qty = 10
        trade.pnl = 50.0
        trade.entry_time = pd.Timestamp("2020-01-02")
        trade.exit_time = pd.Timestamp("2020-01-05")
        result = _make_result([trade])
        curve = pd.Series([100_000.0] * 6, index=pd.date_range("2020-01-01", periods=6, freq="D"))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert len(net_pct) > 0
        assert last_exp >= 0
        assert "AAPL" in top3

    def test_overlapping_trades(self):
        trade1 = MagicMock()
        trade1.symbol = "AAPL"
        trade1.entry_price = 100.0
        trade1.qty = 10
        trade1.pnl = 50.0
        trade1.entry_time = pd.Timestamp("2020-01-02")
        trade1.exit_time = pd.Timestamp("2020-01-05")

        trade2 = MagicMock()
        trade2.symbol = "NVDA"
        trade2.entry_price = 200.0
        trade2.qty = 5
        trade2.pnl = 100.0
        trade2.entry_time = pd.Timestamp("2020-01-03")
        trade2.exit_time = pd.Timestamp("2020-01-06")

        result = _make_result([trade1, trade2])
        curve = pd.Series([100_000.0] * 7, index=pd.date_range("2020-01-01", periods=7, freq="D"))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert len(net_pct) > 0
        assert last_exp >= 0

    def test_short_curve_returns_empty(self):
        trade = MagicMock()
        trade.symbol = "AAPL"
        trade.entry_price = 100.0
        trade.qty = 10
        trade.pnl = 50.0
        trade.entry_time = pd.Timestamp("2020-01-01")
        trade.exit_time = pd.Timestamp("2020-01-10")
        result = _make_result([trade])
        curve = pd.Series([100_000.0], index=pd.date_range("2020-01-01", periods=1))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert len(net_pct) == 0
        assert last_exp == 0.0

    def test_multiple_symbols_top3(self):
        trades = []
        for sym, entry, qty, pnl in [
            ("AAPL", 100.0, 10, 100.0),
            ("NVDA", 200.0, 5, 200.0),
            ("TSLA", 150.0, 8, 300.0),
            ("QQQ", 300.0, 3, 50.0),
        ]:
            t = MagicMock()
            t.symbol = sym
            t.entry_price = entry
            t.qty = qty
            t.pnl = pnl
            t.entry_time = pd.Timestamp("2020-01-02")
            t.exit_time = pd.Timestamp("2020-01-10")
            trades.append(t)

        result = _make_result(trades)
        curve = pd.Series([100_000.0] * 10, index=pd.date_range("2020-01-01", periods=10, freq="D"))
        net_pct, last_exp, top3 = exposure_from_trades(result, curve)
        assert len(top3) > 0
        assert len(top3) <= 3
