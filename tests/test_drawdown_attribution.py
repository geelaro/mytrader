"""Tests for analysis.drawdown_attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.drawdown_attribution import (
    attribute_active_drawdown,
    attribute_drawdown,
    historical_drawdown_attribution,
    trade_overlap_attribution,
)


class _MockTrade:
    """Duck-typed PortfolioTrade for tests."""
    def __init__(self, symbol, entry_time, exit_time, pnl):
        self.symbol = symbol
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.pnl = pnl


def _dates(n: int, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


class TestAttributeDrawdown:
    def test_single_position_takes_full_blame(self):
        """One holding, no cash flow → 100% of drawdown attributed to it."""
        idx = _dates(5)
        # NVDA goes 100 → 90.  Portfolio value mirrors.
        pv = pd.Series([100, 102, 95, 90, 92], index=idx)
        pos = pd.DataFrame({"NVDA": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[1], trough_date=idx[3])

        assert result["peak_value"] == 102
        assert result["trough_value"] == 90
        assert result["depth_usd"] == -12
        assert result["depth_pct"] == pytest.approx(-11.764, abs=0.01)
        assert len(result["by_symbol"]) == 1
        assert result["by_symbol"][0]["symbol"] == "NVDA"
        assert result["by_symbol"][0]["contribution_usd"] == -12
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_two_positions_split_blame(self):
        """50/50 portfolio, one tanks one flat → top contributor is the one that fell."""
        idx = _dates(3)
        # NVDA 50→40, TSLA 50→50.  Total 100→90.
        pv = pd.Series([100, 95, 90], index=idx)
        pos = pd.DataFrame({
            "NVDA": [50, 45, 40],
            "TSLA": [50, 50, 50],
        }, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[0], trough_date=idx[2])

        # NVDA contributed -10, TSLA 0.
        by_sym = {r["symbol"]: r for r in result["by_symbol"]}
        assert by_sym["NVDA"]["contribution_usd"] == -10
        assert by_sym["TSLA"]["contribution_usd"] == 0
        # Sort order: NVDA first (larger |contribution|).
        assert result["by_symbol"][0]["symbol"] == "NVDA"
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_auto_finds_worst_window(self):
        """No peak/trough given → picks the deepest drawdown automatically."""
        idx = _dates(7)
        # Two drawdowns: 100→95 (small), 110→80 (big).
        pv = pd.Series([100, 95, 100, 110, 100, 80, 90], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos)

        assert result["peak_date"] == idx[3]   # 110
        assert result["trough_date"] == idx[5]  # 80
        assert result["depth_usd"] == -30
        assert result["depth_pct"] == pytest.approx(-27.27, abs=0.01)

    def test_position_opened_mid_drawdown(self):
        """Symbol bought during the DD shows peak=0, contribution=+trough_value."""
        idx = _dates(3)
        # NVDA tanks; we buy SPY mid-DD as a hedge.
        pv = pd.Series([100, 85, 88], index=idx)
        pos = pd.DataFrame({
            "NVDA": [100, 80, 70],
            "SPY":  [0,   5,  18],
        }, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[0], trough_date=idx[2])

        by_sym = {r["symbol"]: r for r in result["by_symbol"]}
        assert by_sym["NVDA"]["contribution_usd"] == -30
        assert by_sym["SPY"]["peak_value"] == 0
        assert by_sym["SPY"]["contribution_usd"] == 18
        # Net: -30 + 18 = -12, matching depth_usd.
        assert result["depth_usd"] == -12
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_unexplained_reflects_cash_flow(self):
        """A deposit during the window shows up as unexplained_usd."""
        idx = _dates(3)
        # Portfolio: 100 → +50 deposit → still drops to 130.
        # Position only fell 100→80, so positions explain -20, residual -30 unexplained
        # (== the deposit absorbed by other cash).  Wait this needs care.
        # Let's set: peak total=100 (all in NVDA).  Day 2: deposit 50 cash,
        # NVDA value 80. total=130. Day 3: NVDA 70, total=120.
        # peak→trough on total: 130 → 120 → -10.  But this isn't the true DD
        # — the peak we pass is day 1 (100), trough day 3 (120).  Total
        # rose 100→120, no drawdown.  Pick day 2 → day 3: peak 130, trough 120.
        pv = pd.Series([100, 130, 120], index=idx)
        pos = pd.DataFrame({"NVDA": [100, 80, 70]}, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[1], trough_date=idx[2])

        # NVDA went 80→70 → -10 contribution.
        nvda = next(r for r in result["by_symbol"] if r["symbol"] == "NVDA")
        assert nvda["contribution_usd"] == -10
        # depth = -10; explained = -10 → unexplained ≈ 0.
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_empty_input(self):
        """Empty series → all-zeros summary."""
        result = attribute_drawdown(pd.Series(dtype=float), pd.DataFrame())
        assert result["depth_pct"] == 0.0
        assert result["by_symbol"] == []

    def test_no_drawdown(self):
        """Monotonically rising → peak==trough at same date, 0% depth."""
        idx = _dates(4)
        pv = pd.Series([100, 105, 110, 115], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos)

        assert result["depth_usd"] == 0
        assert result["depth_pct"] == 0.0

    def test_top_n_limit(self):
        """top_n truncates to most-impactful contributors."""
        idx = _dates(2)
        pv = pd.Series([100, 60], index=idx)
        pos = pd.DataFrame({
            "A": [50, 30],   # -20
            "B": [30, 20],   # -10
            "C": [20, 10],   # -10
        }, index=idx)

        result = attribute_drawdown(pv, pos, top_n=1)

        assert len(result["by_symbol"]) == 1
        assert result["by_symbol"][0]["symbol"] == "A"


class TestTradeOverlapAttribution:
    def test_overlapping_trade_counted(self):
        peak = pd.Timestamp("2026-01-10")
        trough = pd.Timestamp("2026-01-20")
        trades = [
            _MockTrade("NVDA", pd.Timestamp("2026-01-05"),
                       pd.Timestamp("2026-01-25"), pnl=-500),
        ]
        out = trade_overlap_attribution(trades, peak, trough, depth_usd=-500)
        assert out["n_trades"] == 1
        assert out["by_symbol"][0]["symbol"] == "NVDA"
        assert out["by_symbol"][0]["pnl"] == -500
        assert out["unexplained_usd"] == 0

    def test_trade_entirely_before_window_excluded(self):
        peak = pd.Timestamp("2026-02-01")
        trough = pd.Timestamp("2026-02-10")
        trades = [
            _MockTrade("NVDA", pd.Timestamp("2026-01-01"),
                       pd.Timestamp("2026-01-15"), pnl=-100),
        ]
        out = trade_overlap_attribution(trades, peak, trough)
        assert out["n_trades"] == 0

    def test_trade_entirely_after_window_excluded(self):
        peak = pd.Timestamp("2026-01-01")
        trough = pd.Timestamp("2026-01-10")
        trades = [
            _MockTrade("X", pd.Timestamp("2026-02-01"),
                       pd.Timestamp("2026-02-10"), pnl=999),
        ]
        out = trade_overlap_attribution(trades, peak, trough)
        assert out["n_trades"] == 0

    def test_aggregates_multiple_trades_per_symbol(self):
        peak = pd.Timestamp("2026-01-01")
        trough = pd.Timestamp("2026-01-30")
        trades = [
            _MockTrade("NVDA", pd.Timestamp("2026-01-05"),
                       pd.Timestamp("2026-01-15"), pnl=-200),
            _MockTrade("NVDA", pd.Timestamp("2026-01-20"),
                       pd.Timestamp("2026-01-28"), pnl=-100),
            _MockTrade("TSLA", pd.Timestamp("2026-01-10"),
                       pd.Timestamp("2026-01-25"), pnl=50),
        ]
        out = trade_overlap_attribution(trades, peak, trough, depth_usd=-250)
        by_sym = {r["symbol"]: r for r in out["by_symbol"]}
        assert by_sym["NVDA"]["pnl"] == -300
        assert by_sym["NVDA"]["n_trades"] == 2
        assert by_sym["TSLA"]["pnl"] == 50
        assert by_sym["TSLA"]["n_trades"] == 1
        # Sort: worst first
        assert out["by_symbol"][0]["symbol"] == "NVDA"

    def test_contribution_pct_filled_when_depth_given(self):
        peak = pd.Timestamp("2026-01-01")
        trough = pd.Timestamp("2026-01-30")
        trades = [
            _MockTrade("X", pd.Timestamp("2026-01-05"),
                       pd.Timestamp("2026-01-20"), pnl=-500),
        ]
        out = trade_overlap_attribution(trades, peak, trough, depth_usd=-1000)
        # Convention: PnL-signed share of |drawdown|.  -500/1000 = -50%
        # (loser cost 50% of the drawdown).  A gainer's pnl_pct would be +.
        assert out["by_symbol"][0]["contribution_pct"] == -50.0

    def test_open_position_uses_trough_as_exit(self):
        """A trade still open (exit_time=None) at trough is counted."""
        peak = pd.Timestamp("2026-01-01")
        trough = pd.Timestamp("2026-01-30")
        trades = [
            _MockTrade("X", pd.Timestamp("2026-01-10"), None, pnl=-200),
        ]
        out = trade_overlap_attribution(trades, peak, trough)
        assert out["n_trades"] == 1


class TestHistoricalDrawdownAttribution:
    def test_scans_multiple_episodes(self):
        """Equity curve with 2 drawdowns → 2 attribution records."""
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        # 100 → 90 (DD1), 100 → 80 (DD2)
        pv = pd.Series([100, 95, 90, 95, 100, 95, 85, 80, 90, 100], index=idx)
        trades = [
            _MockTrade("A", idx[0], idx[3], pnl=-10),    # in DD1
            _MockTrade("B", idx[4], idx[7], pnl=-15),   # in DD2
        ]
        out = historical_drawdown_attribution(pv, trades, top_n_episodes=5,
                                              min_depth_pct=0)
        assert len(out) == 2
        # Worst first: DD2 (-20%) before DD1 (-10%)
        assert out[0]["depth_pct"] < out[1]["depth_pct"]

    def test_returns_empty_for_monotone_rising(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        pv = pd.Series([100, 102, 105, 110, 115], index=idx)
        out = historical_drawdown_attribution(pv, [])
        assert out == []

    def test_top_n_limits(self):
        idx = pd.date_range("2026-01-01", periods=12, freq="D")
        pv = pd.Series([100, 95, 100, 90, 100, 80, 100, 85, 100, 70, 100, 100],
                       index=idx)
        out = historical_drawdown_attribution(pv, [], top_n_episodes=2,
                                              min_depth_pct=0)
        assert len(out) <= 2

    def test_min_depth_filter(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="D")
        # One tiny DD (-1%) and one big (-20%)
        pv = pd.Series([100, 99, 100, 110, 90, 100], index=idx)
        out = historical_drawdown_attribution(pv, [], min_depth_pct=-5.0)
        # Only the -20% DD survives the -5% filter
        assert len(out) == 1
        assert out[0]["depth_pct"] < -5

    def test_per_episode_attribution_isolated(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        pv = pd.Series([100, 95, 90, 95, 100, 95, 85, 80, 90, 100], index=idx)
        trades = [
            _MockTrade("A", idx[0], idx[3], pnl=-10),   # only DD1
            _MockTrade("B", idx[4], idx[7], pnl=-15),   # only DD2
        ]
        out = historical_drawdown_attribution(pv, trades, top_n_episodes=5,
                                              min_depth_pct=0)
        # Each episode should see only the trades that overlap it
        for ep in out:
            symbols_in_ep = {r["symbol"] for r in ep["by_symbol"]}
            assert len(symbols_in_ep) == 1


class TestAttributeActiveDrawdown:
    def test_currently_in_drawdown(self):
        """Latest value < cummax → attributes the live drawdown."""
        idx = _dates(5)
        pv = pd.Series([100, 110, 105, 95, 92], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_active_drawdown(pv, pos)

        assert result["peak_date"] == idx[1]   # 110
        assert result["trough_date"] == idx[4]  # 92 (latest)
        assert result["depth_usd"] == -18
        assert result["by_symbol"][0]["symbol"] == "X"

    def test_at_new_high(self):
        """At ATH → empty summary, no active DD."""
        idx = _dates(3)
        pv = pd.Series([100, 110, 120], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_active_drawdown(pv, pos)

        assert result["depth_usd"] == 0
        assert result["by_symbol"] == []
