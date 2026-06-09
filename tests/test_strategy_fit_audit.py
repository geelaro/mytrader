"""Tests for scripts/strategy_fit_audit.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from strategy_fit_audit import (
    StrategyAudit,
    best_audit,
    _bh_metrics,
    format_per_symbol,
    format_summary,
    load_watchlist_active,
)


# -- StrategyAudit dataclass --------------------------------------------


class TestStrategyAudit:
    def test_delta_calculations(self):
        a = StrategyAudit(
            symbol="X", strategy="s1",
            cagr=10.0, sharpe=0.8, max_dd=-30.0, total_trades=10,
            bh_cagr=8.0, bh_sharpe=0.6, bh_max_dd=-40.0,
        )
        assert a.cagr_delta == pytest.approx(2.0)
        assert a.sharpe_delta == pytest.approx(0.2)
        # dd_delta: 策略 -30 比 B&H -40 浅 10pp → positive
        assert a.dd_delta == pytest.approx(10.0)


# -- _bh_metrics --------------------------------------------------------


class TestBhMetrics:
    def test_constant_price_zero_cagr(self):
        s = pd.Series([100.0] * 252, index=pd.date_range("2020-01-01", periods=252))
        cagr, sharpe, dd = _bh_metrics(s)
        assert abs(cagr) < 0.001
        # std=0 → sharpe should be 0 (fallback path)
        assert sharpe == 0
        assert dd == 0

    def test_monotone_up_positive_cagr_no_dd(self):
        s = pd.Series([100.0 + i for i in range(252)],
                       index=pd.date_range("2020-01-01", periods=252))
        cagr, sharpe, dd = _bh_metrics(s)
        assert cagr > 0
        assert dd == 0  # monotone up has no drawdown


# -- best_audit ---------------------------------------------------------


class TestBestAudit:
    def _make(self, strategy: str, sharpe: float, cagr: float):
        return StrategyAudit(
            symbol="X", strategy=strategy,
            cagr=cagr, sharpe=sharpe, max_dd=-30.0, total_trades=10,
            bh_cagr=8.0, bh_sharpe=0.6, bh_max_dd=-40.0,
        )

    def test_empty_returns_none(self):
        assert best_audit([]) is None

    def test_sharpe_picks_highest_sharpe(self):
        a = self._make("a", sharpe=0.5, cagr=20.0)
        b = self._make("b", sharpe=0.9, cagr=10.0)
        assert best_audit([a, b], sort_by="sharpe").strategy == "b"

    def test_cagr_picks_highest_cagr(self):
        a = self._make("a", sharpe=0.5, cagr=20.0)
        b = self._make("b", sharpe=0.9, cagr=10.0)
        assert best_audit([a, b], sort_by="cagr").strategy == "a"


# -- load_watchlist_active ---------------------------------------------


class TestLoadWatchlistActive:
    def test_missing_file(self, tmp_path):
        assert load_watchlist_active(str(tmp_path / "no.toml")) == {}

    def test_parses_active_strings(self, tmp_path):
        cfg = tmp_path / "wl.toml"
        cfg.write_text(
            '[[watchlist]]\nsymbol="AAPL"\nactive="weekly_macd"\n'
            '[[watchlist]]\nsymbol="MU"\nactive="weekly_macd_kdj"\n'
        )
        result = load_watchlist_active(str(cfg))
        assert result == {"AAPL": "weekly_macd", "MU": "weekly_macd_kdj"}

    def test_ignores_list_active(self, tmp_path):
        # ensemble mode (active = list) is skipped — only string active counted
        cfg = tmp_path / "wl.toml"
        cfg.write_text(
            '[[watchlist]]\nsymbol="X"\nactive=["a","b"]\n'
            '[[watchlist]]\nsymbol="Y"\nactive="s1"\n'
        )
        result = load_watchlist_active(str(cfg))
        assert result == {"Y": "s1"}


# -- format functions ---------------------------------------------------


class TestFormatPerSymbol:
    def test_empty_audits(self):
        out = format_per_symbol([], None)
        assert "no audits" in out.lower()

    def test_marks_active(self):
        a = StrategyAudit("X", "s1", 10, 0.8, -30, 10, 8, 0.6, -40)
        out = format_per_symbol([a], current_active="s1")
        assert "<- active" in out

    def test_no_marker_when_active_differs(self):
        a = StrategyAudit("X", "s1", 10, 0.8, -30, 10, 8, 0.6, -40)
        out = format_per_symbol([a], current_active="other")
        assert "<- active" not in out


class TestFormatSummary:
    def test_flags_change_when_recommended_differs(self):
        a1 = StrategyAudit("X", "weekly_macd", 10, 1.5, -30, 10, 8, 0.6, -40)
        a2 = StrategyAudit("X", "weekly_macd_kdj", 8, 0.7, -35, 12, 8, 0.6, -40)
        out = format_summary({"X": [a1, a2]}, {"X": "weekly_macd_kdj"}, "sharpe")
        assert "[CHANGE]" in out
        assert "weekly_macd" in out

    def test_flags_same_when_match(self):
        a = StrategyAudit("X", "s1", 10, 0.9, -30, 10, 8, 0.6, -40)
        out = format_summary({"X": [a]}, {"X": "s1"}, "sharpe")
        assert "[same]" in out
