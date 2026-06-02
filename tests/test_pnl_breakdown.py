"""Tests for analysis/pnl_breakdown.py — realized + unrealized aggregation."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from analysis.pnl_breakdown import (
    pnl_summary,
    realized_pnl_summary,
    resolve_period,
    unrealized_pnl_summary,
)


# ---------------------------------------------------------------------------
# Fake cache for realized tests
# ---------------------------------------------------------------------------


def _make_cache(rows):
    """Build a MagicMock cache with .query_trade_pnl returning ``rows``."""
    c = MagicMock()
    c.query_trade_pnl = MagicMock(return_value=rows)
    return c


def _trade(symbol, pnl, exit_date, **extra):
    """Build a trade_pnl row matching the SQL schema dict shape."""
    base = {
        "symbol": symbol,
        "side": "BUY",
        "qty": 10,
        "entry_price": 100.0,
        "exit_price": 100.0 + pnl / 10,
        "pnl": pnl,
        "pnl_pct": pnl / 1000 * 100,
        "exit_date": exit_date,
        "order_id": f"ord_{symbol}_{exit_date}",
    }
    base.update(extra)
    return base


# ===================================================================
# resolve_period
# ===================================================================


class TestResolvePeriod:
    def test_all_returns_none(self):
        assert resolve_period("all") == (None, None)
        assert resolve_period("") == (None, None)
        assert resolve_period(None) == (None, None)

    def test_named_periods(self):
        ref = datetime(2026, 6, 1, 12, 0, 0)
        since, until = resolve_period("30d", now=ref)
        assert until == "2026-06-01"
        assert since == "2026-05-02"

    def test_ytd(self):
        ref = datetime(2026, 6, 1, 12, 0, 0)
        since, until = resolve_period("ytd", now=ref)
        assert since == "2026-01-01"
        assert until == "2026-06-01"

    def test_unknown_token_returns_none(self):
        assert resolve_period("forever") == (None, None)


# ===================================================================
# realized_pnl_summary
# ===================================================================


class TestRealizedPnl:
    def test_empty_cache(self):
        s = realized_pnl_summary(_make_cache([]))
        assert s["total"] == 0
        assert s["n_trades"] == 0
        assert s["by_symbol"] == {}
        assert s["trades"] == []

    def test_aggregates_total(self):
        rows = [
            _trade("AAPL", 100, "2026-05-01"),
            _trade("AAPL", -30, "2026-05-15"),
            _trade("MSFT", 50, "2026-05-20"),
        ]
        s = realized_pnl_summary(_make_cache(rows))
        assert s["total"] == 120
        assert s["n_trades"] == 3
        assert s["n_wins"] == 2
        assert s["n_losses"] == 1
        assert s["win_rate_pct"] == pytest.approx(66.67, abs=0.01)

    def test_by_symbol_aggregation(self):
        rows = [
            _trade("AAPL", 100, "2026-05-01"),
            _trade("AAPL", -30, "2026-05-15"),
            _trade("MSFT", 50, "2026-05-20"),
        ]
        s = realized_pnl_summary(_make_cache(rows))
        assert s["by_symbol"]["AAPL"] == 70  # 100 - 30
        assert s["by_symbol"]["MSFT"] == 50
        # Sorted by absolute pnl descending: AAPL (70) > MSFT (50)
        assert list(s["by_symbol"].keys()) == ["AAPL", "MSFT"]

    def test_by_month_aggregation(self):
        rows = [
            _trade("AAPL", 100, "2026-05-01"),
            _trade("MSFT", 50, "2026-05-20"),
            _trade("JPM", 80, "2026-06-01"),
        ]
        s = realized_pnl_summary(_make_cache(rows))
        assert s["by_month"] == {"2026-05": 150, "2026-06": 80}

    def test_date_window_filter(self):
        rows = [
            _trade("AAPL", 100, "2026-04-01"),  # outside
            _trade("MSFT", 50, "2026-05-15"),   # inside
            _trade("JPM", 80, "2026-06-30"),    # outside
        ]
        s = realized_pnl_summary(_make_cache(rows),
                                 since="2026-05-01", until="2026-05-31")
        assert s["n_trades"] == 1
        assert s["total"] == 50
        assert "MSFT" in s["by_symbol"]
        assert "AAPL" not in s["by_symbol"]

    def test_avg_win_loss(self):
        rows = [
            _trade("AAPL", 100, "2026-05-01"),
            _trade("MSFT", 200, "2026-05-15"),
            _trade("JPM", -50, "2026-05-20"),
            _trade("GOOG", -100, "2026-05-25"),
        ]
        s = realized_pnl_summary(_make_cache(rows))
        assert s["avg_win"] == 150  # (100 + 200) / 2
        assert s["avg_loss"] == -75  # (-50 + -100) / 2

    def test_trades_sorted_newest_first(self):
        rows = [
            _trade("AAPL", 100, "2026-05-01"),
            _trade("MSFT", 50, "2026-05-20"),
            _trade("JPM", 30, "2026-05-10"),
        ]
        s = realized_pnl_summary(_make_cache(rows))
        dates = [t["exit_date"] for t in s["trades"]]
        assert dates == sorted(dates, reverse=True)


# ===================================================================
# unrealized_pnl_summary
# ===================================================================


class TestUnrealizedPnl:
    def test_empty(self):
        s = unrealized_pnl_summary([])
        assert s["total"] == 0
        assert s["n_positions"] == 0
        assert s["by_symbol"] == []

    def test_positive_unrealized(self):
        positions = [{
            "symbol": "AAPL", "entry_price": 100, "current_price": 110, "shares": 10,
        }]
        s = unrealized_pnl_summary(positions)
        assert s["total"] == 100  # (110 - 100) * 10
        assert s["n_positions"] == 1
        assert s["n_winning"] == 1
        assert s["n_losing"] == 0
        row = s["by_symbol"][0]
        assert row["pnl"] == 100
        assert row["pnl_pct"] == pytest.approx(10.0, abs=0.01)

    def test_mixed_positions(self):
        positions = [
            {"symbol": "AAPL", "entry_price": 100, "current_price": 110, "shares": 10},
            {"symbol": "JPM", "entry_price": 50, "current_price": 45, "shares": 20},
        ]
        s = unrealized_pnl_summary(positions)
        # AAPL: +100, JPM: (45-50)*20 = -100 → total 0
        assert s["total"] == 0
        assert s["n_winning"] == 1
        assert s["n_losing"] == 1

    def test_sorted_by_absolute_pnl(self):
        positions = [
            {"symbol": "A", "entry_price": 100, "current_price": 105, "shares": 1},   # +5
            {"symbol": "B", "entry_price": 100, "current_price": 50, "shares": 10},   # -500
            {"symbol": "C", "entry_price": 100, "current_price": 200, "shares": 5},   # +500
        ]
        s = unrealized_pnl_summary(positions)
        symbols = [r["symbol"] for r in s["by_symbol"]]
        # B (500) and C (500) before A (5).  Order between B/C may swap.
        assert symbols.index("A") == 2

    def test_invalid_positions_skipped(self):
        positions = [
            {"symbol": "A", "entry_price": 0, "current_price": 100, "shares": 10},   # entry=0
            {"symbol": "B", "entry_price": 100, "current_price": 0, "shares": 10},   # current=0
            {"symbol": "", "entry_price": 100, "current_price": 110, "shares": 10},  # no symbol
            {"symbol": "C", "entry_price": 100, "current_price": 110, "shares": 0},  # shares=0
            {"symbol": "D", "entry_price": 100, "current_price": 110, "shares": 5},  # valid
        ]
        s = unrealized_pnl_summary(positions)
        assert s["n_positions"] == 1
        assert s["by_symbol"][0]["symbol"] == "D"

    def test_strategy_passed_through(self):
        positions = [{
            "symbol": "AAPL", "entry_price": 100, "current_price": 110,
            "shares": 10, "strategy": "weekly_macd_kdj",
        }]
        s = unrealized_pnl_summary(positions)
        assert s["by_symbol"][0]["strategy"] == "weekly_macd_kdj"


# ===================================================================
# pnl_summary — end-to-end
# ===================================================================


class TestPnlSummary:
    def test_realized_plus_unrealized(self):
        cache = _make_cache([
            _trade("AAPL", 100, "2026-05-01"),
            _trade("MSFT", 50, "2026-05-20"),
        ])
        positions = [{
            "symbol": "TSLA", "entry_price": 200, "current_price": 220, "shares": 5,
        }]
        s = pnl_summary(cache, positions)
        # realized = 100 + 50 = 150
        # unrealized = (220 - 200) * 5 = 100
        # total = 250
        assert s["realized"]["total"] == 150
        assert s["unrealized"]["total"] == 100
        assert s["total"] == 250

    def test_period_filter_propagates(self):
        cache = _make_cache([
            _trade("AAPL", 100, "2026-04-01"),
            _trade("MSFT", 50, "2026-05-30"),
        ])
        ref = datetime(2026, 6, 1)
        s = pnl_summary(cache, [], period="30d", now=ref)
        # 30d before 2026-06-01 = 2026-05-02 → AAPL excluded, MSFT included
        assert s["realized"]["total"] == 50
        assert s["since"] == "2026-05-02"

    def test_no_data_returns_zero(self):
        s = pnl_summary(_make_cache([]), [])
        assert s["total"] == 0
        assert s["realized"]["n_trades"] == 0
        assert s["unrealized"]["n_positions"] == 0
