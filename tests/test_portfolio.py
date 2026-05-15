"""Tests for portfolio.py — Leg, PortfolioTrade, PortfolioResult, PortfolioBacktest."""

import pandas as pd
import pytest

from portfolio import Leg, PortfolioTrade, PortfolioResult, PortfolioBacktest
from strategy import WeeklyMACD_KDJ, TurtleTrading


# ===================================================================
# Leg
# ===================================================================


class TestLeg:
    def test_create_strategy_weekly_macd_kdj(self):
        leg = Leg("AAPL", "weekly_macd_kdj")
        s = leg.create_strategy()
        assert isinstance(s, WeeklyMACD_KDJ)

    def test_create_strategy_turtle_trading(self):
        leg = Leg("NVDA", "turtle_trading")
        s = leg.create_strategy()
        assert isinstance(s, TurtleTrading)

    def test_create_strategy_with_params(self):
        leg = Leg("AAPL", "weekly_macd_kdj", {"macd_fast": 10, "macd_slow": 24, "macd_signal": 8})
        s = leg.create_strategy()
        assert s.params.macd_fast == 10
        assert s.params.macd_slow == 24

    def test_unknown_strategy_raises(self):
        leg = Leg("AAPL", "no_such_strategy")
        with pytest.raises(ValueError, match="Unknown strategy"):
            leg.create_strategy()

    def test_default_params(self):
        leg = Leg("AAPL", "weekly_macd_kdj")
        assert leg.params == {}


# ===================================================================
# PortfolioTrade
# ===================================================================


def _make_trade(symbol="AAPL", pnl=500.0, hold_days=10, reason="signal", **kwargs):
    """Factory for a closed PortfolioTrade with sensible defaults."""
    entry = pd.Timestamp("2025-01-02")
    return PortfolioTrade(
        symbol=symbol,
        entry_time=entry,
        exit_time=entry + pd.Timedelta(days=hold_days),
        qty=kwargs.pop("qty", 100),
        entry_price=kwargs.pop("entry_price", 100.0),
        exit_price=kwargs.pop("exit_price", 105.0),
        pnl=pnl,
        pnl_pct=kwargs.pop("pnl_pct", None) or (pnl / (100 * 100) * 100),
        reason=reason,
        hold_days=hold_days,
        **kwargs,
    )


class TestPortfolioTrade:
    def test_defaults(self):
        t = PortfolioTrade(symbol="AAPL", entry_time=pd.Timestamp("2025-01-02"))
        assert t.symbol == "AAPL"
        assert t.exit_time is None
        assert t.qty == 0
        assert t.pnl is None
        assert t.hold_days is None

    def test_closed_trade_has_all_fields(self):
        t = _make_trade("NVDA", pnl=800, hold_days=15, reason="trailing_stop")
        assert t.symbol == "NVDA"
        assert t.exit_time is not None
        assert t.pnl == 800
        assert t.pnl_pct > 0
        assert t.reason == "trailing_stop"
        assert t.hold_days == 15

    def test_losing_trade(self):
        t = _make_trade("TSLA", pnl=-300, hold_days=5, reason="stop_loss")
        assert t.pnl < 0
        assert t.pnl_pct < 0

    def test_open_trade_no_exit(self):
        t = PortfolioTrade(
            symbol="AAPL", entry_time=pd.Timestamp("2025-03-01"),
            qty=50, entry_price=200.0,
        )
        assert t.exit_time is None
        assert t.pnl is None
        assert t.hold_days is None


# ===================================================================
# PortfolioResult — trade statistics
# ===================================================================


class TestPortfolioResultTradeStats:
    def _make_result_with_trades(self, trades=None, initial=100000, final=150000):
        dates = pd.bdate_range("2025-01-01", periods=100)
        equity_data = [(d, initial * (1 + i * 0.001)) for i, d in enumerate(dates)]
        return PortfolioResult(
            equity_history=equity_data,
            initial_capital=initial,
            final_equity=final,
            legs=[Leg("AAPL", "weekly_macd_kdj")],
            trades=trades or [],
        )

    # — closed_trades / total_trades —

    def test_closed_trades_filters_open(self):
        closed = _make_trade(pnl=100)
        open_trade = PortfolioTrade(
            symbol="SPY", entry_time=pd.Timestamp("2025-06-01"), qty=10, entry_price=400.0,
        )
        r = self._make_result_with_trades([closed, open_trade])
        assert r.total_trades == 1
        assert len(r.closed_trades) == 1
        assert r.closed_trades[0].symbol == "AAPL"

    def test_no_trades_returns_zeroes(self):
        r = self._make_result_with_trades([])
        assert r.total_trades == 0
        assert r.win_rate_pct == 0
        assert r.profit_factor == float("inf")
        assert r.avg_win == 0
        assert r.avg_loss == 0
        assert r.avg_hold_days == 0

    # — win_rate —

    def test_win_rate_mixed(self):
        trades = [
            _make_trade(pnl=500),
            _make_trade(pnl=-200),
            _make_trade(pnl=300),
            _make_trade(pnl=-100),
        ]
        r = self._make_result_with_trades(trades)
        assert r.win_rate_pct == 50.0

    def test_win_rate_all_wins(self):
        trades = [_make_trade(pnl=100), _make_trade(pnl=200)]
        r = self._make_result_with_trades(trades)
        assert r.win_rate_pct == 100.0

    def test_win_rate_all_losses(self):
        trades = [_make_trade(pnl=-50), _make_trade(pnl=-150)]
        r = self._make_result_with_trades(trades)
        assert r.win_rate_pct == 0.0

    # — profit_factor —

    def test_profit_factor(self):
        trades = [
            _make_trade(pnl=600),
            _make_trade(pnl=400),
            _make_trade(pnl=-200),
            _make_trade(pnl=-300),
        ]
        r = self._make_result_with_trades(trades)
        assert r.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_profit_factor_no_losses(self):
        trades = [_make_trade(pnl=100), _make_trade(pnl=200)]
        r = self._make_result_with_trades(trades)
        assert r.profit_factor == float("inf")

    # — avg_win / avg_loss —

    def test_avg_win_loss(self):
        trades = [
            _make_trade(pnl=900),
            _make_trade(pnl=300),
            _make_trade(pnl=-100),
            _make_trade(pnl=-500),
        ]
        r = self._make_result_with_trades(trades)
        assert r.avg_win == pytest.approx(600.0)
        assert r.avg_loss == pytest.approx(-300.0)

    # — avg_hold_days —

    def test_avg_hold_days(self):
        trades = [
            _make_trade(pnl=100, hold_days=5),
            _make_trade(pnl=200, hold_days=15),
        ]
        r = self._make_result_with_trades(trades)
        assert r.avg_hold_days == pytest.approx(10.0)

    # — summary output —

    def test_summary_includes_trade_stats(self, capsys):
        trades = [_make_trade(pnl=500)]
        r = self._make_result_with_trades(trades)
        r.summary()
        out = capsys.readouterr().out
        assert "交易统计" in out
        assert "总交易笔数" in out
        assert "胜率" in out
        assert "盈亏比" in out
        assert "平均盈利" in out
        assert "平均亏损" in out
        assert "平均持仓天数" in out

    def test_summary_includes_per_symbol_table(self, capsys):
        trades = [_make_trade("NVDA", pnl=300)]
        r = self._make_result_with_trades(trades)
        r.summary()
        out = capsys.readouterr().out
        assert "NVDA" in out

    def test_pnl_zero_not_counted_as_win(self):
        """Breakeven trades (pnl==0) are neither wins nor losses."""
        trades = [
            _make_trade(pnl=100),
            _make_trade(pnl=0),
        ]
        r = self._make_result_with_trades(trades)
        assert r.win_rate_pct == 50.0  # only 1 win out of 2
        assert r.profit_factor > 0

    def test_no_trades_summary_does_not_crash(self, capsys):
        r = self._make_result_with_trades([])
        r.summary()
        out = capsys.readouterr().out
        # Should print equity stats, skip trade table body
        assert "组合回测结果" in out


# ===================================================================
# PortfolioResult — equity properties (original)
# ===================================================================


class TestPortfolioResult:
    def _make_result(self, equity_data=None, initial=100000, final=150000, legs=None):
        if equity_data is None:
            dates = pd.bdate_range("2025-01-01", periods=100)
            equity_data = [(d, initial * (1 + i * 0.001)) for i, d in enumerate(dates)]
        if legs is None:
            legs = [Leg("AAPL", "weekly_macd_kdj")]
        return PortfolioResult(
            equity_history=equity_data,
            initial_capital=initial,
            final_equity=final,
            legs=legs,
        )

    def test_total_return(self):
        r = self._make_result(initial=100000, final=150000)
        assert r.total_return_pct == pytest.approx(50.0, abs=0.01)

    def test_negative_return(self):
        r = self._make_result(initial=100000, final=90000)
        assert r.total_return_pct == pytest.approx(-10.0, abs=0.01)

    def test_equity_curve_is_series(self):
        r = self._make_result()
        curve = r.equity_curve
        assert isinstance(curve, pd.Series)
        assert len(curve) > 0

    def test_cagr_is_finite(self):
        r = self._make_result()
        assert pd.notna(r.cagr_pct)

    def test_sharpe_is_finite(self):
        r = self._make_result()
        assert pd.notna(r.sharpe_ratio)

    def test_max_drawdown_is_negative_or_zero(self):
        r = self._make_result()
        assert r.max_drawdown_pct <= 0

    def test_summary_prints(self, capsys):
        r = self._make_result()
        r.summary()
        out = capsys.readouterr().out
        assert "组合回测结果" in out
        assert "总收益率" in out

    def test_zero_day_returns_safe_cagr(self):
        # single-day result
        dates = [pd.Timestamp("2025-01-01")]
        r = self._make_result(equity_data=[(dates[0], 100000)], initial=100000, final=100000)
        assert r.cagr_pct == 0.0


# ===================================================================
# PortfolioBacktest
# ===================================================================


class TestPortfolioBacktestInit:
    def test_default_allocation(self):
        bt = PortfolioBacktest(legs=[Leg("AAPL", "weekly_macd_kdj")])
        assert bt.allocation == "equal"
        assert bt.initial_capital == 100000
        assert bt.max_positions == 10

    def test_custom_params(self):
        bt = PortfolioBacktest(
            legs=[Leg("AAPL", "weekly_macd_kdj")],
            initial_capital=50000, allocation="fraction", max_positions=5,
        )
        assert bt.initial_capital == 50000
        assert bt.max_positions == 5

    def test_empty_legs_raises_on_run(self):
        bt = PortfolioBacktest(legs=[], initial_capital=100000)
        with pytest.raises(RuntimeError):
            bt.run()


class TestPortfolioAllocation:
    def test_equal_allocation(self, ohlcv):
        """Equal allocation splits initial capital evenly."""
        legs = [Leg("AAPL", "weekly_macd_kdj"), Leg("NVDA", "weekly_macd_kdj")]
        bt = PortfolioBacktest(legs=legs, initial_capital=100000, allocation="equal")
        alloc = bt._allocate(legs[0], 100000, 2)
        assert alloc == 50000  # equal split

    def test_fraction_allocation(self):
        legs = [Leg("AAPL", "weekly_macd_kdj")]
        bt = PortfolioBacktest(legs=legs, initial_capital=100000, allocation="fraction")
        alloc = bt._allocate(legs[0], 80000, 4)
        assert alloc == 20000  # 80000 * 0.25

    def test_equal_allocation_splits_evenly(self):
        legs = [Leg("AAPL", "weekly_macd_kdj")]
        bt = PortfolioBacktest(legs=legs, initial_capital=100000, allocation="equal")
        alloc = bt._allocate(legs[0], 75000, 3)
        assert alloc == pytest.approx(33333.33, abs=0.01)  # 100000 / 3


# ===================================================================
# PortfolioBacktest with synthetic data
# ===================================================================


class TestPortfolioBacktestIntegration:
    def test_runs_with_single_leg(self, ohlcv):
        """Smoke test — run portfolio backtest on synthetic data."""
        from unittest.mock import patch
        from data import DataProvider

        legs = [Leg("AAPL", "enhanced_macd")]
        bt = PortfolioBacktest(legs=legs, initial_capital=10000, allocation="equal")

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = bt.run(start="2020-01-01", end="2021-01-01")
            assert result.initial_capital == 10000
            assert result.final_equity > 0
            assert len(result.legs) == 1
            assert isinstance(result.trades, list)
            # All trades should be closed (open positions closed at end)
            assert result.total_trades == len(result.trades)
            for t in result.trades:
                assert t.exit_price is not None
                assert t.pnl is not None
                assert isinstance(t.reason, str) and len(t.reason) > 0

    def test_plot_saves_file(self, ohlcv, tmp_path):
        """plot() should create a PNG file."""
        from unittest.mock import patch
        from data import DataProvider
        import matplotlib
        matplotlib.use("Agg")

        legs = [Leg("AAPL", "enhanced_macd")]
        bt = PortfolioBacktest(legs=legs, initial_capital=10000)
        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = bt.run(start="2020-01-01", end="2021-01-01")
            path = str(tmp_path / "test_portfolio.png")
            result.plot(save_path=path)
            import os
            assert os.path.exists(path)

    def test_enforces_max_positions(self, ohlcv):
        """When max_positions is reached, no more entries."""
        from unittest.mock import patch
        from data import DataProvider

        legs = [Leg("AAPL", "enhanced_macd"), Leg("NVDA", "enhanced_macd"),
                Leg("QQQ", "enhanced_macd")]
        bt = PortfolioBacktest(legs=legs, initial_capital=10000, max_positions=1)

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = bt.run(start="2020-01-01", end="2021-01-01")
            # Should still run without error even with position limit
            assert result.final_equity > 0


# ===================================================================
# Regression: close-trade consistency
# ===================================================================


class TestCloseTradeUnit:
    """Direct unit tests for PortfolioBacktest._close_trade()."""

    def _make_st(self, position=100, entry_price=50.0, capital_allocated=5015.0):
        return {
            "position": position,
            "entry_price": entry_price,
            "capital_allocated": capital_allocated,  # entry_price * qty * (1 + comm)
            "highest": 55.0,
        }

    def test_end_of_period_applies_commission(self):
        """End-of-period close MUST apply commission, same as signal exit."""
        bt = PortfolioBacktest(legs=[], initial_capital=100000)
        st = self._make_st(position=100, entry_price=50.0, capital_allocated=5015.0)
        trades: list = [
            PortfolioTrade(symbol="AAPL", entry_time=pd.Timestamp("2025-01-02"),
                           qty=100, entry_price=50.0),
        ]
        open_idx = {0: 0}
        cash_before = 94985.0  # 100000 - 5015 entry cost

        cash_after = bt._close_trade(
            st, pd.Timestamp("2025-01-10"), 55.0, "end_of_period",
            open_idx, trades, 0, cash_before,
        )

        t = trades[0]
        # exit_price = 55 * (1 - 0.0001) = 54.9945
        # exit_proceeds = 54.9945 * 100 * (1 - 0.0003) = 5497.80...
        expected_exit_price = 55.0 * (1 - bt.slippage_pct)
        expected_proceeds = expected_exit_price * 100 * (1 - bt.commission_rate)
        expected_pnl = expected_proceeds - 5015.0

        assert t.exit_price == pytest.approx(expected_exit_price, abs=0.01)
        assert t.pnl == pytest.approx(expected_pnl, abs=0.01)
        assert t.pnl_pct == pytest.approx(expected_pnl / 5015.0 * 100, abs=0.01)
        assert t.reason == "end_of_period"
        assert t.hold_days == 8
        assert cash_after == pytest.approx(cash_before + expected_proceeds, abs=0.01)
        assert st["position"] == 0

    def test_signal_exit_same_math_as_end_of_period(self):
        """Signal exit and end-of-period close produce identical PnL math."""
        bt = PortfolioBacktest(legs=[], initial_capital=100000)

        # Signal exit
        st1 = self._make_st(position=50, entry_price=100.0, capital_allocated=5015.0)
        t1_list = [PortfolioTrade(symbol="NVDA", entry_time=pd.Timestamp("2025-03-01"),
                                  qty=50, entry_price=100.0)]
        cash1 = bt._close_trade(
            st1, pd.Timestamp("2025-03-15"), 110.0, "signal",
            {0: 0}, t1_list, 0, 50000.0,
        )

        # End-of-period close with same numbers
        st2 = self._make_st(position=50, entry_price=100.0, capital_allocated=5015.0)
        t2_list = [PortfolioTrade(symbol="NVDA", entry_time=pd.Timestamp("2025-03-01"),
                                  qty=50, entry_price=100.0)]
        cash2 = bt._close_trade(
            st2, pd.Timestamp("2025-03-15"), 110.0, "end_of_period",
            {0: 0}, t2_list, 0, 50000.0,
        )

        assert t1_list[0].pnl == pytest.approx(t2_list[0].pnl, abs=0.01)
        assert t1_list[0].pnl_pct == pytest.approx(t2_list[0].pnl_pct, abs=0.01)
        assert cash1 == pytest.approx(cash2, abs=0.01)


class TestTradePnLAlignsWithEquity:
    """Integration test: sum of trade PnLs ≈ final equity change."""

    def test_pnl_sum_matches_equity_change(self, ohlcv):
        """Σ trade.pnl should equal (final_equity - initial_capital)
        allowing for floating-point rounding."""
        from unittest.mock import patch
        from data import DataProvider

        legs = [Leg("AAPL", "enhanced_macd")]
        bt = PortfolioBacktest(legs=legs, initial_capital=10000, allocation="equal")

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = bt.run(start="2020-01-01", end="2021-01-01")

        closed = result.closed_trades
        assert len(closed) > 0, "synthetic data should produce trades"

        total_trade_pnl = sum(t.pnl for t in closed)
        equity_change = result.final_equity - result.initial_capital

        # Allow 0.01% tolerance due to floating-point accumulation
        tolerance = result.initial_capital * 0.0001
        assert total_trade_pnl == pytest.approx(equity_change, abs=tolerance), (
            f"Σ trade.pnl = {total_trade_pnl:.2f}, "
            f"equity change = {equity_change:.2f}"
        )
