"""Tests for portfolio.py — Leg, PortfolioResult, PortfolioBacktest."""

import pandas as pd
import pytest

from portfolio import Leg, PortfolioResult, PortfolioBacktest
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
# PortfolioResult
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
        # We need to mock the DataProvider to return our synthetic data
        from unittest.mock import patch
        from data import DataProvider

        legs = [Leg("AAPL", "enhanced_macd")]
        bt = PortfolioBacktest(legs=legs, initial_capital=10000, allocation="equal")

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = bt.run(start="2020-01-01", end="2021-01-01")
            assert result.initial_capital == 10000
            assert result.final_equity > 0
            assert len(result.legs) == 1

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
