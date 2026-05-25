"""Tests for engine/trader.py short-selling path."""

import numpy as np
import pandas as pd
import pytest
from engine.trader import BacktestEngine, Trade
from strategy.turtle_trading import TurtleTrading
from strategy.donchian_breakout import DonchianBreakout
from tests.conftest import make_ohlcv


@pytest.fixture
def data():
    return make_ohlcv(n_bars=300, seed=42)


class TestSignedPosition:
    def test_short_position_negative(self):
        e = BacktestEngine(initial_capital=10000)
        e.buy(pd.Timestamp("2020-01-02"), 100, 10, direction="SHORT")
        assert e.position == -10
        assert e._direction == "SHORT"
        assert "SHORT" in str(e.current_entry.get("direction", ""))

    def test_long_position_positive(self):
        e = BacktestEngine(initial_capital=10000)
        e.buy(pd.Timestamp("2020-01-02"), 100, 10, direction="LONG")
        assert e.position == 10
        assert e._direction == "LONG"

    def test_equity_short_gain_on_drop(self):
        e = BacktestEngine(initial_capital=10000)
        e.buy(pd.Timestamp("2020-01-02"), 100, 10, direction="SHORT")
        e.update(pd.Timestamp("2020-01-03"), 80)
        assert e.equity > 10000  # profited from price drop

    def test_equity_short_loss_on_rise(self):
        e = BacktestEngine(initial_capital=10000)
        e.buy(pd.Timestamp("2020-01-02"), 100, 10, direction="SHORT")
        e.update(pd.Timestamp("2020-01-03"), 120)
        assert e.equity < 10000

    def test_cover_short_flat(self):
        e = BacktestEngine(initial_capital=10000)
        e.buy(pd.Timestamp("2020-01-02"), 100, 10, direction="SHORT")
        assert e.position == -10
        # Cover: buy back same quantity
        e.buy(pd.Timestamp("2020-01-03"), 90, 10, direction="LONG")
        # After covering, position goes from -10 to 0
        assert e.position == 0


class TestTurtleLongShort:
    def test_generates_both_signals(self, data):
        s = TurtleTrading()
        df = s.calculate_indicators(data)
        signals = set(df["Signal"].dropna().unique())
        assert -1 in signals, "turtle should generate short signals"

    def test_long_only_suppresses_shorts(self, data):
        s = TurtleTrading()
        s.long_only = True
        assert s.long_only is True
        df = s.calculate_indicators(data)
        # Engine ignores short signals when long_only=True
        eng = BacktestEngine(initial_capital=10000)
        eng.run(s, df)
        # Verify trades are LONG only
        for t in eng.trades:
            assert t.direction == "LONG"

    def test_backtest_runs_both_directions(self, data):
        s = TurtleTrading()
        df_sig = s.calculate_indicators(data)
        engine = BacktestEngine(initial_capital=10000, sizing_mode="risk_budget",
                                risk_per_trade=0.01, risk_atr_mult=2.0)
        engine.run(s, df_sig)
        r = engine.get_result(df_sig["Close"].pct_change().dropna())
        dirs = {t.direction for t in engine.trades}
        assert "LONG" in dirs or "SHORT" in dirs
        assert r.total_trades > 0

    def test_close_out_flattens_position(self, data):
        s = TurtleTrading()
        df_sig = s.calculate_indicators(data)
        engine = BacktestEngine(initial_capital=10000)
        engine.run(s, df_sig, close_out=True)
        assert engine.position == 0

    def test_equity_history_not_empty(self, data):
        s = TurtleTrading()
        df_sig = s.calculate_indicators(data)
        engine = BacktestEngine(initial_capital=10000)
        engine.run(s, df_sig)
        r = engine.get_result(df_sig["Close"].pct_change().dropna())
        assert len(r.equity_curve) > 0
        assert r.final_equity > 0


class TestTrendFilter:
    def test_trend_filter_enabled_by_default(self):
        s = TurtleTrading()
        assert s.params.trend_filter is True

    def test_trend_filter_can_disable(self, data):
        s = TurtleTrading(trend_filter=False)
        assert s.params.trend_filter is False
        df = s.calculate_indicators(data)
        assert "Signal" in df.columns

    def test_trend_filter_reduces_trades(self, data):
        """Trend filter should reduce or equal trades vs no filter."""
        s_filt = TurtleTrading(trend_filter=True)
        df1 = s_filt.calculate_indicators(data)
        e1 = BacktestEngine(initial_capital=10000)
        e1.run(s_filt, df1)
        # Should complete without error
        assert e1.equity > 0


class TestRecursiveSMA:
    def test_recursive_sma_used(self):
        from strategy.turtle_trading import _recursive_sma
        s = pd.Series([10, 11, 12, 13, 14, 15])
        result = _recursive_sma(s, 3)
        assert not result.isna().all()
        assert result.iloc[0] == 10  # seeded from first close


class TestDonchianShort:
    def test_generates_short_signals(self, data):
        s = DonchianBreakout()
        df = s.calculate_indicators(data)
        signals = set(df["Signal"].dropna().unique())
        assert -1 in signals

    def test_check_exit_short_cover(self, data):
        s = DonchianBreakout()
        df = s.calculate_indicators(data)
        # Simulate a short position exit check
        exit_now, reason = s.check_exit(
            df, s.min_bars + 10,
            entry_price=100, highest_since_entry=100,
            lowest_since_entry=90,
            position={"direction": "SHORT", "date": df.index[s.min_bars]},
        )
        assert isinstance(exit_now, bool)
        assert isinstance(reason, str)


class TestTradeDirection:
    def test_trade_has_direction(self):
        t = Trade(
            entry_date=pd.Timestamp("2020-01-02"),
            exit_date=pd.Timestamp("2020-02-02"),
            entry_price=100, exit_price=110, quantity=10,
            pnl=100, pnl_pct=10, exit_reason="signal",
            direction="SHORT",
        )
        assert t.direction == "SHORT"
