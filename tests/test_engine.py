"""Tests for trader.py — BacktestEngine and integration."""

import numpy as np
import pytest
from trader import BacktestEngine, BacktestResult, Trade


class TestBacktestEngine:
    def test_initial_state(self):
        engine = BacktestEngine(initial_capital=50000)
        assert engine.cash == 50000
        assert engine.position == 0
        assert engine.equity == 50000

    def test_buy_adds_position(self):
        engine = BacktestEngine(initial_capital=50000)
        date = pd.Timestamp("2025-01-15")
        ok = engine.buy(date, 100.0, 10)
        assert ok
        assert engine.position == 10
        assert engine.cash < 50000

    def test_buy_caps_at_cash_limit(self):
        """Buy should be rejected if even 1 share exceeds available cash."""
        engine = BacktestEngine(initial_capital=100)  # only $100
        ok = engine.buy(pd.Timestamp("2025-01-15"), 1000, 2)  # $2000 → rejected
        assert ok is False
        assert engine.position == 0

    def test_buy_rejected_when_price_zero(self):
        engine = BacktestEngine()
        assert engine.buy(pd.Timestamp("2025-01-15"), 0, 10) is False

    def test_sell_reduces_position(self):
        engine = BacktestEngine(initial_capital=50000)
        d1 = pd.Timestamp("2025-01-15")
        d2 = pd.Timestamp("2025-01-16")
        engine.buy(d1, 100.0, 10)
        trade = engine.sell(d2, 110.0, 10)
        assert engine.position == 0
        assert trade is not None
        assert trade.pnl > 0

    def test_partial_sell(self):
        engine = BacktestEngine(initial_capital=50000)
        d1 = pd.Timestamp("2025-01-15")
        d2 = pd.Timestamp("2025-01-16")
        engine.buy(d1, 100.0, 10)
        engine.sell(d2, 110.0, 5)
        assert engine.position == 5

    def test_update_tracks_equity(self):
        engine = BacktestEngine()
        engine.update(pd.Timestamp("2025-01-15"), 100)
        engine.update(pd.Timestamp("2025-01-16"), 105)
        assert len(engine.equity_history) == 2


class TestBacktestResult:
    def test_result_from_empty_engine_raises(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError):
            engine.get_result()

    def test_result_metrics_are_finite(self):
        engine = BacktestEngine(initial_capital=10000)
        d1, d2 = pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-16")
        engine.buy(d1, 100, 10)
        engine.update(d1, 100)
        engine.update(d2, 101)
        engine.sell(d2, 101, reason="test")
        engine.update(d2, 101)
        result = engine.get_result()
        assert result.initial_capital == 10000
        assert result.total_trades == 1
        # All float metrics should be finite (profit_factor can be inf with no losers)
        for attr in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                      "win_rate_pct"]:
            val = getattr(result, attr)
            assert np.isfinite(val), f"{attr} = {val}"
        assert result.profit_factor > 0  # inf or finite, but always positive

    def test_zero_trades_gives_neutral_metrics(self):
        engine = BacktestEngine(initial_capital=10000)
        engine.update(pd.Timestamp("2025-01-15"), 100)
        engine.update(pd.Timestamp("2025-01-16"), 100)
        result = engine.get_result()
        assert result.total_trades == 0
        assert result.win_rate_pct == 0
        assert result.profit_factor == 0


class TestTrade:
    def test_trade_holding_days(self):
        t = Trade(
            entry_date=pd.Timestamp("2025-01-01"),
            exit_date=pd.Timestamp("2025-01-11"),
            entry_price=100, exit_price=110,
            quantity=10, pnl=100, pnl_pct=10.0,
            exit_reason="signal",
        )
        assert t.holding_days == 10


# ===================================================================
# End-to-end: strategy + engine on synthetic data
# ===================================================================


def test_full_backtest_run(ohlcv):
    """Smoke test — run EnhancedMACD through BacktestEngine on synthetic data."""
    from strategy import EnhancedMACDStrategy

    strategy = EnhancedMACDStrategy()
    df = strategy.calculate_indicators(ohlcv)
    engine = BacktestEngine(initial_capital=10000)
    highest = 0

    for i in range(strategy.min_bars, len(df)):
        date_idx = df.index[i]
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])

        if engine.position > 0 and engine.current_entry:
            if price > highest:
                highest = price
            exit_now, reason = strategy.check_exit(
                df, i,
                entry_price=engine.current_entry["price"],
                highest_since_entry=highest,
            )
            if exit_now:
                engine.sell(date_idx, price, reason=reason)
        elif strategy.entry_signal(df, i) and engine.position == 0:
            qty = strategy.position_size(engine.cash, price, atr)
            if qty > 0:
                engine.buy(date_idx, price, qty)
                highest = price
        engine.update(date_idx, price)

    if engine.position > 0:
        last_price = float(df["Close"].iloc[-1])
        engine.sell(df.index[-1], last_price, reason="回测结束")
        engine.update(df.index[-1], last_price)

    result = engine.get_result(df["Close"].pct_change().dropna())
    assert result.initial_capital == 10000
    assert result.final_equity > 0
    assert result.total_trades >= 0
    # Sharpe should be finite
    assert np.isfinite(result.sharpe_ratio)


def test_all_four_strategies_run_without_error(ohlcv):
    """Every strategy should complete calculate_indicators + check_exit without exception."""
    from strategy import (
        EnhancedMACDStrategy,
        TrendFollower,
        WeeklyMACD,
        WeeklyMACD_KDJ,
    )
    strategies = [
        EnhancedMACDStrategy(),
        TrendFollower(),
        WeeklyMACD(),
        WeeklyMACD_KDJ(),
    ]
    for s in strategies:
        df = s.calculate_indicators(ohlcv)
        assert "Signal" in df.columns
        assert "ATR" in df.columns
        # check_exit should not raise
        s.check_exit(df, s.min_bars + 5, entry_price=100, highest_since_entry=105)


# ===================================================================
# print_result
# ===================================================================


class TestPrintResult:
    def test_prints_without_crash(self, capsys):
        from trader import print_result, BacktestResult
        import numpy as np

        result = BacktestResult(
            trades=[],
            equity_curve=pd.Series([10000, 10100, 10200],
                                   index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])),
            total_return_pct=2.0, cagr_pct=10.0, sharpe_ratio=1.5,
            max_drawdown_pct=-1.0, win_rate_pct=60.0, profit_factor=2.0,
            avg_win_pct=3.0, avg_loss_pct=-2.0,
            total_trades=5, winning_trades=3, losing_trades=2,
            buy_hold_return_pct=1.5,
            initial_capital=10000, final_equity=10200,
        )
        print_result(result)
        out = capsys.readouterr().out
        assert len(out) > 0


class TestPlotResult:
    def test_plot_saves_file(self, tmp_path):
        """plot_result should create a PNG file."""
        import matplotlib
        matplotlib.use("Agg")
        from trader import plot_result, BacktestResult, BacktestEngine
        import numpy as np

        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                "2025-01-06", "2025-01-07"])
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
            "Volume": [1_000_000] * 5,
        }, index=dates)

        result = BacktestResult(
            trades=[],
            equity_curve=pd.Series([10000, 10100, 10200, 10300, 10400], index=dates),
            total_return_pct=4.0, cagr_pct=20.0, sharpe_ratio=2.0,
            max_drawdown_pct=0.0, win_rate_pct=100.0, profit_factor=999,
            avg_win_pct=2.0, avg_loss_pct=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            buy_hold_return_pct=5.0,
            initial_capital=10000, final_equity=10400,
        )
        path = str(tmp_path / "test_plot.png")
        fig = plot_result(result, df, symbol="TEST", save_path=path)
        import os
        assert os.path.exists(path)
        import matplotlib.pyplot as plt
        plt.close(fig)


# Need pd for timestamp
import pandas as pd
