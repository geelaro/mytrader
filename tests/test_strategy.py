"""Tests for strategy/ module — BaseStrategy contract + all 4 strategies."""

import numpy as np
import pandas as pd
import pytest

from strategy import (
    BaseStrategy,
    StrategyParams,
    EnhancedMACDStrategy,
    EnhancedMACDParams,
    TrendFollower,
    TrendFollowerParams,
    WeeklyMACD,
    WeeklyMACDParams,
    WeeklyMACD_KDJ,
    WeeklyMACDKDJParams,
)
from strategy.base import compute_atr, compute_macd, compute_kdj, resample_weekly


# ===================================================================
# Base utilities
# ===================================================================


class TestComputeHelpers:
    def test_atr_is_not_null(self, ohlcv):
        atr = compute_atr(ohlcv, 14)
        assert len(atr) == len(ohlcv)
        assert atr.iloc[-1] > 0

    def test_macd_adds_columns(self, ohlcv):
        df = compute_macd(ohlcv.copy(), 12, 26, 9)
        for col in ["MACD", "MACD_signal", "MACD_hist"]:
            assert col in df.columns

    def test_kdj_adds_columns(self, ohlcv):
        df = compute_kdj(ohlcv.copy(), 9, 3, 3)
        for col in ["K", "D", "J"]:
            assert col in df.columns

    def test_resample_weekly(self, ohlcv):
        w = resample_weekly(ohlcv)
        assert len(w) < len(ohlcv)
        assert "Open" in w.columns


# ===================================================================
# StrategyParams
# ===================================================================


class TestParams:
    def test_default_values(self):
        p = EnhancedMACDParams()
        assert p.short_ma == 20
        assert p.long_ma == 50

    def test_override(self):
        p = EnhancedMACDParams(short_ma=15, long_ma=45)
        assert p.short_ma == 15

    def test_validation_rejects_bad_params(self):
        with pytest.raises(AssertionError):
            EnhancedMACDParams(short_ma=50, long_ma=20)  # short > long

    def test_immutable(self):
        p = WeeklyMACDParams(macd_fast=8)
        with pytest.raises(Exception):
            p.macd_fast = 10


# ===================================================================
# BaseStrategy contract
# ===================================================================


class TestBaseStrategyContract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_min_bars_required(self):
        s = EnhancedMACDStrategy()
        assert s.min_bars > 0

    def test_position_size_default(self, ohlcv):
        s = WeeklyMACD()
        qty = s.position_size(10000, 200, 10)
        assert qty > 0


# ===================================================================
# EnhancedMACDStrategy
# ===================================================================


class TestEnhancedMACD:
    def test_output_has_signal_column(self, ohlcv):
        s = EnhancedMACDStrategy()
        df = s.calculate_indicators(ohlcv)
        assert "Signal" in df.columns
        assert df["Signal"].isin([-1, 0, 1]).all()

    def test_output_has_atr(self, ohlcv):
        s = EnhancedMACDStrategy()
        df = s.calculate_indicators(ohlcv)
        assert "ATR" in df.columns
        assert df["ATR"].iloc[-1] > 0

    def test_entry_signal_method(self, ohlcv):
        s = EnhancedMACDStrategy()
        df = s.calculate_indicators(ohlcv)
        # Should not throw — entry_signal reads from Signal column
        for i in range(s.min_bars, len(df)):
            result = s.entry_signal(df, i)
            assert isinstance(result, bool)

    def test_check_exit_returns_tuple(self, ohlcv):
        s = EnhancedMACDStrategy()
        df = s.calculate_indicators(ohlcv)
        exit_now, reason = s.check_exit(df, s.min_bars, 100, 105)
        assert isinstance(exit_now, bool)
        assert isinstance(reason, str)

    def test_stop_loss_triggers_exit(self, ohlcv):
        s = EnhancedMACDStrategy(atr_stop_mult=2.0)
        df = s.calculate_indicators(ohlcv)
        # Find a bar where Signal == -1 (natural exit) or simulate stop-loss
        # Stop-loss: entry_price - 2 * ATR < current price → exit
        exit_now, reason = s.check_exit(df, s.min_bars + 10, entry_price=200, highest_since_entry=205)
        # If current price is much lower than entry, stop-loss should trigger
        if exit_now:
            assert reason in ("止损", "止盈", "卖出信号")

    def test_position_size_returns_zero_for_bad_inputs(self):
        s = EnhancedMACDStrategy()
        assert s.position_size(10000, 0, 10) == 0
        assert s.position_size(10000, 100, 0) == 0
        assert s.position_size(10000, 100, float("nan")) == 0


# ===================================================================
# TrendFollower
# ===================================================================


class TestTrendFollower:
    def test_output_has_adx(self, ohlcv):
        s = TrendFollower()
        df = s.calculate_indicators(ohlcv)
        assert "ADX" in df.columns
        assert "+DI" in df.columns
        assert "-DI" in df.columns

    def test_only_entry_signals(self, ohlcv):
        """TrendFollower does NOT emit sell signals — exit is via Chandelier."""
        s = TrendFollower()
        df = s.calculate_indicators(ohlcv)
        assert set(df["Signal"].unique()).issubset({0, 1})

    def test_chandelier_exit_triggers(self, ohlcv):
        s = TrendFollower()
        df = s.calculate_indicators(ohlcv)
        # If price is well below highest_since_entry − 3*ATR, exit should trigger
        atr = float(df["ATR"].iloc[s.min_bars + 5])
        exit_now, reason = s.check_exit(
            df, s.min_bars + 5,
            entry_price=100, highest_since_entry=200,
        )
        # May or may not trigger depending on data — just verify signature
        assert isinstance(exit_now, bool)
        if exit_now:
            assert reason == "移动止损"


# ===================================================================
# WeeklyMACD
# ===================================================================


class TestWeeklyMACD:
    def test_resampled_output(self, ohlcv):
        s = WeeklyMACD()
        df = s.calculate_indicators(ohlcv)
        # Weekly output should have fewer rows than daily input
        assert len(df) < len(ohlcv)
        assert "Signal" in df.columns

    def test_has_both_signals(self, ohlcv):
        s = WeeklyMACD()
        df = s.calculate_indicators(ohlcv)
        assert 1 in df["Signal"].values or -1 in df["Signal"].values

    def test_default_exit_uses_signal(self, ohlcv):
        s = WeeklyMACD()
        df = s.calculate_indicators(ohlcv)
        # Find a bar with signal == -1
        sell_bars = df[df["Signal"] == -1]
        if len(sell_bars) > 0:
            idx = df.index.get_loc(sell_bars.index[0])
            exit_now, reason = s.check_exit(df, idx, 100, 100)
            assert exit_now
            assert reason == "卖出信号"


# ===================================================================
# WeeklyMACD_KDJ
# ===================================================================


class TestWeeklyMACDKDJ:
    def test_output_has_kdj_columns(self, ohlcv):
        s = WeeklyMACD_KDJ()
        df = s.calculate_indicators(ohlcv)
        for col in ["K", "D", "MACD", "MACD_signal"]:
            assert col in df.columns

    def test_entry_is_kdj_golden(self, ohlcv):
        s = WeeklyMACD_KDJ()
        df = s.calculate_indicators(ohlcv)
        buy_bars = df[df["Signal"] == 1]
        if len(buy_bars) > 0:
            row = buy_bars.iloc[0]
            assert row["K"] > row["D"]  # Golden cross condition
