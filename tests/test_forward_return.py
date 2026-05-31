"""Tests for analysis/forward_return.py."""

import numpy as np
import pandas as pd
import pytest

from analysis.forward_return import (
    compute_forward_returns,
    summarise,
    HorizonStats,
    ForwardReturnStats,
)


def _make_frame(closes, signals, start="2020-01-01"):
    """Build a minimal frame with Signal + Close columns indexed by business days."""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "Close": closes,
        "Signal": signals,
    }, index=dates)


# ===================================================================
# Basic correctness
# ===================================================================


class TestComputeForwardReturns:
    def test_no_signals(self):
        df = _make_frame([100] * 50, [0] * 50)
        fr = compute_forward_returns(df, horizons=[30], direction=1)
        assert fr.empty

    def test_single_buy_signal_simple_ramp(self):
        # Price ramps from 100 → 130 over 30 bars; signal at bar 0
        closes = list(np.linspace(100, 130, 60))
        signals = [1] + [0] * 59
        df = _make_frame(closes, signals)
        fr = compute_forward_returns(df, horizons=[30], direction=1)
        assert len(fr) == 1
        # Close[30] / Close[0] - 1 ≈ +30%/2 since ramp covers 60 bars
        actual = fr.iloc[0, 0]
        expected = closes[30] / closes[0] - 1
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_signal_too_close_to_end_returns_nan(self):
        # 50 bars, signal at bar 40, asking 90-day horizon → NaN
        closes = list(np.linspace(100, 150, 50))
        signals = [0] * 40 + [1] + [0] * 9
        df = _make_frame(closes, signals)
        fr = compute_forward_returns(df, horizons=[90], direction=1)
        assert len(fr) == 1
        assert pd.isna(fr.iloc[0, 0])

    def test_multiple_horizons(self):
        closes = list(np.linspace(100, 200, 200))
        signals = [0] * 200
        signals[10] = 1
        signals[50] = 1
        df = _make_frame(closes, signals)
        fr = compute_forward_returns(df, horizons=[30, 90], direction=1)
        assert set(fr.columns) == {30, 90}
        assert len(fr) == 2

    def test_short_signal_inverts_return(self):
        # Price drops 20% → short forward return should be +20%
        closes = [100] * 5 + list(np.linspace(100, 80, 30)) + [80] * 25
        signals = [0] * 5 + [-1] + [0] * 54
        df = _make_frame(closes, signals)
        fr = compute_forward_returns(df, horizons=[30], direction=-1)
        # Entry at idx 5 (price 100), exit at idx 35 (price ~80)
        actual = fr.iloc[0, 0]
        assert actual > 0
        # Approximately +20% (price dropped, inverted)
        assert actual == pytest.approx(0.20, abs=0.02)

    def test_missing_signal_column_raises(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(KeyError, match="Signal"):
            compute_forward_returns(df, direction=1)

    def test_missing_price_column_raises(self):
        df = pd.DataFrame({"Signal": [0, 1, 0]})
        with pytest.raises(KeyError, match="Close"):
            compute_forward_returns(df, direction=1)


# ===================================================================
# Summarise statistics
# ===================================================================


class TestSummarise:
    def test_empty_frame(self):
        df = pd.DataFrame(columns=[30, 90])
        stats = summarise(df, direction=1)
        assert stats.n_signals == 0
        assert stats.horizons == []
        assert "无信号" in stats.verdict

    def test_summary_fields_populated(self):
        # 100 random signals, each with random returns in [-0.1, +0.15]
        rng = np.random.default_rng(0)
        n = 100
        fr = pd.DataFrame({
            30: rng.uniform(-0.1, 0.15, n),
            90: rng.uniform(-0.1, 0.15, n),
        }, index=pd.bdate_range("2020-01-01", periods=n))
        stats = summarise(fr)
        assert stats.n_signals == 100
        assert len(stats.horizons) == 2
        h30 = stats.by_horizon(30)
        assert h30.n == 100
        assert -0.15 <= h30.median <= 0.20
        assert h30.std > 0
        assert 0 <= h30.win_rate <= 1

    def test_handles_nan_only_column(self):
        # All values NaN → horizon stats should still be computed but n=0,
        # and the horizon dropped from output
        fr = pd.DataFrame({30: [np.nan, np.nan, np.nan]},
                         index=pd.bdate_range("2020-01-01", periods=3))
        stats = summarise(fr)
        assert stats.n_signals == 3
        assert len(stats.horizons) == 0  # nothing valid to report


# ===================================================================
# Verdict logic
# ===================================================================


class TestVerdict:
    def _stats(self, h, n, win_rate, sharpe, median):
        return ForwardReturnStats(
            direction="long",
            n_signals=n,
            horizons=[HorizonStats(
                horizon=h, n=n, median=median, mean=median, std=0.05,
                win_rate=win_rate, sharpe=sharpe,
                min=-0.1, max=0.2, p25=-0.02, p75=0.08,
            )],
        )

    def test_real_signal(self):
        v = self._stats(90, 50, win_rate=0.62, sharpe=0.8, median=0.04).verdict
        assert "信号有效" in v

    def test_marginal(self):
        v = self._stats(90, 50, win_rate=0.58, sharpe=0.3, median=0.02).verdict
        assert "边际" in v

    def test_no_predictive_power(self):
        v = self._stats(90, 50, win_rate=0.51, sharpe=0.05, median=0.001).verdict
        # median < 1% AND win_rate < 0.55 → "no predictive power"
        assert "无预测力" in v or "弱信号" in v

    def test_small_sample(self):
        v = self._stats(90, 5, win_rate=0.8, sharpe=2.0, median=0.05).verdict
        assert "样本不足" in v


# ===================================================================
# End-to-end with a simple strategy-like signal pattern
# ===================================================================


class TestEndToEnd:
    def test_perfect_signal_high_sharpe(self):
        """Signals always fire right before a +10% move → high sharpe."""
        rng = np.random.default_rng(42)
        n = 600
        # Random walk with mild drift
        log_ret = rng.normal(0.0001, 0.01, n)
        closes = 100 * np.exp(np.cumsum(log_ret))
        # Inject 20 "perfect" buy signals: pick bars where next-90-day
        # return happened to be > 5%
        signals = np.zeros(n, dtype=int)
        for i in range(20, n - 100):
            future_ret = closes[i + 60] / closes[i] - 1
            if future_ret > 0.05 and signals[max(0, i-10):i].sum() == 0:
                signals[i] = 1
        df = _make_frame(closes.tolist(), signals.tolist())
        fr = compute_forward_returns(df, horizons=[30, 60, 90], direction=1)
        stats = summarise(fr)
        h60 = stats.by_horizon(60)
        assert h60 is not None
        # By construction, win rate should be very high at 60d
        assert h60.win_rate > 0.95
        assert h60.median > 0.05
