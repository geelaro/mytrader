"""Tests for utils/market_state.py — MarketStateClassifier and regime filtering."""

import numpy as np
import pandas as pd
import pytest

from utils.market_state import (
    MarketStateClassifier, MarketRegime, Volatility, MarketState,
    is_trend_strategy, is_mean_reversion_strategy,
)


def make_trend_up(n_bars=500) -> pd.DataFrame:
    """Strong uptrend with low noise."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    drift = 0.001
    noise = rng.normal(0, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(np.full(n_bars, drift) + noise))
    data = []
    for i, c in enumerate(close):
        r = c * abs(rng.normal(0.015, 0.005))
        data.append({
            "Date": dates[i], "Open": c - r / 2, "High": c + r,
            "Low": c - r, "Close": c, "Volume": 10_000_000,
        })
    return pd.DataFrame(data).set_index("Date")


def make_ranging(n_bars=500) -> pd.DataFrame:
    """Sideways / ranging market."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    noise = rng.normal(0, 0.012, n_bars)
    close = 100 * np.exp(np.cumsum(noise * 0.3))  # low drift
    data = []
    for i, c in enumerate(close):
        r = c * abs(rng.normal(0.015, 0.005))
        data.append({
            "Date": dates[i], "Open": c - r / 2, "High": c + r,
            "Low": c - r, "Close": c, "Volume": 10_000_000,
        })
    return pd.DataFrame(data).set_index("Date")


def make_high_vol(n_bars=500) -> pd.DataFrame:
    """High volatility — large daily swings."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    noise = rng.normal(0, 0.03, n_bars)  # 3% daily noise
    close = 100 * np.exp(np.cumsum(noise))
    data = []
    for i, c in enumerate(close):
        r = c * abs(rng.normal(0.04, 0.01))  # wide bars
        data.append({
            "Date": dates[i], "Open": c - r / 2, "High": c + r,
            "Low": c - r, "Close": c, "Volume": 10_000_000,
        })
    return pd.DataFrame(data).set_index("Date")


class TestMarketRegime:
    def test_trending_up(self):
        df = make_trend_up()
        c = MarketStateClassifier(df)
        state = c.classify()
        assert state.regime == MarketRegime.TRENDING_UP

    def test_ranging(self):
        df = make_ranging()
        c = MarketStateClassifier(df, adx_threshold=25.0)
        state = c.classify()
        # Ranging data should produce RANGING or TRANSITIONAL
        assert state.regime in (MarketRegime.RANGING, MarketRegime.TRANSITIONAL)

    def test_fallback_on_thin_data(self):
        df = make_trend_up(20)  # only 20 bars, not enough
        c = MarketStateClassifier(df)
        state = c.classify()
        assert state.regime == MarketRegime.TRANSITIONAL
        assert state.volatility == Volatility.NORMAL


class TestVolatility:
    def test_normal_on_trend_data(self):
        df = make_trend_up()
        c = MarketStateClassifier(df)
        state = c.classify()
        assert state.volatility in (Volatility.NORMAL, Volatility.LOW)


class TestStrategyTypeChecks:
    def test_trend_strategies(self):
        assert is_trend_strategy("turtle_trading") is True
        assert is_trend_strategy("enhanced_macd") is True
        assert is_trend_strategy("weekly_macd_kdj") is True
        assert is_trend_strategy("bollinger_mean_reversion") is False
        assert is_trend_strategy("daily_macd_kdj") is False

    def test_mean_reversion_strategies(self):
        assert is_mean_reversion_strategy("bollinger_mean_reversion") is True
        assert is_mean_reversion_strategy("turtle_trading") is False
        assert is_mean_reversion_strategy("daily_macd_kdj") is False
