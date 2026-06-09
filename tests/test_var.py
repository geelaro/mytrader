"""Tests for analysis/var.py — VaR / ES / portfolio aggregation."""

import numpy as np
import pandas as pd
import pytest

from analysis.var import (
    conditional_var,
    historical_var,
    parametric_var,
    portfolio_returns,
    var_summary,
)


@pytest.fixture
def normal_returns():
    """1000 daily returns ~ N(μ=0.0005, σ=0.015) — realistic equity."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.015, 1000))


@pytest.fixture
def fat_tail_returns():
    """Student-t with df=4 — fat tails, similar to real market crashes."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_t(4, 1000) * 0.01)


# ===================================================================
# Historical VaR
# ===================================================================


class TestHistoricalVaR:
    def test_returns_positive_loss(self, normal_returns):
        v95 = historical_var(normal_returns, 0.95)
        assert v95 > 0
        # ~5th percentile of N(0.0005, 0.015): roughly 0.024
        assert 0.015 < v95 < 0.035

    def test_higher_confidence_means_larger_var(self, normal_returns):
        assert historical_var(normal_returns, 0.99) > historical_var(normal_returns, 0.95)

    def test_empty_series_returns_zero(self):
        assert historical_var(pd.Series(dtype=float), 0.95) == 0.0

    def test_all_positive_returns_yields_zero(self):
        """If even the worst day is a gain, VaR is 0."""
        r = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
                       0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
                       0.017, 0.018, 0.019, 0.020])
        assert historical_var(r, 0.95) == 0.0

    def test_nan_and_inf_dropped(self):
        r = pd.Series([0.01, -0.02, np.nan, np.inf, -np.inf, -0.03, 0.005])
        result = historical_var(r, 0.5)
        assert result > 0  # something computable from cleaned data
        assert not np.isnan(result)


# ===================================================================
# Parametric VaR
# ===================================================================


class TestParametricVaR:
    def test_matches_normal_closed_form(self, normal_returns):
        """For genuinely normal data, parametric ~= historical."""
        ph = parametric_var(normal_returns, 0.95)
        hh = historical_var(normal_returns, 0.95)
        assert abs(ph - hh) < 0.005  # within 50bp

    def test_understates_fat_tail_risk(self, fat_tail_returns):
        """For fat-tailed returns, historical > parametric (intentional)."""
        ph = parametric_var(fat_tail_returns, 0.99)
        hh = historical_var(fat_tail_returns, 0.99)
        # Fat tails: historical 99th percentile worse than gaussian assumption
        assert hh > ph

    def test_zero_std_returns_zero(self):
        """No volatility → no risk under Gaussian assumption."""
        r = pd.Series([0.001] * 100)
        assert parametric_var(r, 0.95) == 0.0


# ===================================================================
# Conditional VaR (Expected Shortfall)
# ===================================================================


class TestConditionalVaR:
    def test_cvar_at_least_var(self, normal_returns):
        """ES ≥ VaR always (mean of tail ≥ threshold of tail)."""
        for c in (0.90, 0.95, 0.99):
            assert conditional_var(normal_returns, c) >= historical_var(normal_returns, c)

    def test_cvar_strictly_greater_for_continuous(self, fat_tail_returns):
        """For continuous distributions ES is strictly larger than VaR."""
        for c in (0.95, 0.99):
            assert conditional_var(fat_tail_returns, c) > historical_var(fat_tail_returns, c)

    def test_empty_tail_returns_zero(self):
        r = pd.Series([0.001] * 100)  # no losses
        assert conditional_var(r, 0.95) == 0.0


# ===================================================================
# Input validation
# ===================================================================


class TestValidation:
    def test_invalid_confidence_raises(self):
        for c in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError):
                historical_var(pd.Series([0.01]), c)
            with pytest.raises(ValueError):
                parametric_var(pd.Series([0.01]), c)
            with pytest.raises(ValueError):
                conditional_var(pd.Series([0.01]), c)


# ===================================================================
# Portfolio aggregation
# ===================================================================


class TestPortfolioReturns:
    @pytest.fixture
    def prices(self):
        """100-day prices for 3 symbols."""
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=100)
        return pd.DataFrame({
            "AAPL": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 100))),
            "MSFT": 200 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 100))),
            "GOOG": 150 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 100))),
        }, index=dates)

    def test_equal_weight_average(self, prices):
        pf = portfolio_returns(prices, {"AAPL": 1, "MSFT": 1, "GOOG": 1})
        assert isinstance(pf, pd.Series)
        # 1 less than prices because pct_change drops first row
        assert len(pf) == len(prices) - 1
        # Equal-weight portfolio return == average of symbol returns
        expected = prices.pct_change(fill_method=None).dropna().mean(axis=1)
        pd.testing.assert_series_equal(pf, expected, check_names=False)

    def test_weights_normalised(self, prices):
        """Weights are normalised — sum doesn't need to be 1."""
        pf1 = portfolio_returns(prices, {"AAPL": 1, "MSFT": 1})
        pf2 = portfolio_returns(prices, {"AAPL": 10, "MSFT": 10})
        pd.testing.assert_series_equal(pf1, pf2)

    def test_missing_symbol_ignored(self, prices):
        """Weight for non-existent symbol silently dropped."""
        pf = portfolio_returns(prices, {"AAPL": 1, "TSLA": 1})
        # Effectively AAPL with full weight
        expected = prices["AAPL"].pct_change(fill_method=None).dropna()
        pd.testing.assert_series_equal(pf, expected, check_names=False)

    def test_empty_inputs(self, prices):
        assert portfolio_returns(pd.DataFrame(), {"AAPL": 1}).empty
        assert portfolio_returns(prices, {}).empty
        assert portfolio_returns(prices, {"AAPL": 0}).empty


# ===================================================================
# var_summary convenience
# ===================================================================


class TestVarSummary:
    def test_default_confidences(self, normal_returns):
        s = var_summary(normal_returns)
        assert "95%" in s and "99%" in s
        assert "n_obs" in s and "mean" in s and "std" in s
        assert s["n_obs"] == len(normal_returns)
        for key in ("historical", "parametric", "cvar"):
            assert key in s["95%"]
            assert key in s["99%"]
            assert s["95%"][key] >= 0
            assert s["99%"][key] >= 0

    def test_custom_confidences(self, normal_returns):
        s = var_summary(normal_returns, confidences=(0.90, 0.975))
        assert "90%" in s
        assert "97%" in s  # 0.975 * 100 → 97
        assert "95%" not in s

    def test_99_larger_than_95(self, normal_returns):
        s = var_summary(normal_returns)
        # 99% VaR > 95% VaR for all three estimators (with high probability)
        assert s["99%"]["historical"] >= s["95%"]["historical"]
        assert s["99%"]["parametric"] >= s["95%"]["parametric"]
        assert s["99%"]["cvar"] >= s["95%"]["cvar"]
