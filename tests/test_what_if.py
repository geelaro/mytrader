"""Tests for analysis/what_if.py — rebalance previews."""

import numpy as np
import pandas as pd
import pytest

from analysis.what_if import apply_rebalance, compare_portfolios


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=500)
    rets = rng.normal(0.0005, 0.015, (500, 3))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["AAPL", "MSFT", "JPM"])


# ===================================================================
# apply_rebalance
# ===================================================================


class TestApplyRebalance:
    def test_simple_increase(self):
        result = apply_rebalance({"A": 1, "B": 1}, {"A": 0.5})
        assert result == {"A": 1.5, "B": 1}

    def test_simple_decrease(self):
        result = apply_rebalance({"A": 1, "B": 1}, {"A": -0.3})
        assert result == {"A": 0.7, "B": 1}

    def test_zero_or_negative_dropped(self):
        result = apply_rebalance({"A": 1, "B": 1}, {"A": -1.0})
        assert "A" not in result
        assert result == {"B": 1}

    def test_overshoot_drop(self):
        """Delta more negative than current → position closed, not negative."""
        result = apply_rebalance({"A": 0.5}, {"A": -2.0})
        assert result == {}

    def test_add_new_symbol(self):
        result = apply_rebalance({"A": 1}, {"B": 0.5})
        assert result == {"A": 1, "B": 0.5}

    def test_no_deltas_returns_copy(self):
        original = {"A": 1, "B": 2}
        result = apply_rebalance(original, {})
        assert result == original
        # Mutation isolation
        result["A"] = 99
        assert original["A"] == 1

    def test_multiple_deltas(self):
        result = apply_rebalance(
            {"A": 1, "B": 1, "C": 1},
            {"A": 0.5, "B": -0.3, "D": 0.4},
        )
        assert result == {"A": 1.5, "B": 0.7, "C": 1, "D": 0.4}


# ===================================================================
# compare_portfolios
# ===================================================================


class TestComparePortfolios:
    def test_zero_rebalance_zero_deltas(self, synthetic_prices):
        """Same weights → all deltas = 0."""
        w = {"AAPL": 1, "MSFT": 1, "JPM": 1}
        result = compare_portfolios(synthetic_prices, w, w)
        for v in result["deltas"].values():
            assert abs(v) < 1e-6

    def test_concentrating_increases_hhi(self, synthetic_prices):
        before = {"AAPL": 1, "MSFT": 1, "JPM": 1}  # equal
        after = {"AAPL": 3, "MSFT": 1, "JPM": 1}   # more AAPL
        result = compare_portfolios(synthetic_prices, before, after)
        # HHI should increase
        assert result["deltas"]["hhi"] > 0
        # Effective N should decrease
        assert result["deltas"]["effective_n"] < 0
        # Both before/after snapshots populated
        assert result["before"]["hhi"] > 0
        assert result["after"]["hhi"] > 0

    def test_diversifying_decreases_hhi(self, synthetic_prices):
        before = {"AAPL": 4, "MSFT": 1}  # concentrated
        after = {"AAPL": 1, "MSFT": 1, "JPM": 1}  # diversified
        result = compare_portfolios(synthetic_prices, before, after)
        assert result["deltas"]["hhi"] < 0
        assert result["deltas"]["effective_n"] > 0

    def test_sector_map_propagates(self, synthetic_prices):
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Fin"}
        w = {"AAPL": 1, "MSFT": 1, "JPM": 1}
        result = compare_portfolios(synthetic_prices, w, w, sector_map=sectors)
        assert "sector_hhi" in result["before"]
        assert "sector_exposure" in result["before"]
        assert "sector_hhi" in result["deltas"]

    def test_summary_text_format(self, synthetic_prices):
        before = {"AAPL": 1, "MSFT": 1, "JPM": 1}
        after = {"AAPL": 3, "MSFT": 1, "JPM": 1}
        result = compare_portfolios(synthetic_prices, before, after)
        text = result["summary_text"]
        assert "VaR" in text
        assert "HHI" in text
        # Arrow indicates direction
        assert "↑" in text or "↓" in text

    def test_var_changes_with_weight_shift(self, synthetic_prices):
        before = {"AAPL": 1, "MSFT": 1, "JPM": 1}
        after = {"AAPL": 5, "MSFT": 0.1, "JPM": 0.1}  # concentrate in AAPL
        result = compare_portfolios(synthetic_prices, before, after)
        # Higher concentration usually increases idiosyncratic risk → VaR up
        # But this is data-dependent — just check VaR changed
        assert abs(result["deltas"]["var_pct"]) > 0


# ===================================================================
# Integration with apply_rebalance
# ===================================================================


class TestRebalanceCompareIntegration:
    def test_trim_dominant_then_compare_workflow(self, synthetic_prices):
        """Realistic flow: portfolio with one dominant position, trim it down,
        verify the rebalance preview shows HHI dropping."""
        # AAPL is 4× the others — clearly the dominant position
        current = {"AAPL": 4, "MSFT": 1, "JPM": 1}
        trimmed = apply_rebalance(current, {"AAPL": -2.0})  # halve the dominant
        assert trimmed["AAPL"] == 2.0
        result = compare_portfolios(synthetic_prices, current, trimmed)
        # Trimming the dominant position toward equal weight → HHI drops
        assert result["deltas"]["hhi"] < 0
        # And the rebalanced portfolio has higher effective N (more diversified)
        assert result["deltas"]["effective_n"] > 0
