"""Tests for analysis/risk_decomposition.py — MVaR / CVaR / risk parity."""

import numpy as np
import pandas as pd
import pytest

from analysis.risk_decomposition import (
    component_var,
    inverse_volatility_weights,
    marginal_var,
    parametric_portfolio_var,
    risk_contribution_pct,
    risk_decomposition_summary,
    risk_parity_weights,
)


# ---------------------------------------------------------------------------
# Synthetic price panels with known properties
# ---------------------------------------------------------------------------


@pytest.fixture
def uncorrelated_equal_vol():
    """3 independent symbols, same volatility.  500 daily bars."""
    rng = np.random.default_rng(42)
    n, nsym = 500, 3
    dates = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0.0005, 0.015, (n, nsym))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def two_asset_one_volatile():
    """A: low vol 0.005, B: high vol 0.030.  Uncorrelated."""
    rng = np.random.default_rng(7)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.005, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.030, n)))
    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def perfectly_correlated_pair():
    """A and B share the same underlying noise — correlation ≈ 1."""
    rng = np.random.default_rng(1)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    shared = rng.normal(0.0005, 0.015, n)
    a = 100 * np.exp(np.cumsum(shared))
    b = 200 * np.exp(np.cumsum(shared))  # different start, same path
    return pd.DataFrame({"A": a, "B": b}, index=dates)


# ===================================================================
# Parametric VaR (sanity vs analysis.var)
# ===================================================================


class TestParametricPortfolioVar:
    def test_positive_for_uncorrelated_equal(self, uncorrelated_equal_vol):
        v = parametric_portfolio_var(uncorrelated_equal_vol, {"A": 1, "B": 1, "C": 1})
        assert v > 0
        # Equal-weight 3 uncorrelated σ=1.5% should give roughly z(0.05)×σ_p
        # σ_p = √(3 × (1/3)² × 0.015²) ≈ 0.00866, VaR_95 ≈ 1.43%
        assert 0.005 < v < 0.025

    def test_empty_returns_zero(self):
        assert parametric_portfolio_var(pd.DataFrame(), {"A": 1}) == 0.0
        assert parametric_portfolio_var(pd.DataFrame({"A": [100]}), {}) == 0.0


# ===================================================================
# Component VaR sums to total — Euler property
# ===================================================================


class TestEulerDecomposition:
    def test_components_sum_to_total(self, uncorrelated_equal_vol):
        """Σ component_var_i ≈ parametric_portfolio_var (Euler theorem)."""
        weights = {"A": 1, "B": 2, "C": 3}
        total = parametric_portfolio_var(uncorrelated_equal_vol, weights)
        cvar = component_var(uncorrelated_equal_vol, weights)
        assert cvar.sum() == pytest.approx(total, abs=1e-9)

    def test_components_sum_extreme_weights(self, two_asset_one_volatile):
        """Same property under skewed weights."""
        weights = {"A": 9, "B": 1}
        total = parametric_portfolio_var(two_asset_one_volatile, weights)
        cvar = component_var(two_asset_one_volatile, weights)
        assert cvar.sum() == pytest.approx(total, abs=1e-9)


# ===================================================================
# High-vol symbol should dominate risk contribution
# ===================================================================


class TestRiskContribution:
    def test_high_vol_dominates_equal_weight(self, two_asset_one_volatile):
        """Equal capital, but B is 6× more volatile → B contributes way more."""
        rc = risk_contribution_pct(two_asset_one_volatile, {"A": 1, "B": 1})
        assert rc["B"] > rc["A"]
        # 6× σ → roughly 6² = 36× variance contribution under uncorrelated
        # → B should hold most of the risk
        assert rc["B"] > 80

    def test_sums_to_100_percent(self, uncorrelated_equal_vol):
        rc = risk_contribution_pct(uncorrelated_equal_vol, {"A": 1, "B": 1, "C": 1})
        assert rc.sum() == pytest.approx(100.0, abs=0.01)

    def test_uncorrelated_equal_weight_equal_contribution(self, uncorrelated_equal_vol):
        """Equal vol + equal weight + uncorrelated → each ~33%."""
        rc = risk_contribution_pct(uncorrelated_equal_vol, {"A": 1, "B": 1, "C": 1})
        for sym in ("A", "B", "C"):
            assert 25 < rc[sym] < 45  # ~33% with realisation noise


# ===================================================================
# Marginal VaR sign + monotonicity sanity
# ===================================================================


class TestMarginalVaR:
    def test_marginal_var_positive_for_long_position(self, uncorrelated_equal_vol):
        """Adding weight to a long, uncorrelated position increases VaR."""
        mvar = marginal_var(uncorrelated_equal_vol, {"A": 1, "B": 1, "C": 1})
        assert (mvar > 0).all()

    def test_higher_vol_has_higher_mvar(self, two_asset_one_volatile):
        mvar = marginal_var(two_asset_one_volatile, {"A": 1, "B": 1})
        assert mvar["B"] > mvar["A"]


# ===================================================================
# Correlation effects — the value-add
# ===================================================================


class TestCorrelationEffects:
    def test_correlated_pair_concentrates_risk(self, perfectly_correlated_pair):
        """Perfectly correlated A,B at 50/50 — total risk == one-asset risk."""
        rc = risk_contribution_pct(perfectly_correlated_pair, {"A": 1, "B": 1})
        # Each contributes 50% under perfect correlation + equal weight
        assert rc.sum() == pytest.approx(100.0, abs=0.01)
        assert 45 < rc["A"] < 55
        assert 45 < rc["B"] < 55


# ===================================================================
# Risk Parity
# ===================================================================


class TestRiskParity:
    def test_inverse_vol_normalises(self, two_asset_one_volatile):
        w = inverse_volatility_weights(two_asset_one_volatile)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        # Lower vol → bigger weight
        assert w["A"] > w["B"]

    def test_inverse_vol_ratio_approximates_vol_ratio(self, two_asset_one_volatile):
        w = inverse_volatility_weights(two_asset_one_volatile)
        # B has 6× σ → A should have ~6× weight
        assert w["A"] / w["B"] == pytest.approx(6.0, rel=0.4)

    def test_risk_parity_equalises_risk_contribution(self, two_asset_one_volatile):
        """Risk-parity weights should give ~equal risk contributions."""
        w = risk_parity_weights(two_asset_one_volatile)
        weights_dict = w.to_dict()
        rc = risk_contribution_pct(two_asset_one_volatile, weights_dict)
        # Each contributes ~50%
        assert abs(rc["A"] - rc["B"]) < 5  # within 5 percentage points

    def test_risk_parity_uncorrelated_equal_vol_equals_equal_weight(
            self, uncorrelated_equal_vol):
        """For 3 equal-vol uncorrelated, risk parity = 1/3 each (within noise)."""
        w = risk_parity_weights(uncorrelated_equal_vol)
        for sym in ("A", "B", "C"):
            assert abs(w[sym] - 1 / 3) < 0.05


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_zero_weight_excluded(self, uncorrelated_equal_vol):
        cvar = component_var(uncorrelated_equal_vol, {"A": 1, "B": 1, "C": 0})
        assert "C" not in cvar.index

    def test_missing_symbol_ignored(self, uncorrelated_equal_vol):
        cvar = component_var(uncorrelated_equal_vol, {"A": 1, "TSLA": 1})
        assert list(cvar.index) == ["A"]

    def test_insufficient_data_returns_empty(self):
        """Less than 30 obs → can't estimate covariance reliably."""
        dates = pd.bdate_range("2025-01-01", periods=10)
        prices = pd.DataFrame({"A": range(10), "B": range(10)}, index=dates)
        assert component_var(prices, {"A": 1, "B": 1}).empty

    def test_summary_includes_top_contributor(self, two_asset_one_volatile):
        s = risk_decomposition_summary(two_asset_one_volatile, {"A": 1, "B": 1})
        assert s["top_contributor"] == "B"
        assert s["top_contributor_pct"] > 80
        assert s["total_var_pct"] > 0
        assert "weight" in s["by_symbol"].columns
        assert "cvar_pct" in s["by_symbol"].columns
