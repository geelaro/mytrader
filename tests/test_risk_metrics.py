"""Tests for analysis/risk_metrics.py — Sortino / Calmar / Omega / etc."""

import numpy as np
import pandas as pd
import pytest

from analysis.risk_metrics import (
    calmar_ratio,
    information_ratio,
    mar_ratio,
    omega_ratio,
    pain_index,
    pain_ratio,
    risk_adjusted_summary,
    sortino_ratio,
)


@pytest.fixture
def symmetric_returns():
    """N(0.0005, 0.015), 1000 daily."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.015, 1000))


@pytest.fixture
def winning_streak():
    """100 days of mostly positive returns."""
    return pd.Series([0.005] * 90 + [-0.003] * 10)


# ===================================================================
# Sortino
# ===================================================================


class TestSortino:
    def test_no_downside_returns_zero(self, winning_streak):
        """If all returns > target (0 by default), no downside risk → 0.0."""
        all_positive = pd.Series([0.001] * 100)
        assert sortino_ratio(all_positive) == 0.0

    def test_deeper_loss_lowers_sortino(self):
        """Replace mild losses with bigger losses: Sortino drops."""
        mild = pd.Series([0.01, 0.01, 0.01, 0.01, -0.02] * 50)
        nasty = pd.Series([0.01, 0.01, 0.01, 0.01, -0.10] * 50)
        assert sortino_ratio(nasty) < sortino_ratio(mild)

    def test_target_threshold_used(self):
        """Higher target → more returns count as 'downside' → ratio drops."""
        r = pd.Series([0.005] * 50 + [-0.002] * 50)
        s_zero = sortino_ratio(r, target=0.0)
        s_high = sortino_ratio(r, target=0.003)  # Many positives now "downside"
        assert s_zero > s_high

    def test_empty(self):
        assert sortino_ratio(pd.Series(dtype=float)) == 0.0


# ===================================================================
# Calmar / MAR
# ===================================================================


class TestCalmar:
    def test_known_value(self):
        """Returns producing 10% annual, -20% MaxDD → Calmar = 0.5.

        Use 252 daily returns: cumulative = 1.10.  No drawdown intra-year
        → MaxDD ~ 0; force one by inserting a -20% day mid-year then recovery.
        """
        # 252 days. First 100 days flat, day 100 drops -25%, next 152 days recover to +10% total
        rng = np.random.default_rng(1)
        returns = np.zeros(252)
        # Smooth recovery so MaxDD is exactly the day-100 drop
        returns[100] = -0.20
        # Recover + grow: target cumulative = 1.10
        # After day 100: cum = 0.80. Need × 1.375 over 152 days = 0.00207/day
        returns[101:] = (1.10 / 0.80) ** (1 / 151) - 1
        s = pd.Series(returns)
        c = calmar_ratio(s)
        # CAGR ≈ 10%, MaxDD ≈ 20% → calmar ≈ 0.5
        assert 0.35 < c < 0.65

    def test_no_drawdown_returns_zero(self):
        r = pd.Series([0.001] * 252)
        assert calmar_ratio(r) == 0.0

    def test_mar_lookback_filter(self, symmetric_returns):
        """MAR with lookback should differ from full-history calmar
        when older data is dropped."""
        full = mar_ratio(symmetric_returns)  # no lookback
        recent = mar_ratio(symmetric_returns, lookback_years=1.0)
        # Different time windows → almost certainly different values
        assert full != recent


# ===================================================================
# Omega
# ===================================================================


class TestOmega:
    def test_threshold_zero_gain_loss_balance(self):
        """Symmetric N(0, σ) → Omega ≈ 1 (gains ≈ losses)."""
        rng = np.random.default_rng(42)
        zero_mean = pd.Series(rng.normal(0, 0.01, 5000))
        omega = omega_ratio(zero_mean, threshold=0)
        assert 0.85 < omega < 1.15

    def test_positive_drift_omega_gt_one(self, symmetric_returns):
        """N(0.0005, …) has positive drift → Omega(0) > 1."""
        assert omega_ratio(symmetric_returns, threshold=0) > 1.0

    def test_no_losses_returns_inf(self):
        r = pd.Series([0.001] * 100)
        assert omega_ratio(r, threshold=0) == float("inf")

    def test_no_gains_returns_zero(self):
        r = pd.Series([-0.001] * 100)
        assert omega_ratio(r, threshold=0) == 0.0


# ===================================================================
# Information Ratio
# ===================================================================


class TestInformationRatio:
    def test_zero_when_returns_equal_benchmark(self):
        r = pd.Series([0.001, 0.002, -0.001, 0.003] * 50)
        assert information_ratio(r, r) == 0.0

    def test_positive_when_outperforms(self):
        rng = np.random.default_rng(7)
        bm = pd.Series(rng.normal(0.0003, 0.01, 2000))
        # Strong positive alpha so 2000-sample mean is reliably > 0
        active_alpha = pd.Series(rng.normal(0.001, 0.002, 2000))
        r = bm + active_alpha
        ir = information_ratio(r, bm)
        assert ir > 0

    def test_handles_misaligned_indices(self):
        r = pd.Series([0.01, 0.02, 0.01], index=pd.bdate_range("2025-01-01", periods=3))
        b = pd.Series([0.005, 0.005], index=pd.bdate_range("2025-01-02", periods=2))
        # Two overlapping dates: active = [0.015, 0.005], non-zero std
        ir = information_ratio(r, b)
        # Just check it ran and produced a finite number
        assert np.isfinite(ir)


# ===================================================================
# Pain index / Pain ratio
# ===================================================================


class TestPain:
    def test_pain_index_no_drawdown_zero(self):
        r = pd.Series([0.001] * 100)
        assert pain_index(r) == 0.0

    def test_pain_index_is_positive_when_drawdown_exists(self):
        """Any sequence dipping below its peak has positive pain index."""
        r = pd.Series([0.05, -0.10, 0.03, -0.05, 0.02])
        assert pain_index(r) > 0

    def test_pain_index_lt_max_dd(self, symmetric_returns):
        """Average DD ≤ Max DD always."""
        from analysis.risk_metrics import _max_drawdown_pct
        assert pain_index(symmetric_returns) <= _max_drawdown_pct(symmetric_returns)

    def test_pain_ratio_zero_when_flat(self):
        r = pd.Series([0.001] * 100)
        assert pain_ratio(r) == 0.0


# ===================================================================
# Summary
# ===================================================================


class TestSummary:
    def test_all_keys_present(self, symmetric_returns):
        s = risk_adjusted_summary(symmetric_returns)
        for key in ("n_obs", "annual_return_pct", "max_drawdown_pct",
                    "sharpe", "sortino", "calmar", "omega",
                    "pain_index_pct", "pain_ratio"):
            assert key in s
        assert "information_ratio" not in s  # not provided

    def test_with_benchmark(self, symmetric_returns):
        bm = pd.Series(np.zeros(len(symmetric_returns)))
        s = risk_adjusted_summary(symmetric_returns, benchmark_returns=bm)
        assert "information_ratio" in s
        # With zero benchmark, IR == Sharpe (approximately, for ddof differences)
        assert abs(s["information_ratio"] - s["sharpe"]) < 0.01

    def test_n_obs_matches_input(self, symmetric_returns):
        s = risk_adjusted_summary(symmetric_returns)
        assert s["n_obs"] == len(symmetric_returns)
