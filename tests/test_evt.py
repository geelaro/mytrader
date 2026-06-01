"""Tests for analysis/evt.py — POT/GPD tail estimation."""

import numpy as np
import pandas as pd
import pytest

from analysis.evt import (
    evt_es,
    evt_summary,
    evt_var,
    fit_gpd,
)


# ---------------------------------------------------------------------------
# Fixtures — distributions with known tail behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_returns():
    """1500 daily ~ N(0.0005, 0.015) — thin tails, ξ should be ~0."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.015, 1500))


@pytest.fixture
def student_t_returns():
    """Student-t with df=3, scaled — heavy tails, ξ should be > 0."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_t(3, 1500) * 0.01)


@pytest.fixture
def bounded_returns():
    """Truncated uniform — bounded tail, ξ should be < 0 (light tail)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(-0.03, 0.03, 1500))


# ===================================================================
# fit_gpd
# ===================================================================


class TestFitGpd:
    def test_returns_required_keys(self, normal_returns):
        fit = fit_gpd(normal_returns)
        assert fit is not None
        for key in ("xi", "beta", "threshold", "n_exceed", "n_total"):
            assert key in fit

    def test_threshold_is_positive_for_typical_data(self, normal_returns):
        """For roughly-zero-mean returns, 95th percentile loss > 0."""
        fit = fit_gpd(normal_returns, threshold_quantile=0.95)
        assert fit["threshold"] > 0

    def test_n_exceed_matches_threshold(self, normal_returns):
        """N_exceed should be ~ 5% of n_total when threshold = 95%."""
        fit = fit_gpd(normal_returns, threshold_quantile=0.95)
        expected = int(0.05 * len(normal_returns))
        # Off-by-one tolerance for percentile interpolation
        assert abs(fit["n_exceed"] - expected) <= 3

    def test_xi_positive_for_heavy_tails(self, student_t_returns):
        """Student-t(df=3) is heavy-tailed → ξ > 0."""
        fit = fit_gpd(student_t_returns)
        assert fit is not None
        assert fit["xi"] > 0

    def test_xi_near_zero_for_normal(self, normal_returns):
        """Normal distribution should have ξ near 0."""
        fit = fit_gpd(normal_returns)
        assert fit is not None
        assert abs(fit["xi"]) < 0.5

    def test_beta_positive(self, normal_returns):
        fit = fit_gpd(normal_returns)
        assert fit["beta"] > 0

    def test_insufficient_data_returns_none(self):
        """Too few exceedances → can't fit."""
        # With threshold 0.95 and 100 obs, only ~5 exceedances — below 20 floor
        rng = np.random.default_rng(42)
        small = pd.Series(rng.normal(0, 0.01, 100))
        fit = fit_gpd(small, threshold_quantile=0.95)
        assert fit is None

    def test_empty_returns_none(self):
        assert fit_gpd(pd.Series(dtype=float)) is None

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            fit_gpd(pd.Series([0.01] * 100), threshold_quantile=0)
        with pytest.raises(ValueError):
            fit_gpd(pd.Series([0.01] * 100), threshold_quantile=1.5)


# ===================================================================
# evt_var
# ===================================================================


class TestEvtVar:
    def test_invalid_confidence_raises(self, normal_returns):
        for c in (0, 1, -0.5, 1.5):
            with pytest.raises(ValueError):
                evt_var(normal_returns, confidence=c)

    def test_below_threshold_returns_zero(self, normal_returns):
        """EVT not applicable to confidence < threshold_quantile."""
        result = evt_var(normal_returns, confidence=0.90, threshold_quantile=0.95)
        assert result == 0.0

    def test_returns_positive_for_valid_input(self, normal_returns):
        var = evt_var(normal_returns, confidence=0.99)
        assert var > 0

    def test_higher_confidence_larger_var(self, normal_returns):
        """VaR is monotone-increasing in confidence."""
        v99 = evt_var(normal_returns, confidence=0.99)
        v995 = evt_var(normal_returns, confidence=0.995)
        v999 = evt_var(normal_returns, confidence=0.999)
        assert v99 < v995 < v999

    def test_heavy_tail_var_exceeds_normal_at_extreme(
            self, normal_returns, student_t_returns):
        """At 99.9%, t-distribution VaR should be much larger than normal."""
        v_normal = evt_var(normal_returns, confidence=0.999)
        v_t = evt_var(student_t_returns, confidence=0.999)
        assert v_t > v_normal

    def test_no_fit_returns_zero(self):
        """No fit possible → return 0 silently."""
        small = pd.Series([0.01, -0.01] * 5)
        assert evt_var(small, confidence=0.99) == 0.0


# ===================================================================
# evt_es
# ===================================================================


class TestEvtEs:
    def test_es_at_least_var(self, normal_returns):
        """ES ≥ VaR always (mean of tail ≥ threshold)."""
        for c in (0.99, 0.995, 0.999):
            var = evt_var(normal_returns, confidence=c)
            es = evt_es(normal_returns, confidence=c)
            assert es >= var

    def test_es_zero_when_var_zero(self, normal_returns):
        """If VaR isn't computable, ES isn't either."""
        assert evt_es(normal_returns, confidence=0.90) == 0.0

    def test_es_finite_for_xi_lt_one(self, normal_returns):
        es = evt_es(normal_returns, confidence=0.99)
        assert np.isfinite(es)


# ===================================================================
# evt_summary
# ===================================================================


class TestEvtSummary:
    def test_returns_all_confidences(self, normal_returns):
        s = evt_summary(normal_returns)
        confs = {row["confidence"] for row in s["comparison"]}
        assert confs == {"95%", "99%", "99.5%", "99.9%"}

    def test_includes_fit_details(self, normal_returns):
        s = evt_summary(normal_returns)
        assert s["fit"] is not None
        assert "xi" in s["fit"]
        assert "n_exceed" in s["fit"]

    def test_warning_on_failed_fit(self):
        """Too little data → warning explaining missing EVT."""
        rng = np.random.default_rng(42)
        tiny = pd.Series(rng.normal(0, 0.01, 50))
        s = evt_summary(tiny)
        assert s["fit"] is None
        assert s["warning"] is not None

    def test_evt_extrapolates_beyond_historical(self, normal_returns):
        """At 99.9% historical has 1-2 obs; EVT should still produce a value."""
        s = evt_summary(normal_returns)
        # Find the 99.9% row
        for row in s["comparison"]:
            if row["confidence"] == "99.9%":
                # EVT should give a meaningful (positive) estimate
                assert row["evt"] > 0
                break
        else:
            pytest.fail("99.9% row missing")

    def test_evt_matches_historical_at_lower_confidence(self, normal_returns):
        """At 99%, EVT and historical should be in the same ballpark
        (within a factor of ~2)."""
        s = evt_summary(normal_returns)
        for row in s["comparison"]:
            if row["confidence"] == "99%":
                hist = row["historical"]
                evt = row["evt"]
                if hist > 0:
                    ratio = evt / hist
                    assert 0.5 < ratio < 2.0
                break
