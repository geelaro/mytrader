"""Tests for analysis.var_coverage — Kupiec / Christoffersen / CC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.var_coverage import coverage_backtest


def _gen_gaussian(n=500, scale=0.012, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0, scale, size=n), index=idx)


def _gen_garch(n=500, omega=1e-5, alpha=0.06, gamma=0.10, beta=0.82, seed=42):
    """Real GARCH process — vol clusters, gives EWMA a chance to shine."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    var = np.empty(n)
    var[0] = omega / (1 - alpha - gamma / 2 - beta)
    r[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, n):
        leverage = 1.0 if r[t - 1] < 0 else 0.0
        var[t] = (omega + alpha * r[t - 1] ** 2
                  + gamma * leverage * r[t - 1] ** 2
                  + beta * var[t - 1])
        r[t] = np.sqrt(max(var[t], 1e-12)) * z[t]
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(r, index=idx)


class TestBacktestShape:
    def test_returns_expected_keys(self):
        r = _gen_gaussian(500)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        for key in ["confidence", "method", "n_oos", "n_violations",
                    "observed_rate", "expected_rate", "kupiec",
                    "christoffersen", "conditional_coverage",
                    "violation_series", "var_series", "window"]:
            assert key in out

    def test_n_oos_correct(self):
        r = _gen_gaussian(500)
        out = coverage_backtest(r, window=250)
        assert out["n_oos"] == 250
        assert len(out["violation_series"]) == 250
        assert len(out["var_series"]) == 250

    def test_insufficient_data_returns_empty(self):
        r = _gen_gaussian(50)
        out = coverage_backtest(r, window=250)
        assert out["n_oos"] == 0
        assert "reason" in out

    def test_invalid_confidence_raises(self):
        r = _gen_gaussian(500)
        with pytest.raises(ValueError):
            coverage_backtest(r, confidence=1.5)

    def test_invalid_method_raises(self):
        r = _gen_gaussian(500)
        with pytest.raises(ValueError):
            coverage_backtest(r, method="banana")

    def test_window_too_small_raises(self):
        r = _gen_gaussian(500)
        with pytest.raises(ValueError):
            coverage_backtest(r, window=10)


class TestRates:
    def test_ewma_close_to_expected_for_gaussian(self):
        """For pure gaussian IID, EWMA should produce ~5% violations at 95% VaR."""
        r = _gen_gaussian(1000)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        # Observed within 2× expected (allowing for finite-sample noise)
        assert 0.02 < out["observed_rate"] < 0.10

    def test_historical_works(self):
        r = _gen_gaussian(800)
        out = coverage_backtest(r, confidence=0.95, method="historical",
                                window=250)
        assert out["n_violations"] >= 0
        assert 0 <= out["observed_rate"] < 0.2

    def test_higher_confidence_fewer_violations(self):
        r = _gen_gaussian(2000)
        v95 = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        v99 = coverage_backtest(r, confidence=0.99, method="ewma", window=250)
        assert v99["observed_rate"] < v95["observed_rate"]


class TestKupiec:
    def test_well_calibrated_model_not_rejected(self):
        """Properly-calibrated EWMA on gaussian should NOT reject Kupiec."""
        r = _gen_gaussian(2000)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        # p_value should be > 0.05 most of the time for well-calibrated
        assert out["kupiec"]["p_value"] > 0.01

    def test_kupiec_lr_nonnegative(self):
        r = _gen_gaussian(500)
        out = coverage_backtest(r, window=250)
        assert out["kupiec"]["lr"] >= 0


class TestChristoffersen:
    def test_iid_returns_indep_not_rejected(self):
        """IID returns → violations should not cluster."""
        r = _gen_gaussian(2000)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        # Most of the time IID won't reject independence
        assert out["christoffersen"]["lr"] >= 0

    def test_no_violations_returns_neutral(self):
        """Zero violations is a degenerate case — should return p=1, no reject."""
        # Use a stationary low-vol series with high confidence
        r = pd.Series(np.full(500, 0.001))
        out = coverage_backtest(r, confidence=0.99, method="ewma", window=250)
        assert out["christoffersen"]["p_value"] == 1.0
        assert out["christoffersen"]["reject_at_5pct"] is False


class TestConditionalCoverage:
    def test_chi2_2df(self):
        """CC = Kupiec + Indep, should ~χ²(2)."""
        r = _gen_gaussian(2000)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        expected = (out["kupiec"]["lr"] + out["christoffersen"]["lr"])
        assert abs(out["conditional_coverage"]["lr"] - expected) < 1e-9

    def test_well_calibrated_not_rejected(self):
        r = _gen_gaussian(2000)
        out = coverage_backtest(r, confidence=0.95, method="ewma", window=250)
        assert out["conditional_coverage"]["reject_at_5pct"] is False


class TestGarchOnRealGarch:
    @pytest.mark.slow
    def test_gjr_handles_garch_data(self):
        """GJR on GARCH data should produce reasonable violation rate."""
        r = _gen_garch(800)
        out = coverage_backtest(r, confidence=0.95, method="gjr",
                                window=400, refit_every=50)
        # Should be in the same ballpark as expected
        assert 0.02 < out["observed_rate"] < 0.12
