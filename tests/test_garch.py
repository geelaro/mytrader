"""Tests for analysis.garch — EWMA, GJR-GARCH, forward VaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.garch import (
    RISKMETRICS_LAMBDA,
    ewma_volatility,
    fit_gjr_garch,
    forecast_volatility,
    forward_var,
    forward_var_summary,
)


def _gen_returns(n=500, seed=42, scale=0.012):
    """Stationary IID gaussian returns for stable tests."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(loc=0.0, scale=scale, size=n))


def _gen_garch_returns(n=500, seed=42,
                      omega=1e-5, alpha=0.06, gamma=0.10, beta=0.82):
    """Simulate returns from a true GJR-GARCH(1,1) process — used to test
    that the MLE can recover *something* close to a real GARCH process."""
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
    return pd.Series(r)


def _gen_vol_regime(n=500, seed=42):
    """Mix two regimes: low vol then high vol — vol forecasts should rise."""
    rng = np.random.default_rng(seed)
    low = rng.normal(loc=0, scale=0.005, size=n // 2)
    high = rng.normal(loc=0, scale=0.025, size=n - n // 2)
    return pd.Series(np.concatenate([low, high]))


class TestEWMA:
    def test_returns_series_same_length(self):
        r = _gen_returns(100)
        sigma = ewma_volatility(r)
        assert len(sigma) == len(r)
        assert sigma.index.equals(r.index)

    def test_sigma_strictly_positive(self):
        r = _gen_returns(100)
        sigma = ewma_volatility(r)
        assert (sigma > 0).all()

    def test_responds_to_shock(self):
        """A spike at t=100 should raise σ at t=101+."""
        r = _gen_returns(200, scale=0.005).copy()
        r.iloc[100] = -0.10   # 10% loss
        sigma = ewma_volatility(r)
        # σ after shock > σ just before
        assert sigma.iloc[105] > sigma.iloc[99]

    def test_default_lambda_is_riskmetrics(self):
        assert RISKMETRICS_LAMBDA == 0.94

    def test_lambda_bounds_validated(self):
        r = _gen_returns(50)
        with pytest.raises(ValueError):
            ewma_volatility(r, lambda_=1.0)
        with pytest.raises(ValueError):
            ewma_volatility(r, lambda_=0.0)

    def test_empty_input(self):
        assert ewma_volatility(pd.Series(dtype=float)).empty


class TestGJRGarch:
    def test_fit_returns_expected_keys(self):
        r = _gen_returns(500)
        m = fit_gjr_garch(r)
        for key in ["fitted", "method", "omega", "alpha", "gamma", "beta",
                    "persistence", "long_run_var", "sigma", "loglik", "n_obs"]:
            assert key in m

    def test_fit_succeeds_with_sufficient_data(self):
        """MLE should converge on real GARCH-process data."""
        r = _gen_garch_returns(800)
        m = fit_gjr_garch(r)
        assert m["fitted"] is True
        assert m["method"] == "gjr"

    def test_fitted_params_stationary(self):
        r = _gen_garch_returns(800)
        m = fit_gjr_garch(r)
        assert m["omega"] > 0
        assert 0 <= m["alpha"]
        assert 0 <= m["gamma"]
        assert 0 <= m["beta"]
        assert m["persistence"] < 1.0

    def test_falls_back_to_ewma_when_insufficient_data(self):
        r = _gen_returns(100)
        m = fit_gjr_garch(r)
        assert m["fitted"] is False
        assert m["method"] == "ewma_fallback"
        assert not m["sigma"].empty

    def test_sigma_series_length_matches_input(self):
        r = _gen_returns(500)
        m = fit_gjr_garch(r)
        assert len(m["sigma"]) == len(r)

    def test_recovers_garch_parameters_roughly(self):
        """On a sample big enough, MLE params should be in the right ballpark."""
        true_alpha, true_gamma, true_beta = 0.06, 0.10, 0.82
        r = _gen_garch_returns(2000, alpha=true_alpha, gamma=true_gamma,
                               beta=true_beta)
        m = fit_gjr_garch(r)
        assert m["fitted"]
        # Persistence is the most robustly identified quantity (sum of params)
        # — individual α/β have known identification problems.  Allow 30%.
        true_persistence = true_alpha + true_gamma / 2 + true_beta
        assert abs(m["persistence"] - true_persistence) < 0.15


class TestForecast:
    def test_one_step_returns_positive(self):
        r = _gen_garch_returns(800)
        m = fit_gjr_garch(r)
        f = forecast_volatility(m, last_return=float(r.iloc[-1]), horizon=1)
        assert f > 0

    def test_long_horizon_converges_to_long_run(self):
        """h→∞ should converge to long_run_var^0.5 for fitted GARCH."""
        r = _gen_garch_returns(800)
        m = fit_gjr_garch(r)
        if not m["fitted"]:
            pytest.skip("Model did not converge — skip")
        long_run_sigma = float(np.sqrt(m["long_run_var"]))
        f_far = forecast_volatility(m, last_return=0.0, horizon=500)
        assert abs(f_far - long_run_sigma) < long_run_sigma * 0.05

    def test_horizon_validated(self):
        r = _gen_returns(50)
        m = fit_gjr_garch(r)
        with pytest.raises(ValueError):
            forecast_volatility(m, horizon=0)


class TestForwardVaR:
    def test_returns_positive_loss(self):
        r = _gen_returns(500)
        v = forward_var(r, confidence=0.95)
        assert v > 0

    def test_higher_confidence_higher_var(self):
        r = _gen_returns(500)
        v95 = forward_var(r, confidence=0.95)
        v99 = forward_var(r, confidence=0.99)
        assert v99 > v95

    def test_ewma_and_gjr_close_for_gaussian_returns(self):
        """For stationary IID gaussian returns, both methods agree within 30%."""
        r = _gen_returns(500)
        v_e = forward_var(r, confidence=0.95, method="ewma")
        v_g = forward_var(r, confidence=0.95, method="gjr")
        ratio = max(v_e, v_g) / max(min(v_e, v_g), 1e-9)
        assert ratio < 1.3

    def test_higher_vol_regime_gives_higher_var(self):
        """A series ending in high-vol regime should forecast higher VaR
        than the same series ending in low-vol."""
        rising = _gen_vol_regime(500)
        falling = _gen_vol_regime(500)
        # Flip second half to make falling end in low vol.
        falling = pd.Series(falling.values[::-1])
        v_rising = forward_var(rising, confidence=0.95, method="ewma")
        v_falling = forward_var(falling, confidence=0.95, method="ewma")
        assert v_rising > v_falling

    def test_horizon_scaling(self):
        """5-day VaR > 1-day VaR (loss accumulates)."""
        r = _gen_returns(500)
        v1 = forward_var(r, confidence=0.95, horizon=1)
        v5 = forward_var(r, confidence=0.95, horizon=5)
        assert v5 > v1

    def test_empty_returns_zero(self):
        assert forward_var(pd.Series(dtype=float)) == 0.0

    def test_unknown_method_raises(self):
        r = _gen_returns(100)
        with pytest.raises(ValueError):
            forward_var(r, method="banana")


class TestSummary:
    def test_summary_shape(self):
        r = _gen_returns(500)
        s = forward_var_summary(r, confidences=(0.95, 0.99))
        assert s["n_obs"] == 500
        assert "95%" in s
        assert "99%" in s
        assert "ewma" in s["95%"]
        assert "gjr" in s["95%"]
        assert "sigma_forecast" in s
        assert "gjr_params" in s

    def test_summary_default_confidences(self):
        r = _gen_returns(500)
        s = forward_var_summary(r)
        assert "95%" in s
        assert "99%" in s
