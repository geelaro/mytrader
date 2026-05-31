"""Tests for analysis/factor_returns.py + factor_attribution.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analysis.factor_returns import FactorReturns
from analysis.factor_attribution import (
    FactorAttribution, AttributionResult, TRADING_DAYS,
)


# ===================================================================
# FactorReturns
# ===================================================================


def _make_synthetic_etf(start: str, end: str, n_bars: int = 500,
                        drift: float = 0.0003, vol: float = 0.012,
                        seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic ETF OHLCV frame on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)[:n_bars]
    log_ret = rng.normal(drift, vol, len(dates))
    close = 100 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": 1_000_000,
    }, index=dates)


@pytest.fixture
def mock_provider():
    """Provider returning deterministic synthetic prices per ticker."""
    provider = MagicMock()
    seeds = {"SPY": 1, "IWM": 2, "IVE": 3, "IVW": 4, "MTUM": 5, "QUAL": 6,
             "USMV": 7, "SHV": 8}

    def fake_get_daily(symbol, start=None, end=None, **kwargs):
        return _make_synthetic_etf(
            start or "2018-01-01",
            end or "2024-12-31",
            seed=seeds.get(symbol.upper(), 99),
        )
    provider.get_daily.side_effect = fake_get_daily
    return provider


class TestFactorReturnsModes:
    def test_full_mode_factor_names(self, mock_provider):
        fr = FactorReturns(mode="full", provider=mock_provider)
        assert fr.factor_names == ["MKT", "SMB", "HML", "MOM", "QMJ", "BAB"]

    def test_ff3_mode_factor_names(self, mock_provider):
        fr = FactorReturns(mode="ff3", provider=mock_provider)
        assert fr.factor_names == ["MKT", "SMB", "HML"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be 'full' or 'ff3'"):
            FactorReturns(mode="ff5")

    def test_inception_full(self, mock_provider):
        fr = FactorReturns(mode="full", provider=mock_provider)
        assert fr.inception == pd.Timestamp("2013-08-01")

    def test_inception_ff3(self, mock_provider):
        fr = FactorReturns(mode="ff3", provider=mock_provider)
        assert fr.inception == pd.Timestamp("2000-06-01")


class TestFactorReturnsLoad:
    def test_load_returns_factor_columns(self, mock_provider):
        fr = FactorReturns(mode="full", provider=mock_provider)
        df = fr.load("2018-01-01", "2020-12-31")
        for col in ["mkt", "smb", "hml", "mom", "qmj", "bab", "rf"]:
            assert col in df.columns

    def test_load_ff3_subset_columns(self, mock_provider):
        fr = FactorReturns(mode="ff3", provider=mock_provider)
        df = fr.load("2018-01-01", "2020-12-31")
        assert set(df.columns) == {"mkt", "smb", "hml", "rf"}

    def test_load_clips_to_inception(self, mock_provider, caplog):
        fr = FactorReturns(mode="full", provider=mock_provider)
        # Request data earlier than the 2013-08 floor — should warn + clip
        with caplog.at_level("WARNING"):
            fr.load("2010-01-01", "2014-12-31")
        assert any("clipping" in r.message.lower() for r in caplog.records)

    def test_load_drops_nan_rows(self, mock_provider):
        fr = FactorReturns(mode="full", provider=mock_provider)
        df = fr.load("2018-01-01", "2020-12-31")
        assert not df.isna().any().any()


# ===================================================================
# FactorAttribution — synthetic alpha/beta recovery
# ===================================================================


def _make_factors(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic factor DataFrame for regression tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame({
        "mkt": rng.normal(0.0003, 0.012, n_days),
        "smb": rng.normal(0.0001, 0.008, n_days),
        "hml": rng.normal(0.0001, 0.007, n_days),
        "rf": rng.normal(0.00008, 0.0001, n_days),
    }, index=dates)


def _make_equity_from_factors(factors: pd.DataFrame, true_alpha: float,
                              true_betas: dict, noise_vol: float = 0.003,
                              seed: int = 7) -> pd.Series:
    """Construct an equity curve with known alpha + beta structure.

    portfolio_ret = α + Σ β_i × f_i + ε
    """
    rng = np.random.default_rng(seed)
    factor_cols = [c for c in factors.columns if c != "rf"]
    factor_loading = sum(
        true_betas.get(c, 0) * factors[c] for c in factor_cols
    )
    excess_ret = true_alpha + factor_loading + rng.normal(0, noise_vol, len(factors))
    total_ret = excess_ret + factors["rf"]
    equity = 100 * (1 + total_ret).cumprod()
    return equity


class TestFactorAttributionRecovery:
    """Verify the regression recovers known synthetic α / β."""

    def test_recovers_zero_alpha(self):
        factors = _make_factors(n_days=1000)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0, true_betas={"mkt": 0.8},
            noise_vol=0.001,
        )
        attr = FactorAttribution(equity, factors)
        result = attr.regress()
        # α should be near zero (within 1 bp daily ≈ 2.5% annual)
        assert abs(result.alpha_daily) < 1e-4
        # β_market should recover near 0.8
        assert abs(result.betas["MKT"] - 0.8) < 0.03

    def test_recovers_positive_alpha(self):
        factors = _make_factors(n_days=1000)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0003, true_betas={"mkt": 0.5, "smb": 0.2},
            noise_vol=0.001,
        )
        attr = FactorAttribution(equity, factors)
        result = attr.regress()
        # Tolerances sized at ~2 standard errors (noise_vol / sqrt(N) ≈ 3e-5)
        assert abs(result.alpha_daily - 0.0003) < 8e-5
        assert abs(result.betas["MKT"] - 0.5) < 0.03
        assert abs(result.betas["SMB"] - 0.2) < 0.05

    def test_alpha_significance_flag(self):
        # Large alpha → should be significant
        factors = _make_factors(n_days=1000)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.001, true_betas={"mkt": 0.5}, noise_vol=0.001,
        )
        attr = FactorAttribution(equity, factors)
        result = attr.regress()
        assert result.alpha_is_significant
        assert abs(result.alpha_tstat) > 2.0


class TestFactorAttributionAPI:
    def test_missing_rf_raises(self):
        factors = pd.DataFrame({
            "mkt": [0.01, 0.02, -0.01],
        }, index=pd.bdate_range("2020-01-01", periods=3))
        equity = pd.Series([100, 101, 102], index=factors.index)
        with pytest.raises(ValueError, match="rf"):
            FactorAttribution(equity, factors)

    def test_too_few_observations_raises(self):
        factors = _make_factors(n_days=20)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0, true_betas={"mkt": 1.0},
        )
        attr = FactorAttribution(equity, factors)
        with pytest.raises(ValueError, match="too few"):
            attr.regress()

    def test_result_summary_string(self):
        factors = _make_factors(n_days=500)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0, true_betas={"mkt": 0.8},
        )
        attr = FactorAttribution(equity, factors)
        result = attr.regress()
        s = result.summary()
        assert "Factor Attribution" in s
        assert "MKT" in s
        assert "R²" in s

    def test_contribution_decomposition_columns(self):
        factors = _make_factors(n_days=500)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0001, true_betas={"mkt": 0.6, "smb": 0.1},
        )
        attr = FactorAttribution(equity, factors)
        attr.regress()
        contrib = attr.contribution()
        assert "alpha" in contrib.columns
        assert "MKT" in contrib.columns
        assert "SMB" in contrib.columns
        assert "residual" in contrib.columns
        assert "total" in contrib.columns

    def test_rolling_alpha_basic(self):
        factors = _make_factors(n_days=400)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0002, true_betas={"mkt": 0.7},
        )
        attr = FactorAttribution(equity, factors)
        rolling = attr.rolling_alpha(window_days=TRADING_DAYS)
        # 400 days, window 252 → 149 rolling windows
        assert len(rolling) > 100
        assert set(rolling.columns) >= {"alpha_daily", "alpha_annual", "alpha_tstat", "r_squared"}

    def test_rolling_alpha_too_short_raises(self):
        factors = _make_factors(n_days=100)
        equity = _make_equity_from_factors(
            factors, true_alpha=0.0, true_betas={"mkt": 1.0},
        )
        attr = FactorAttribution(equity, factors)
        with pytest.raises(ValueError, match="need >="):
            attr.rolling_alpha(window_days=TRADING_DAYS)


class TestAttributionVerdict:
    def _make_result(self, alpha_tstat, r_squared):
        return AttributionResult(
            alpha_daily=0.0, alpha_annual=0.0,
            alpha_tstat=alpha_tstat, alpha_pvalue=0.05,
            betas={"MKT": 0.5}, beta_tstats={"MKT": 5.0},
            beta_pvalues={"MKT": 0.0},
            r_squared=r_squared, adj_r_squared=r_squared - 0.01,
            n_obs=1000, residuals=pd.Series([]),
            factor_names=["MKT"],
        )

    def test_real_alpha(self):
        r = self._make_result(alpha_tstat=3.0, r_squared=0.5)
        assert "真 alpha" in r.verdict

    def test_suspicious_alpha(self):
        r = self._make_result(alpha_tstat=2.5, r_squared=0.9)
        assert "可疑" in r.verdict

    def test_no_alpha(self):
        r = self._make_result(alpha_tstat=0.5, r_squared=0.9)
        assert "无 alpha" in r.verdict

    def test_weak_signal(self):
        r = self._make_result(alpha_tstat=0.3, r_squared=0.4)
        assert "信号弱" in r.verdict
