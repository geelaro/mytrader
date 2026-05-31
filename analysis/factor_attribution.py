"""Factor-attribution regression — decompose a portfolio's returns into
factor exposures + idiosyncratic alpha.

Model
-----
    r_p,t - rf_t = α + Σ β_i × f_i,t + ε_t

where ``f_i,t`` are the factor excess returns from :mod:`analysis.factor_returns`.

Statistics use Newey-West HAC standard errors (5-lag default) to correct for
serial correlation and heteroskedasticity in daily strategy returns.

The annualised α is reported on a 252-trading-day basis, with the t-stat
derived from the daily regression — large t (|t| > 2) means the alpha is
statistically distinguishable from zero, not that the alpha is economically
meaningful.

Usage
-----
    from analysis.factor_returns import FactorReturns
    from analysis.factor_attribution import FactorAttribution

    factors = FactorReturns(mode="full").load("2018-01-01", "2024-12-31")
    attr = FactorAttribution(equity_curve, factors)
    result = attr.regress()
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
TRADING_WEEKS = 52
TRADING_MONTHS = 12


@dataclass
class AttributionResult:
    """OLS regression output for one portfolio against a factor set."""

    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    beta_pvalues: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series
    factor_names: list[str] = field(default_factory=list)

    # Verdict helpers --------------------------------------------------

    @property
    def alpha_is_significant(self) -> bool:
        return abs(self.alpha_tstat) >= 2.0

    @property
    def verdict(self) -> str:
        """One-line interpretation following the README rules table."""
        sig = self.alpha_is_significant
        explained = self.r_squared
        if sig and explained < 0.7:
            return "真 alpha — alpha 显著 + 因子解释 < 70%, 值得放大"
        if sig and explained >= 0.85:
            return "可疑 alpha — alpha 显著但 R² 极高, 可能是因子组合"
        if not sig and explained >= 0.85:
            return "基本无 alpha — 收益基本被因子暴露解释"
        return "信号弱 — alpha 不显著且 R² 偏低, 样本不足或策略不稳"

    def summary(self) -> str:
        """Format for terminal / report output."""
        lines = [
            "=" * 60,
            "  Factor Attribution",
            "=" * 60,
            f"  Observations:    {self.n_obs}",
            f"  R²:              {self.r_squared:.3f}",
            f"  Adj R²:          {self.adj_r_squared:.3f}",
            "",
            f"  α (daily):       {self.alpha_daily * 100:+.4f}%",
            f"  α (annualised):  {self.alpha_annual * 100:+.2f}%   "
            f"t={self.alpha_tstat:+.2f}  p={self.alpha_pvalue:.3f}",
            "",
            f"  {'Factor':<8s} {'β':>8s} {'t-stat':>8s} {'p-value':>8s}",
            "  " + "-" * 36,
        ]
        for name in self.factor_names:
            beta = self.betas.get(name, np.nan)
            t = self.beta_tstats.get(name, np.nan)
            p = self.beta_pvalues.get(name, np.nan)
            lines.append(f"  {name:<8s} {beta:>+8.3f} {t:>+8.2f} {p:>8.3f}")
        lines.append("")
        lines.append(f"  Verdict: {self.verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def contribution(self) -> pd.DataFrame:
        """Per-day return decomposition: α + Σ β_i × f_i + ε.

        Useful to see *when* each factor contributed (e.g. momentum drove
        2020-Q2, market drove 2022 drawdown).
        """
        df = pd.DataFrame(index=self.residuals.index)
        df["alpha"] = self.alpha_daily
        # Factor contributions need the original factor returns, which the
        # caller must pass in via attribution_contribution_series().
        df["residual"] = self.residuals
        return df


class FactorAttribution:
    """Run OLS factor regression with Newey-West HAC standard errors.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity values indexed by date. Returns are derived
        internally via pct_change().
    factors : pd.DataFrame
        Factor returns + ``rf`` column from :class:`FactorReturns`.
    nw_lags : int
        Newey-West lag length (default 5 ≈ one trading week).
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        factors: pd.DataFrame,
        nw_lags: int = 5,
    ):
        if "rf" not in factors.columns:
            raise ValueError("factors must include an 'rf' (risk-free) column")
        self.equity_curve = equity_curve
        self.factors = factors
        self.nw_lags = nw_lags
        self._result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self._aligned: Optional[pd.DataFrame] = None
        # Detect equity frequency once. Weekly strategies (macd_kdj freq="W")
        # produce a weekly equity curve — comparing weekly returns against
        # daily factor returns inflates α by ~5×. Detect + resample factors
        # to match. Threshold of 3 days catches weekly (~7d) and beyond.
        self._periods_per_year = self._detect_periods_per_year(equity_curve)

    # ------------------------------------------------------------------

    def regress(self) -> AttributionResult:
        """Run OLS + Newey-West and return a parsed AttributionResult."""
        aligned = self._align()
        if len(aligned) < 30:
            raise ValueError(
                f"too few overlapping observations ({len(aligned)}); "
                f"need >= 30 for a meaningful regression"
            )

        factor_cols = [c for c in aligned.columns if c not in ("ret", "rf")]
        y = aligned["ret"] - aligned["rf"]  # excess return
        X = sm.add_constant(aligned[factor_cols])

        model = sm.OLS(y, X)
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": self.nw_lags})
        self._result = result

        params = result.params
        tvalues = result.tvalues
        pvalues = result.pvalues

        alpha_daily = float(params["const"])
        # Annualise by the actual sampling frequency, not by hardcoded 252.
        # Weekly equity → ×52, monthly → ×12, daily → ×252.
        alpha_annual = alpha_daily * self._periods_per_year

        return AttributionResult(
            alpha_daily=alpha_daily,
            alpha_annual=alpha_annual,
            alpha_tstat=float(tvalues["const"]),
            alpha_pvalue=float(pvalues["const"]),
            betas={c.upper(): float(params[c]) for c in factor_cols},
            beta_tstats={c.upper(): float(tvalues[c]) for c in factor_cols},
            beta_pvalues={c.upper(): float(pvalues[c]) for c in factor_cols},
            r_squared=float(result.rsquared),
            adj_r_squared=float(result.rsquared_adj),
            n_obs=int(result.nobs),
            residuals=pd.Series(result.resid, index=y.index, name="residual"),
            factor_names=[c.upper() for c in factor_cols],
        )

    # ------------------------------------------------------------------

    def rolling_alpha(self, window_days: int = TRADING_DAYS) -> pd.DataFrame:
        """Compute rolling-window α + t-stat to spot decay.

        ``window_days`` is in trading days (252 = 1y). For weekly equity curves
        it's translated to ``window_days / 5`` periods internally.

        Returns
        -------
        DataFrame[date, alpha_daily, alpha_annual, alpha_tstat, r_squared]
        indexed by the window's end date.
        """
        aligned = self._align()
        # Translate the day-based window into the actual sampling frequency
        period_window = max(20, int(window_days * self._periods_per_year / TRADING_DAYS))
        if len(aligned) < period_window + 5:
            raise ValueError(
                f"need >= {period_window + 5} observations for rolling alpha "
                f"(requested {window_days} trading days = {period_window} periods)"
            )

        factor_cols = [c for c in aligned.columns if c not in ("ret", "rf")]
        rows = []
        for end_i in range(period_window, len(aligned) + 1):
            window = aligned.iloc[end_i - period_window:end_i]
            y = window["ret"] - window["rf"]
            X = sm.add_constant(window[factor_cols])
            try:
                fit = sm.OLS(y, X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": self.nw_lags}
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
            rows.append({
                "date": window.index[-1],
                "alpha_daily": float(fit.params["const"]),
                "alpha_annual": float(fit.params["const"]) * self._periods_per_year,
                "alpha_tstat": float(fit.tvalues["const"]),
                "r_squared": float(fit.rsquared),
            })
        return pd.DataFrame(rows).set_index("date")

    # ------------------------------------------------------------------

    def contribution(self) -> pd.DataFrame:
        """Per-bar return decomposition: portfolio_ret = α + Σ β_i × f_i + ε.

        Columns: ``alpha`` (constant), one column per factor (β_i × f_i,t),
        ``residual`` (ε), ``total`` (sum sanity check).
        """
        if self._result is None:
            self.regress()
        aligned = self._aligned
        factor_cols = [c for c in aligned.columns if c not in ("ret", "rf")]
        df = pd.DataFrame(index=aligned.index)
        df["alpha"] = float(self._result.params["const"])
        for c in factor_cols:
            df[c.upper()] = aligned[c] * float(self._result.params[c])
        df["residual"] = pd.Series(self._result.resid, index=aligned.index)
        df["total"] = df.sum(axis=1)
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _align(self) -> pd.DataFrame:
        """Align equity returns with factor returns on common dates.

        If the equity curve is sampled less frequently than the factor data
        (e.g. weekly equity vs daily factors), the factor returns are
        resampled to the equity frequency by compounding within each period.
        Otherwise α would be inflated by ~5× (weekly) due to mismatched
        return horizons.
        """
        if self._aligned is not None:
            return self._aligned
        rets = self.equity_curve.pct_change(fill_method=None).dropna()
        rets.name = "ret"

        factors = self.factors
        if self._periods_per_year < TRADING_DAYS / 2:
            # Equity is non-daily (weekly / monthly). Compound factor returns
            # within each equity period: r_W = Π(1 + r_daily) - 1.
            freq_alias = self._equity_freq_alias()
            if freq_alias is not None:
                logger.info(
                    "resampling factor returns to %s to match equity frequency",
                    freq_alias,
                )
                factors = (
                    (1 + factors).resample(freq_alias).prod() - 1
                ).dropna()

        merged = pd.concat([rets, factors], axis=1, join="inner").dropna()
        self._aligned = merged
        return merged

    @staticmethod
    def _detect_periods_per_year(equity: pd.Series) -> float:
        """Infer the equity sampling frequency in periods/year.

        Uses the median gap between consecutive index dates. Daily ≈ 252,
        weekly ≈ 52, monthly ≈ 12. Falls back to 252 for irregular data.
        """
        if len(equity) < 2:
            return TRADING_DAYS
        gaps = pd.Series(equity.index).diff().dropna().dt.days
        if gaps.empty:
            return TRADING_DAYS
        median_gap = float(gaps.median())
        if median_gap >= 25:
            return TRADING_MONTHS
        if median_gap >= 4:
            return TRADING_WEEKS
        return TRADING_DAYS

    def _equity_freq_alias(self) -> Optional[str]:
        """Return a pandas resample alias matching the equity frequency."""
        if self._periods_per_year == TRADING_WEEKS:
            # Use the actual weekday of the equity index to avoid losing rows.
            last = self.equity_curve.index[-1]
            return f"W-{last.strftime('%a').upper()[:3]}"
        if self._periods_per_year == TRADING_MONTHS:
            return "ME"
        return None
