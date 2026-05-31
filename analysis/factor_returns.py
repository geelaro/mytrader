"""ETF-proxy factor returns for attribution analysis.

We use liquid ETFs as proxies for the standard Fama-French / AQR factors so
the data pipeline can stay inside the existing ``DataProvider`` (no academic
data subscription needed):

    Factor   Proxy formula                Inception   Concept
    -------  ---------------------------  ----------  -------------------------
    MKT      SPY - SHV                    2007        Market excess return
    SMB      IWM - SPY                    2000        Small-minus-big (size)
    HML      IVE - IVW                    2000        High-minus-low (value)
    MOM      MTUM - SPY                   2013-05     Cross-sectional momentum
    QMJ      QUAL - SPY                   2013-07     Quality-minus-junk
    BAB      USMV - SPY                   2011-10     Betting-against-beta (low vol)

The MOM / QMJ / BAB factors limit the full-6 factor set to ``>= 2013-08``.
Use ``mode="ff3"`` for the MKT/SMB/HML subset if you need history back to 2000.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data import DataProvider

logger = logging.getLogger(__name__)


# Earliest inception across the requested factor set — anything before this
# date will return NaN rows that get dropped.
_INCEPTION_FULL = pd.Timestamp("2013-08-01")
_INCEPTION_FF3 = pd.Timestamp("2000-06-01")

# ETF tickers used as factor proxies. Kept here so callers can override or
# extend (e.g. add sector ETFs) without touching the regression code.
_ETF_TICKERS_FULL = ["SPY", "IWM", "IVE", "IVW", "MTUM", "QUAL", "USMV", "SHV"]
_ETF_TICKERS_FF3 = ["SPY", "IWM", "IVE", "IVW", "SHV"]


@dataclass(frozen=True)
class FactorSpec:
    """Definition of one factor: name + proxy ETF tickers + sign."""

    name: str
    long_etf: str
    short_etf: str  # set equal to long_etf and skip subtraction for raw series


# (factor_name, long_etf, short_etf) — short_etf == long_etf means "raw return"
_FACTOR_DEFS_FULL = [
    ("MKT", "SPY", "SHV"),   # market excess
    ("SMB", "IWM", "SPY"),   # small minus big
    ("HML", "IVE", "IVW"),   # value minus growth
    ("MOM", "MTUM", "SPY"),  # momentum minus market
    ("QMJ", "QUAL", "SPY"),  # quality minus market
    ("BAB", "USMV", "SPY"),  # low-vol minus market
]
_FACTOR_DEFS_FF3 = _FACTOR_DEFS_FULL[:3]


class FactorReturns:
    """Build daily factor return series from ETF proxies.

    Parameters
    ----------
    mode : str
        ``"full"`` for the 6-factor set (MKT/SMB/HML/MOM/QMJ/BAB), requires
        data from 2013-08 onward. ``"ff3"`` for the 3-factor subset
        (MKT/SMB/HML), available back to 2000.
    provider : DataProvider, optional
        Inject a configured DataProvider (e.g. with a shared cache).
    """

    def __init__(self, mode: str = "full", provider: Optional[DataProvider] = None):
        if mode not in ("full", "ff3"):
            raise ValueError("mode must be 'full' or 'ff3'")
        self.mode = mode
        self.provider = provider or DataProvider()
        self._defs = _FACTOR_DEFS_FULL if mode == "full" else _FACTOR_DEFS_FF3
        self._tickers = _ETF_TICKERS_FULL if mode == "full" else _ETF_TICKERS_FF3
        self._cache: Optional[pd.DataFrame] = None

    @property
    def factor_names(self) -> list[str]:
        return [d[0] for d in self._defs]

    @property
    def inception(self) -> pd.Timestamp:
        return _INCEPTION_FULL if self.mode == "full" else _INCEPTION_FF3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, start: str, end: str) -> pd.DataFrame:
        """Return daily factor returns + risk-free rate.

        Columns: lowercased factor names + ``rf``. Index: business-day dates.
        Missing dates (any ETF) are dropped.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts < self.inception:
            logger.warning(
                "factor mode=%s requires start >= %s; clipping from %s",
                self.mode, self.inception.date(), start_ts.date(),
            )
            start_ts = self.inception

        rets = self._fetch_etf_returns(start_ts.strftime("%Y-%m-%d"), end)
        if rets.empty:
            return pd.DataFrame()

        # Risk-free daily rate: SHV total return ≈ T-bill yield / 252.
        # (SHV is a 1-month T-bill ETF; its daily return approximates rf_t.)
        rf = rets["SHV"]

        # Build factor columns by long-minus-short. Market factor subtracts rf.
        factor_df = pd.DataFrame(index=rets.index)
        for name, long_etf, short_etf in self._defs:
            if long_etf not in rets.columns or short_etf not in rets.columns:
                logger.warning("missing ETF for factor %s: %s/%s — skipped",
                               name, long_etf, short_etf)
                continue
            factor_df[name.lower()] = rets[long_etf] - rets[short_etf]
        factor_df["rf"] = rf

        # Drop rows where any factor is NaN — usually the first few bars
        # after an ETF inception that hasn't aligned with the others yet.
        factor_df = factor_df.dropna()
        return factor_df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_etf_returns(self, start: str, end: str) -> pd.DataFrame:
        """Pull each ETF's daily close and convert to simple returns."""
        closes: dict[str, pd.Series] = {}
        for ticker in self._tickers:
            try:
                df = self.provider.get_daily(ticker, start=start, end=end)
            except Exception:
                logger.exception("provider failed for %s", ticker)
                continue
            if df is None or df.empty:
                logger.warning("no data for factor proxy ETF %s — factor "
                               "values depending on it will be skipped", ticker)
                continue
            closes[ticker] = df["Close"]

        if not closes:
            return pd.DataFrame()

        prices = pd.DataFrame(closes).sort_index()
        # Forward-fill at most 1 bar to handle isolated holiday gaps between
        # ETFs that trade on different exchanges. Don't ffill aggressively —
        # that would mask true missing-data problems.
        prices = prices.ffill(limit=1)
        rets = prices.pct_change(fill_method=None).dropna(how="all")
        return rets
