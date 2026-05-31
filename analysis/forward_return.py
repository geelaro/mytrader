"""Forward return analysis — how does the market behave AFTER each signal?

Separates "is the signal predictive" from "is the full strategy profitable".
A strategy can be unprofitable (poor exit rules / sizing) while its entry
signal is actually predictive — forward returns reveal that.

Usage
-----
    from analysis.forward_return import compute_forward_returns, summarise

    fr = compute_forward_returns(df_with_signals, horizons=[30, 90, 180], direction=1)
    stats = summarise(fr)
    print(stats.summary())

Definitions
-----------
- A "signal" is a bar where ``df['Signal'] == direction`` (1=buy, -1=sell).
- "Forward return" at horizon H is ``Close[i + H] / Close[i] - 1`` measured
  in **trading days** (rows in the input frame).
- Statistics are computed across all signal occurrences in the input frame,
  excluding signals too close to the end (no forward data available).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [30, 90, 180]


@dataclass
class HorizonStats:
    horizon: int
    n: int
    median: float          # decimal return (0.08 = +8%)
    mean: float
    std: float
    win_rate: float        # fraction with return > 0
    sharpe: float          # mean / std (per-period; not annualised)
    min: float
    max: float
    p25: float
    p75: float


@dataclass
class ForwardReturnStats:
    direction: str         # "long" (buy signals) or "short" (sell signals)
    n_signals: int         # total signals seen (including those clipped at the end)
    horizons: List[HorizonStats] = field(default_factory=list)

    def by_horizon(self, h: int) -> Optional[HorizonStats]:
        for s in self.horizons:
            if s.horizon == h:
                return s
        return None

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"  Forward Return Stats — {self.direction.upper()} signals (n={self.n_signals})",
            "=" * 70,
            f"  {'horizon':>8s} {'n':>5s} {'median':>8s} {'mean':>8s} "
            f"{'std':>8s} {'win%':>6s} {'sharpe':>7s} {'p25':>8s} {'p75':>8s}",
            "  " + "-" * 64,
        ]
        for s in self.horizons:
            lines.append(
                f"  {s.horizon:>7d}d {s.n:>5d} "
                f"{s.median*100:>+7.2f}% {s.mean*100:>+7.2f}% "
                f"{s.std*100:>7.2f}% {s.win_rate*100:>5.0f}% "
                f"{s.sharpe:>+7.2f} {s.p25*100:>+7.2f}% {s.p75*100:>+7.2f}%"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    @property
    def verdict(self) -> str:
        """One-line interpretation of the strongest horizon (highest sharpe)."""
        if not self.horizons:
            return "无信号 / 数据不足"
        # Compare horizons by sharpe absolute value
        best = max(self.horizons, key=lambda s: abs(s.sharpe))
        if best.n < 10:
            return f"样本不足 (n={best.n}, 需要 ≥ 10)"
        if abs(best.sharpe) >= 0.5 and best.win_rate >= 0.55:
            return (
                f"信号有效 — {best.horizon}d Sharpe={best.sharpe:+.2f} "
                f"win_rate={best.win_rate*100:.0f}% median={best.median*100:+.1f}%"
            )
        if best.win_rate >= 0.55:
            return f"边际有效 — {best.horizon}d win_rate={best.win_rate*100:.0f}% 但 Sharpe={best.sharpe:+.2f} 偏低"
        if abs(best.median) < 0.01:
            return "信号无预测力 — median return 接近 0"
        return f"弱信号 — {best.horizon}d win_rate={best.win_rate*100:.0f}%, Sharpe={best.sharpe:+.2f}"


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = None,
    direction: int = 1,
    signal_col: str = "Signal",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute forward returns at each signal occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + ``Signal`` column. Index should be sorted ascending by date.
    horizons : list of int
        Number of trading-day rows to look forward (default ``[30, 90, 180]``).
    direction : int
        Which signal value to filter on. 1 = buy (forward returns are price
        moves); -1 = sell (forward returns are inverted so a price drop
        registers as positive).
    signal_col, price_col : str
        Column names — defaults match the strategy contract.

    Returns
    -------
    pd.DataFrame
        Indexed by signal date, columns are horizons (int), values are decimal
        returns. NaN where the horizon extends past the end of the input.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if signal_col not in df.columns:
        raise KeyError(f"missing column {signal_col!r}")
    if price_col not in df.columns:
        raise KeyError(f"missing column {price_col!r}")

    signal_dates = df.index[df[signal_col] == direction]
    if len(signal_dates) == 0:
        return pd.DataFrame(columns=horizons)

    prices = df[price_col].astype(float)
    rows: list[dict] = []
    for date in signal_dates:
        i = df.index.get_loc(date)
        entry = prices.iloc[i]
        if entry <= 0:
            continue
        row = {}
        for h in horizons:
            j = i + h
            if j >= len(prices):
                row[h] = float("nan")
            else:
                exit_p = prices.iloc[j]
                ret = exit_p / entry - 1.0
                # Invert for short signals so positive = profitable
                if direction == -1:
                    ret = -ret
                row[h] = ret
        rows.append((date, row))

    if not rows:
        return pd.DataFrame(columns=horizons)

    return pd.DataFrame(
        [r for _, r in rows],
        index=pd.DatetimeIndex([d for d, _ in rows]),
    )[horizons]


def summarise(forward_returns: pd.DataFrame, direction: int = 1) -> ForwardReturnStats:
    """Aggregate per-horizon stats from a forward-returns frame."""
    if forward_returns.empty:
        return ForwardReturnStats(
            direction="long" if direction == 1 else "short",
            n_signals=0,
        )

    horizons_out: List[HorizonStats] = []
    for col in forward_returns.columns:
        series = forward_returns[col].dropna()
        if len(series) == 0:
            continue
        std = float(series.std())
        sharpe = float(series.mean() / std) if std > 0 else 0.0
        horizons_out.append(HorizonStats(
            horizon=int(col),
            n=int(len(series)),
            median=float(series.median()),
            mean=float(series.mean()),
            std=std,
            win_rate=float((series > 0).mean()),
            sharpe=sharpe,
            min=float(series.min()),
            max=float(series.max()),
            p25=float(series.quantile(0.25)),
            p75=float(series.quantile(0.75)),
        ))

    return ForwardReturnStats(
        direction="long" if direction == 1 else "short",
        n_signals=len(forward_returns),
        horizons=horizons_out,
    )
