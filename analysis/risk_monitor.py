"""Market risk light — green / yellow / red based on SPY MA200 + ADX + VIX.

Designed as a glance-and-go indicator at the top of the dashboard. Keep the
logic intentionally simple: 3 inputs, explicit thresholds, human-readable
reasons. No optimisation, no fitting — these levels are conventions, not
backtested.

Thresholds
----------
GREEN  (low risk):
    SPY > MA200 by ≥ 3%  AND  SPY ADX > 25  AND  VIX < 18

RED    (danger):
    SPY < MA200 by ≥ 3%   OR   VIX > 30   OR   SPY 5-day return < -5%

YELLOW (caution):
    everything else — typically SPY near MA200 (±3%), or ADX 15-25 (trend
    weakening), or VIX 18-30 (elevated fear).

Output
------
Each compute_risk_state() call returns a :class:`RiskState` with:
- ``level`` — RiskLevel enum (GREEN / YELLOW / RED)
- ``reasons`` — list of human-readable bullets explaining the call
- ``indicators`` — current numeric values (SPY/MA200/ADX/VIX/5d return)

Usage
-----
    from analysis.risk_monitor import compute_risk_state
    state = compute_risk_state(spy_df, vix_df)
    print(state.level.value, state.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd

from strategy.base import compute_adx

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Tuned for US equity markets 2018-2026 regime. Override via params if needed
# for other markets / regimes (e.g. crypto vol baseline is much higher).
_DEFAULTS = {
    "ma200_period": 200,
    "ma_buffer_pct": 0.03,        # ±3% band around MA200 → yellow zone
    "adx_strong": 25.0,
    "adx_weak": 15.0,
    "vix_low": 18.0,
    "vix_high": 30.0,
    "crash_5d_pct": -5.0,         # SPY drops > 5% in 5 days → instant red
}


@dataclass
class RiskState:
    """Snapshot of the market risk light."""

    level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return {RiskLevel.GREEN: "🟢", RiskLevel.YELLOW: "🟡", RiskLevel.RED: "🔴"}[self.level]

    def summary(self) -> str:
        """One-line summary plus reason bullets."""
        head = f"{self.emoji} {self.level.value.upper()}"
        if not self.reasons:
            return head
        body = "\n".join(f"  - {r}" for r in self.reasons)
        return f"{head}\n{body}"


def compute_risk_state(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    params: Optional[dict] = None,
    realtime_vix: Optional[float] = None,
) -> RiskState:
    """Compute the current risk light from SPY and VIX OHLCV.

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY daily OHLCV. Must include at least ``ma200_period + 30`` bars
        for ADX warm-up.
    vix_df : pd.DataFrame
        VIX daily OHLCV (Close used). Trading-day index aligned with SPY
        is preferred; mismatched dates are resolved by forward-fill.
    params : dict, optional
        Override default thresholds.
    realtime_vix : float, optional
        Live VIX value (e.g. from :func:`data.realtime.get_realtime_vix`).
        When provided, overrides the EOD value from ``vix_df`` for the
        RED/GREEN threshold check.  Use during US trading hours so the
        risk light reflects intraday spikes rather than yesterday's close.
        The ``vix`` indicator in the returned RiskState is the realtime
        value when this override is active.

    Returns
    -------
    RiskState
    """
    p = {**_DEFAULTS, **(params or {})}

    if spy_df is None or spy_df.empty:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=["SPY 数据缺失, 默认黄"])
    if vix_df is None or vix_df.empty:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=["VIX 数据缺失, 默认黄"])

    spy = spy_df.copy()
    if len(spy) < p["ma200_period"] + 20:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=[f"SPY 数据不足 {p['ma200_period']+20} 根, 无法判定"])

    # Indicators
    spy["MA200"] = spy["Close"].rolling(p["ma200_period"]).mean()
    spy_adx = compute_adx(spy.copy(), 14)  # adds ADX column in place
    last = spy.iloc[-1]
    spy_close = float(last["Close"])
    ma200 = float(last["MA200"])
    adx = float(spy_adx["ADX"].iloc[-1]) if "ADX" in spy_adx.columns else 0.0
    ma_pct = (spy_close / ma200 - 1) * 100 if ma200 > 0 else 0.0

    # SPY 5-day return
    if len(spy) >= 6:
        ret_5d = (spy_close / float(spy["Close"].iloc[-6]) - 1) * 100
    else:
        ret_5d = 0.0

    # VIX latest — realtime override takes precedence if supplied,
    # otherwise forward-fill onto SPY's last date from the EOD frame.
    if realtime_vix is not None and realtime_vix > 0:
        vix_now = float(realtime_vix)
        vix_source = "realtime"
    else:
        vix_aligned = vix_df["Close"].reindex(spy.index, method="ffill")
        vix_now = float(vix_aligned.iloc[-1]) if not vix_aligned.empty else float(vix_df["Close"].iloc[-1])
        vix_source = "eod"

    indicators = {
        "spy_close": round(spy_close, 2),
        "spy_ma200": round(ma200, 2),
        "spy_vs_ma200_pct": round(ma_pct, 2),
        "spy_adx": round(adx, 1),
        "spy_5d_return_pct": round(ret_5d, 2),
        "vix": round(vix_now, 2),
        "vix_source": vix_source,
        "as_of": spy.index[-1].date().isoformat(),
    }

    # ── RED triggers (any one) ─────────────────────────────────────
    red_reasons: List[str] = []
    if ma_pct < -p["ma_buffer_pct"] * 100:
        red_reasons.append(
            f"SPY 跌破 MA200 {abs(ma_pct):.1f}% (阈值 -{p['ma_buffer_pct']*100:.0f}%)"
        )
    if vix_now > p["vix_high"]:
        red_reasons.append(f"VIX={vix_now:.1f} > {p['vix_high']:.0f} (极端恐慌)")
    if ret_5d < p["crash_5d_pct"]:
        red_reasons.append(
            f"SPY 5日跌幅 {ret_5d:.1f}% < {p['crash_5d_pct']:.0f}% (闪崩信号)"
        )

    if red_reasons:
        return RiskState(level=RiskLevel.RED, reasons=red_reasons, indicators=indicators)

    # ── GREEN check (all three) ────────────────────────────────────
    green_ok = (
        ma_pct >= p["ma_buffer_pct"] * 100
        and adx >= p["adx_strong"]
        and vix_now < p["vix_low"]
    )
    if green_ok:
        return RiskState(
            level=RiskLevel.GREEN,
            reasons=[
                f"SPY 站上 MA200 {ma_pct:+.1f}%",
                f"ADX={adx:.1f} (强趋势, > {p['adx_strong']:.0f})",
                f"VIX={vix_now:.1f} (低波, < {p['vix_low']:.0f})",
            ],
            indicators=indicators,
        )

    # ── YELLOW (default) ───────────────────────────────────────────
    yellow_reasons: List[str] = []
    if abs(ma_pct) < p["ma_buffer_pct"] * 100:
        yellow_reasons.append(
            f"SPY 接近 MA200 ({ma_pct:+.1f}%, ±{p['ma_buffer_pct']*100:.0f}% 缓冲区内)"
        )
    elif ma_pct >= p["ma_buffer_pct"] * 100:
        yellow_reasons.append(f"SPY 高于 MA200 {ma_pct:+.1f}%, 但其它条件未达绿")

    if p["adx_weak"] <= adx < p["adx_strong"]:
        yellow_reasons.append(f"ADX={adx:.1f} 趋势衰竭 ({p['adx_weak']:.0f}-{p['adx_strong']:.0f})")
    elif adx < p["adx_weak"]:
        yellow_reasons.append(f"ADX={adx:.1f} 无趋势 (< {p['adx_weak']:.0f})")

    if p["vix_low"] <= vix_now <= p["vix_high"]:
        yellow_reasons.append(f"VIX={vix_now:.1f} 偏高 ({p['vix_low']:.0f}-{p['vix_high']:.0f})")

    if not yellow_reasons:
        yellow_reasons.append("综合判定: 介于绿与红之间")

    return RiskState(level=RiskLevel.YELLOW, reasons=yellow_reasons, indicators=indicators)
