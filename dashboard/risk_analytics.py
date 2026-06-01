"""Dashboard tab — VaR / Stress / Concentration in one view.

Pulls together :mod:`analysis.var`, :mod:`analysis.stress`,
:mod:`analysis.concentration` so the user sees a one-page risk picture
of the *current* portfolio.

Portfolio source priority
-------------------------
1. Hypothetical positions from :func:`live.position_stops.compute_hypothetical_positions`
   (what active strategies "would" be holding right now).  Equal-weighted.
2. Fallback: equal-weight every symbol in the watchlist.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import streamlit as st

from analysis.concentration import concentration_summary, hhi_label
from analysis.stress import SCENARIOS, run_scenarios
from analysis.var import portfolio_returns, var_summary
from live.position_stops import compute_hypothetical_positions
from utils.sectors import DEFAULT_SECTORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render_risk_analytics(config: dict, target_date, provider):
    """Render the VaR / Stress / Concentration view."""
    st.header("风险量化")
    st.caption(
        "对**当前组合**(假设持仓 或 等权 watchlist)计算 VaR / "
        "ES / 历史压力测试 / 集中度. 用于回答\"99% 信心明天最多亏多少 %\""
        "和\"如果 2008 重演会怎样\"."
    )

    # ── Portfolio construction ─────────────────────────────────────
    col_src, col_lookback = st.columns([2, 1])
    source = col_src.radio(
        "持仓来源",
        ["假设持仓(active 策略 + 未平 buy 信号)", "等权 watchlist"],
        horizontal=True,
    )
    lookback_years = col_lookback.number_input(
        "VaR 回看年数", min_value=1, max_value=15, value=5, step=1,
    )

    weights = _build_weights(config, target_date, provider, source)
    if not weights:
        st.warning("当前无可用持仓 — active 策略均已平仓, 或 watchlist 为空")
        return

    st.write(f"组合持仓: **{len(weights)}** 个标的, 等权")

    # ── Concentration ──────────────────────────────────────────────
    st.subheader("集中度")
    concentration = concentration_summary(
        weights, sector_map=DEFAULT_SECTORS,
    )
    cols = st.columns(4)
    cols[0].metric("持仓数", concentration["n_holdings"])
    cols[1].metric(
        "HHI",
        f"{concentration['hhi']:.0f}",
        help=f"{concentration['hhi_label']} (<1500 分散 / <2500 中等 / ≥2500 高)",
    )
    cols[2].metric("有效持仓数", f"{concentration['effective_n']:.1f}")
    cols[3].metric("Top-3 占比", f"{concentration['top_3_weight'] * 100:.1f}%")

    sec_h = concentration.get("sector_hhi")
    if sec_h is not None:
        st.write(
            f"**行业 HHI**: {sec_h:.0f}  ({hhi_label(sec_h)}) — "
            f"行业 HHI 远大于 symbol HHI 说明\"看起来分散但只买了一个因子\""
        )
        exposure = concentration.get("sector_exposure", {})
        if exposure:
            exp_df = pd.DataFrame([
                {"行业": k, "占比": f"{v * 100:.1f}%", "_weight": v}
                for k, v in exposure.items()
            ])
            st.dataframe(
                exp_df[["行业", "占比"]],
                hide_index=True, use_container_width=True,
            )

    st.divider()

    # ── VaR / ES ───────────────────────────────────────────────────
    st.subheader("VaR / 期望损失 (ES)")
    prices = _fetch_prices(weights.keys(), target_date, provider, lookback_years)
    pf_ret = portfolio_returns(prices, weights)
    if pf_ret.empty:
        st.warning("无法计算 VaR — 历史价格数据不足")
    else:
        summary = var_summary(pf_ret)
        st.caption(
            f"基于 {summary['n_obs']} 个交易日, "
            f"日均 {summary['mean'] * 100:+.3f}%, "
            f"波动 {summary['std'] * 100:.2f}%. 所有数值为单日损失."
        )
        rows = []
        for conf in ("95%", "99%"):
            metrics = summary[conf]
            rows.append({
                "置信度": conf,
                "Historical VaR": f"{metrics['historical'] * 100:.2f}%",
                "Parametric VaR": f"{metrics['parametric'] * 100:.2f}%",
                "Expected Shortfall (CVaR)": f"{metrics['cvar'] * 100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "解读: Historical 是经验百分位, Parametric 是正态假设, "
            "ES 是\"穿透 VaR 之后平均亏多少\". 三者差距大说明分布有肥尾."
        )

    st.divider()

    # ── Stress Scenarios ───────────────────────────────────────────
    st.subheader("历史场景压力测试")
    # Need long history for 2008/2018/2015. Fetch ≥ 18 years.
    stress_prices = _fetch_prices(weights.keys(), target_date, provider, years=18)
    if stress_prices.empty:
        st.warning("历史价格数据不足以跑场景测试")
    else:
        results = run_scenarios(stress_prices, weights)
        rows = []
        for sid, r in results.items():
            cfg = SCENARIOS[sid]
            ret = r["return_pct"]
            dd = r["max_dd_pct"]
            missing = len(r.get("missing_symbols", []))
            n_used = len(r.get("by_symbol", {}))
            rows.append({
                "场景": cfg["name"],
                "区间": f"{cfg['start']} ~ {cfg['end']}",
                "SPY 当时": f"{cfg['spy_pct']:+.1f}%",
                "组合收益": f"{ret:+.2f}%" if pd.notna(ret) else "数据不足",
                "最大回撤": f"{dd:.2f}%" if pd.notna(dd) else "—",
                "覆盖": f"{n_used}/{n_used + missing}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "解读: 把**当前等权组合**应用到当时的标的日收益序列, "
            "假设当时就持有这个组合. 数据不足的场景说明部分标的当时还没上市. "
            "Daily-rebalanced 收益, 与 buy-and-hold 略有差异."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_weights(config, target_date, provider, source: str) -> dict:
    """Construct {symbol: weight} based on user selection."""
    if "假设持仓" in source:
        rows = compute_hypothetical_positions(config, target_date, provider)
        if rows:
            return {r["symbol"]: 1.0 for r in rows}
        # fall through to watchlist if no open positions
    return {item["symbol"]: 1.0 for item in config.get("watchlist", [])}


def _fetch_prices(symbols, target_date, provider, years: int) -> pd.DataFrame:
    """Build a date×symbol Close DataFrame for the requested lookback."""
    end = pd.Timestamp(target_date) if not isinstance(target_date, pd.Timestamp) else target_date
    start = (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    series = {}
    for sym in symbols:
        try:
            df = provider.get_daily(sym, start=start, end=end_str)
        except Exception as exc:
            logger.debug("risk_analytics fetch failed for %s: %s", sym, exc)
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        series[sym] = df["Close"]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()
