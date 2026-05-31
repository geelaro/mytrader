"""Market state display and today's signals."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from utils import get_logger
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from strategy import SIGNAL_LABEL
from analysis.risk_monitor import compute_risk_state, RiskLevel

logger = get_logger("dashboard.signals")


# Subtle backgrounds — picked so the badge reads at a glance without
# screaming. Foreground stays dark for accessible contrast on all three.
_RISK_THEME = {
    RiskLevel.GREEN:  {"bg": "#d4edda", "border": "#28a745", "fg": "#155724", "label": "低风险"},
    RiskLevel.YELLOW: {"bg": "#fff3cd", "border": "#ffc107", "fg": "#856404", "label": "警戒"},
    RiskLevel.RED:    {"bg": "#f8d7da", "border": "#dc3545", "fg": "#721c24", "label": "高风险"},
}


def render_risk_light(config, target_date, provider):
    """Top-of-dashboard risk light: SPY MA200 + ADX + VIX → green/yellow/red."""
    st.subheader("市场风险灯")

    lookback = config.get("scanner", {}).get("lookback_years", 3)
    start = (pd.Timestamp(target_date) - pd.DateOffset(years=max(lookback, 2))).strftime("%Y-%m-%d")
    end = target_date.isoformat()
    try:
        spy_df = provider.get_daily("SPY", start=start, end=end)
        vix_df = provider.get_daily("^VIX", start=start, end=end)
    except Exception as exc:
        logger.warning("risk light data fetch failed: %s", exc)
        st.info("无法加载 SPY/VIX 数据, 风险灯暂不可用")
        st.divider()
        return

    state = compute_risk_state(spy_df, vix_df)
    theme = _RISK_THEME[state.level]

    # Regime / volatility tag — uses the existing MarketStateClassifier
    # so the dashboard surfaces the same regime info that drives
    # SignalGate / StrategyEnsemble internally. Just a label, no extra
    # numbers (risk light already shows MA200/ADX/VIX).
    regime_tag = ""
    try:
        classifier = MarketStateClassifier(spy_df)
        ms_state = classifier.classify()
        regime_label = {
            MarketRegime.TRENDING_UP: "上升趋势",
            MarketRegime.TRENDING_DOWN: "下降趋势",
            MarketRegime.RANGING: "震荡",
            MarketRegime.TRANSITIONAL: "过渡期",
        }
        vol_label = {Volatility.HIGH: "高波动", Volatility.NORMAL: "正常波动", Volatility.LOW: "低波动"}
        regime_tag = (
            f"{regime_label.get(ms_state.regime, ms_state.regime.name)} × "
            f"{vol_label.get(ms_state.volatility, ms_state.volatility.name)}"
        )
    except Exception as exc:
        logger.debug("regime tag unavailable: %s", exc)

    # Top badge — wide colored card with emoji + label + regime tag
    regime_html = (
        f'<span style="margin-left:18px;font-size:0.95em;opacity:0.9;'
        f'border:1px solid {theme["border"]};padding:2px 10px;border-radius:12px;">'
        f'{regime_tag}</span>'
    ) if regime_tag else ""
    st.markdown(
        f"""
        <div style="background:{theme['bg']};border-left:6px solid {theme['border']};
                    padding:14px 20px;border-radius:8px;margin-bottom:12px;
                    color:{theme['fg']};">
          <span style="font-size:1.6em;font-weight:bold;">{state.emoji} {theme['label']}</span>
          {regime_html}
          <span style="margin-left:18px;font-size:0.85em;opacity:0.75;">
            截至 {state.indicators.get('as_of', '?')}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Indicator row
    ind = state.indicators
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "SPY",
        f"${ind.get('spy_close', 0):.2f}",
        f"{ind.get('spy_vs_ma200_pct', 0):+.1f}% vs MA200",
    )
    c2.metric("SPY 5日", f"{ind.get('spy_5d_return_pct', 0):+.2f}%")
    c3.metric("SPY ADX", f"{ind.get('spy_adx', 0):.1f}")
    c4.metric(
        "VIX",
        f"{ind.get('vix', 0):.2f}",
        delta=None,
        help="<18 低波, 18-30 偏高, >30 极端恐慌",
    )

    # Reason bullets
    if state.reasons:
        st.markdown("**判定依据**")
        for r in state.reasons:
            st.markdown(f"- {r}")

    st.divider()


def render_market_state(config, target_date, provider):
    ms_cfg = config.get("market_state", {})
    if not ms_cfg.get("enabled", False):
        return

    st.subheader("市场状态")

    proxy_sym = ms_cfg.get("proxy_symbol", "SPY")
    lookback = config.get("scanner", {}).get("lookback_years", 3)
    ms_start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback)).strftime("%Y-%m-%d")
    ms_df = provider.get_daily(proxy_sym, start=ms_start, end=target_date.isoformat())

    if ms_df is not None and not ms_df.empty:
        classifier = MarketStateClassifier(ms_df)
        state = classifier.classify()

        regime = state.regime
        vol = state.volatility
        regime_color = {
            MarketRegime.TRENDING_UP: "#2ca02c",
            MarketRegime.TRENDING_DOWN: "#d62728",
            MarketRegime.RANGING: "#ff7f0e",
            MarketRegime.TRANSITIONAL: "#7f7f7f",
        }
        regime_label = {
            MarketRegime.TRENDING_UP: "上升趋势",
            MarketRegime.TRENDING_DOWN: "下降趋势",
            MarketRegime.RANGING: "震荡",
            MarketRegime.TRANSITIONAL: "过渡期",
        }
        vol_label = {Volatility.HIGH: "高波动", Volatility.NORMAL: "正常", Volatility.LOW: "低波动"}

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("市场状态", regime_label.get(regime, regime.name))
        c2.metric("波动率", vol_label.get(vol, vol.name))
        c3.metric("ADX", f"{state.adx:.1f}")
        c4.metric("MA20", f"{state.ma20:.2f}")
        c5.metric("MA200", f"{state.ma200:.2f}")
        c6.metric("BB带宽分位", f"{state.bb_width_pct:.0f}%")

        st.markdown(
            f"<span style='display:inline-block;padding:4px 12px;border-radius:8px;"
            f"background:{regime_color.get(regime, '#7f7f7f')};color:white;font-weight:bold;'>"
            f"{regime_label.get(regime, regime.name)} × {vol_label.get(vol, vol.name)}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"市场状态: {proxy_sym} 数据不可用")

    st.divider()


def render_todays_signals(config, target_date, provider, cache):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"今日信号 ({target_date})")

        from daily import scan_day

        results = scan_day(config, target_date=target_date.isoformat(),
                           provider=provider, cache=cache)

        active_signals = [r for r in results if r["signal"] != 0]
        if active_signals:
            grouped: dict[str, list] = defaultdict(list)
            for r in active_signals:
                grouped[r["symbol"]].append(r)

            for sym, sigs in sorted(grouped.items()):
                parts = []
                has_buy = any(s["signal"] == 1 for s in sigs)
                has_sell = any(s["signal"] == -1 for s in sigs)
                price = sigs[0]["price"]
                for s in sigs:
                    tag = "★" if s["strategy"] == next(
                        (item.get("active", "") for item in config.get("watchlist", [])
                         if item["symbol"] == sym), ""
                    ) else ""
                    parts.append(f"{s['strategy']}{tag} {SIGNAL_LABEL[s['signal']]}")
                line = f"{sym} @ ${price:.2f}:  " + "  |  ".join(parts)
                if has_sell:
                    st.error(line)
                elif has_buy:
                    st.success(line)
                else:
                    st.info(line)
        else:
            st.info("今日无买入/卖出信号")

    with col2:
        st.subheader("策略分布")
        active_counts: dict[str, int] = {}
        for item in config.get("watchlist", []):
            a = item.get("active", "-")
            active_counts[a] = active_counts.get(a, 0) + 1

        for strat, count in sorted(active_counts.items(), key=lambda x: -x[1]):
            st.metric(label=strat, value=f"{count} 个标的")

        st.metric(label="总组合数", value=f"{len(results)} 策略×标的")
