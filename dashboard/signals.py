"""Market state display and today's signals."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from utils import get_logger
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from strategy import SIGNAL_LABEL

logger = get_logger("dashboard.signals")


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
