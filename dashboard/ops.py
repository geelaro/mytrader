"""Ops health, sector chart, and live trade history."""

from collections import Counter

import streamlit as st

import utils
from utils import get_logger
from utils.sectors import get_sector

import matplotlib.pyplot as plt

logger = get_logger("dashboard.ops")


def render_ops(config, cache):
    _render_sector_chart(config)
    _render_live_trades(cache)
    _render_ops_health(cache)


def _render_sector_chart(config):
    with st.expander("持仓行业分布", expanded=False):
        watchlist_syms = [item["symbol"] for item in config.get("watchlist", [])]
        sector_counts = Counter()
        for sym in watchlist_syms:
            sector_counts[get_sector(sym)] += 1

        if sector_counts:
            sector_symbols: dict[str, list] = {}
            for sym in watchlist_syms:
                sec = get_sector(sym)
                sector_symbols.setdefault(sec, []).append(sym)

            labels = list(sector_counts.keys())
            sizes = list(sector_counts.values())
            colors = plt.cm.Set3(range(len(labels)))
            fig, ax = plt.subplots(figsize=(4, 3))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90, pctdistance=0.6,
            )
            for t in autotexts:
                t.set_fontsize(8)
            ax.set_title("标的行业分布", fontsize=10)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.pyplot(fig)

            for sec in sorted(sector_symbols.keys()):
                st.caption(f"{sec}: {', '.join(sector_symbols[sec])}")
        else:
            st.info("无数据")


def _render_live_trades(cache):
    with st.expander("实盘交易记录", expanded=False):
        try:
            trades = cache.query_trade_pnl(limit=30)
        except AttributeError:
            st.info("请重启 Streamlit 加载最新代码")
            trades = []
        if trades:
            rows = []
            for t in trades:
                rows.append({
                    "标的": t["symbol"], "方向": t["side"], "数量": t["qty"],
                    "买入价": f"${t['entry_price']:.2f}",
                    "卖出价": f"${t['exit_price']:.2f}",
                    "PnL": f"${t['pnl']:+,.0f}",
                    "PnL%": f"{t['pnl_pct']:+.1f}%",
                    "日期": t["exit_date"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            total_pnl = sum(t["pnl"] for t in trades)
            st.metric("合计 PnL", f"${total_pnl:+,.0f}")
        else:
            st.info("暂无成交记录")


def _render_ops_health(cache):
    with st.expander("运行健康", expanded=False):
        try:
            ops = cache.conn.execute(
                "SELECT ts, source, level, event, symbol, detail, value FROM ops_log ORDER BY ts DESC LIMIT 50"
            ).fetchall()
            reject_24h = cache.conn.execute(
                "SELECT COUNT(*) FROM ops_log WHERE event IN ('gate_reject','risk_reject') "
                "AND ts >= datetime('now','localtime','-24 hours')"
            ).fetchone()[0]
            total_24h = cache.conn.execute(
                "SELECT COUNT(*) FROM ops_log WHERE ts >= datetime('now','localtime','-24 hours')"
            ).fetchone()[0]
        except Exception:
            ops = []
            reject_24h = 0
            total_24h = 0

        if ops:
            pause_count = sum(1 for o in ops if o[3] in ("gate_reject", "risk_reject"))
            slip_count = sum(1 for o in ops if "slippage" in (o[3] or ""))
            reject_rate = f"{reject_24h}/{total_24h}" if total_24h > 0 else "—"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("拦截事件", pause_count)
            c2.metric("滑点事件", slip_count)
            c3.metric("24h 拒单率", reject_rate)
            c4.metric("总事件", len(ops))

            weekly_counts = Counter(o[3] for o in ops)
            fig, ax = plt.subplots(figsize=(4, 2))
            labels = list(weekly_counts.keys())
            counts = list(weekly_counts.values())
            event_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"][:len(labels)]
            bars = ax.bar(range(len(labels)), counts, color=event_colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_title("事件分布", fontsize=9)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(c), ha="center", va="bottom", fontsize=8)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.pyplot(fig)

            rows = []
            for o in ops:
                source = o[1] or ""
                level = o[2] or "INFO"
                event = o[3]
                symbol = o[4] or ""
                detail = o[5] or ""
                value = o[6] or 0
                rows.append({
                    "时间": o[0], "来源": source, "级别": level,
                    "事件": event, "标的": symbol, "详情": detail, "值": value,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("暂无运行事件")
