"""Dashboard tab — generate & preview the weekly risk report.

Shows the same Markdown the cron job would push to Feishu, plus a button
to actually push it now.
"""

from __future__ import annotations

import logging
from datetime import date

import streamlit as st

from analysis.risk_report import RiskReport

logger = logging.getLogger(__name__)


def render_risk_report(config, target_date, provider, cache):
    st.header("风险报告")
    st.caption(
        "汇总今日所有分析模块输出 (Risk Light / VaR / EVT / Stress / "
        "Concentration / Correlation / Risk Decomp / Brinson / PnL / "
        "Drawdown) 成一份周报. 可手动生成 / 推飞书 / 或交给 cron 跑 "
        "`scripts/weekly_risk_report.py`."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        report_date = st.date_input(
            "报告日期",
            value=target_date if isinstance(target_date, date) else date.today(),
        )
    with col2:
        do_build = st.button("生成报告", type="primary")

    # Persist the built report across Streamlit reruns — every button click
    # re-enters this function, so we can't rely on a local variable.  Without
    # this, the "推送到飞书" button always early-returns because do_build is
    # False on the rerun triggered by the send button.
    if do_build:
        with st.spinner("聚合所有分析模块..."):
            report = RiskReport(config, provider, cache,
                                target_date=report_date)
            st.session_state["_rr_report"] = report
            st.session_state["_rr_data"] = report.build()
            st.session_state["_rr_md"] = report.to_markdown()

    if "_rr_report" not in st.session_state:
        st.info("点 '生成报告' 后会聚合所有分析模块,可能需要几十秒。")
        return

    report = st.session_state["_rr_report"]
    data = st.session_state["_rr_data"]
    md = st.session_state["_rr_md"]

    # ── Inline render ──────────────────────────────────────────────
    st.subheader(f"风险报告 — {data['as_of']}")
    st.caption(f"组合规模: {data['watchlist_size']} 个 watchlist 标的")

    for sec in data["sections"]:
        with st.expander(sec.title, expanded=True):
            if sec.summary:
                st.markdown(f"_{sec.summary}_")
            cols_per_row = 3
            metric_items = list(sec.metrics.items())
            if metric_items:
                rows = [metric_items[i:i + cols_per_row]
                        for i in range(0, len(metric_items), cols_per_row)]
                for row in rows:
                    cols = st.columns(cols_per_row)
                    for col, (k, v) in zip(cols, row):
                        col.metric(k, str(v))
            for table in sec.tables:
                if table.get("title"):
                    st.markdown(f"**{table['title']}**")
                for row in table.get("rows", []):
                    st.markdown(f"- {row}")
            for w in sec.warnings:
                st.warning(w)

    st.divider()

    # ── Markdown copy ──────────────────────────────────────────────
    with st.expander("📋 Markdown 全文 (可复制 / 邮件)", expanded=False):
        st.code(md, language="markdown")

    # ── Send to Feishu ─────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        do_send = st.button("📤 推送到飞书", type="secondary")
    with col2:
        do_clear = st.button("清空", type="secondary",
                             help="清除已生成的报告缓存")
    if do_clear:
        for key in ("_rr_report", "_rr_data", "_rr_md"):
            st.session_state.pop(key, None)
        st.rerun()
    if do_send:
        from utils.notify import Notifier
        nf = Notifier(async_mode=False)
        if not nf.available:
            st.error("飞书未配置 (FEISHU_WEBHOOK 或 FEISHU_APP_* 缺失)")
            return
        card = report.to_feishu_card()
        ok = nf._send({"msg_type": "interactive", "card": card})
        if ok:
            st.success("飞书已发送 ✓")
        else:
            st.error("飞书发送失败,看 logs")
