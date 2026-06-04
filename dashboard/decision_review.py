"""Decision review — let the user audit past trading decisions in context.

What this answers
-----------------
- 过去 N 天我做了多少决策？买多少卖多少？
- RED 灯下的决策 vs GREEN 灯下的决策, 各自有多少？
- Kill Switch 触发过几次？为什么？
- 每个决策当时的组合规模、持仓数 — 是否符合我的决策框架？

Source of truth
---------------
``decision_history`` table — populated by :class:`live.decision_logger.DecisionLogger`
on every order fill, kill-switch trigger, rebalance, etc.  This tab is
read-only — it never writes back.

Outcomes (TODO — future work)
-----------------------------
A follow-up enhancement will join ``decision_history.payload.order_id``
to ``trade_pnl.order_id`` so each decision row can show its realised
N-day PnL — turning the audit into a hit-rate analysis under each
risk-light regime.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


_RISK_LIGHT_EMOJI = {
    "green": "🟢", "yellow": "🟡", "red": "🔴", None: "⚪", "": "⚪",
}

_TYPE_LABEL = {
    "trade_buy":        "买入",
    "trade_sell":       "卖出",
    "kill_switch":      "Kill Switch",
    "rebalance":        "调仓",
    "manual_override":  "人工覆写",
    "signal_ignored":   "信号忽略",
}


def render_decision_review(cache):
    """Render the decision-review tab.

    Pulls decisions from the cache, lets the user filter by time window,
    decision type, and risk-light regime; shows distribution stats +
    detail table.
    """
    st.header("决策复盘")
    st.caption(
        "审查过去的交易决策 — 每次下单 / Kill Switch / 调仓时, 系统都会"
        "快照当时的风险灯 / 持仓数 / 组合规模. 用于回答 \"我在 RED 灯下"
        "的决策成功率 vs GREEN 灯下\" 这类问题."
    )

    # ── Filters ────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)
    days = fc1.selectbox("时间窗", [7, 30, 90, 180, 365], index=2)
    type_options = ["全部", "trade_buy", "trade_sell", "kill_switch",
                    "rebalance", "manual_override", "signal_ignored"]
    type_choice = fc2.selectbox("决策类型", type_options, index=0)
    light_choice = fc3.selectbox("风险灯", ["全部", "green", "yellow", "red"],
                                 index=0)
    symbol_filter = fc4.text_input("标的过滤", value="",
                                   placeholder="可选, e.g. NVDA").strip().upper()

    decisions = cache.query_decisions(
        days=int(days),
        decision_type=None if type_choice == "全部" else type_choice,
        symbol=symbol_filter if symbol_filter else None,
        risk_light=None if light_choice == "全部" else light_choice,
        limit=2000,
    )

    if not decisions:
        st.info(
            f"过去 {days} 天内无符合条件的决策记录. "
            "实盘 / 模拟运行后, 每次下单会自动登记."
        )
        return

    # ── Summary metrics ────────────────────────────────────────────
    df = pd.DataFrame(decisions)
    n_total = len(df)
    n_buy = int((df["decision_type"] == "trade_buy").sum())
    n_sell = int((df["decision_type"] == "trade_sell").sum())
    n_kill = int((df["decision_type"] == "kill_switch").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总决策数", n_total)
    m2.metric("买入", n_buy)
    m3.metric("卖出", n_sell)
    m4.metric("Kill Switch", n_kill,
              delta="⚠ 关注" if n_kill > 0 else None,
              delta_color="inverse" if n_kill > 0 else "off")

    # ── Distribution by risk light ─────────────────────────────────
    st.subheader("决策分布 — 风险灯")
    if df["risk_light"].notna().any():
        by_light = (df.groupby(["risk_light", "decision_type"])
                    .size()
                    .unstack(fill_value=0))
        # Order rows green → yellow → red for readability
        order = [k for k in ["green", "yellow", "red"] if k in by_light.index]
        by_light = by_light.reindex(order) if order else by_light
        by_light.index = [f"{_RISK_LIGHT_EMOJI.get(k, '⚪')} {k}" for k in by_light.index]
        st.dataframe(by_light, use_container_width=True)
    else:
        st.caption("风险灯字段未填充 — RiskAlerter 尚未在持久化 alert:last_risk_level")

    # ── Detail table ───────────────────────────────────────────────
    st.subheader("决策明细")
    display_rows = []
    for r in decisions:
        light = r.get("risk_light") or ""
        display_rows.append({
            "时间": (r["ts"] or "")[:19],
            "类型": _TYPE_LABEL.get(r["decision_type"], r["decision_type"]),
            "标的": r["symbol"] or "—",
            "风险灯": f"{_RISK_LIGHT_EMOJI.get(light, '⚪')} {light or '—'}",
            "VIX": (f"{r['vix']:.1f}" if r.get("vix") else "—"),
            "组合规模": (f"${r['portfolio_value']:,.0f}"
                       if r.get("portfolio_value") else "—"),
            "持仓数": (str(int(r["num_positions"]))
                      if r.get("num_positions") is not None else "—"),
            "理由": (r["reason"] or "")[:60],
        })
    st.dataframe(pd.DataFrame(display_rows),
                 hide_index=True, use_container_width=True)

    # ── Selected-row payload expander ──────────────────────────────
    with st.expander("查看原始 payload (调试用)"):
        idx_options = [f"{r['ts'][:19]}  {_TYPE_LABEL.get(r['decision_type'], r['decision_type'])}  {r['symbol'] or '*'}"
                       for r in decisions]
        sel = st.selectbox("选一条", idx_options) if idx_options else None
        if sel is not None:
            pos = idx_options.index(sel)
            st.json(decisions[pos])
