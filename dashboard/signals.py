"""Market state display and today's signals."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from utils import get_logger
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from strategy import SIGNAL_LABEL, STRATEGY_MAP
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


# ---------------------------------------------------------------------------
# Phase 2.A — signal detail panel
# ---------------------------------------------------------------------------


# Per-direction copy used in the detail card. Keeps the rendering function
# small and lets non-developers tweak the messaging in one place.
_DIRECTION = {
    1: {"label": "买入", "color": "#28a745", "icon": "📈",
        "stop_factor": -2.0, "target_factor": 3.0,
        "stop_word": "止损", "target_word": "止盈"},
    -1: {"label": "卖出", "color": "#dc3545", "icon": "📉",
         "stop_factor": 2.0, "target_factor": -3.0,
         "stop_word": "止损(空)", "target_word": "止盈(空)"},
}


def render_signal_detail(config, target_date, provider, cache):
    """Today's signal summary + per-symbol expandable detail cards.

    Combines the previous compact list (render_todays_signals) and the
    detail panel into one section to avoid duplicate displays. Top row is
    a metric-style summary; below it, each symbol with a non-zero signal
    gets an expander showing trigger indicators, suggested stop / target,
    and other strategies' signals on the same symbol.
    """
    from daily import scan_day

    st.subheader(f"今日信号 ({target_date})")

    results = scan_day(config, target_date=target_date.isoformat(),
                       provider=provider, cache=cache)

    # Summary metrics — always show, even with no signals
    active = [r for r in results if r["signal"] != 0]
    n_buys = sum(1 for r in active if r["signal"] == 1)
    n_sells = sum(1 for r in active if r["signal"] == -1)
    n_symbols = len({r["symbol"] for r in active})
    n_watchlist = len(config.get("watchlist", []))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📈 买入信号", n_buys)
    m2.metric("📉 卖出信号", n_sells)
    m3.metric("有信号标的", f"{n_symbols} / {n_watchlist}")
    m4.metric("总策略×标的", len(results))

    if not active:
        st.info("今日无买入/卖出信号 — 等待")
        st.divider()
        return

    # Group active signals by symbol; collect all-strategy snapshot for context
    grouped: dict[str, list] = defaultdict(list)
    for r in active:
        grouped[r["symbol"]].append(r)
    all_by_symbol: dict[str, list] = defaultdict(list)
    for r in results:
        all_by_symbol[r["symbol"]].append(r)

    # Active strategy lookup (so we can mark which signal is the "primary" one)
    active_map = {
        item["symbol"]: item.get("active", "")
        for item in config.get("watchlist", [])
    }

    for sym, sigs in sorted(grouped.items()):
        # The "primary" signal for display is the active strategy's signal
        # if present, otherwise the first signal from the monitors.
        primary = next(
            (s for s in sigs if s["strategy"] == active_map.get(sym, "")),
            sigs[0],
        )
        direction = primary["signal"]
        cfg = _DIRECTION.get(direction)
        if cfg is None:
            continue
        price = primary["price"]
        atr = primary["atr"]
        stop_price = price + cfg["stop_factor"] * atr if atr > 0 else 0
        target_price = price + cfg["target_factor"] * atr if atr > 0 else 0
        stop_pct = (stop_price - price) / price * 100 if price > 0 and atr > 0 else 0
        target_pct = (target_price - price) / price * 100 if price > 0 and atr > 0 else 0

        title = (
            f"{cfg['icon']} {sym}  {cfg['label']}  @ ${price:.2f}  "
            f"({primary['strategy']})"
        )
        with st.expander(title, expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("触发价", f"${price:.2f}")
            c2.metric("ATR(14)", f"${atr:.2f}" if atr > 0 else "—")
            if atr > 0:
                c3.metric(
                    f"建议 {cfg['stop_word']}",
                    f"${stop_price:.2f}",
                    f"{stop_pct:+.2f}%",
                )
                c4.metric(
                    f"建议 {cfg['target_word']}",
                    f"${target_price:.2f}",
                    f"{target_pct:+.2f}%",
                )
            else:
                c3.metric(f"建议 {cfg['stop_word']}", "—")
                c4.metric(f"建议 {cfg['target_word']}", "—")

            # Indicator snapshot — the strategy's own context, filtered to
            # human-readable values only. Strategy authors put whatever
            # technicals matter in the indicators dict, so this is the
            # truest representation of "why did the signal fire".
            ind = primary.get("indicators") or {}
            if ind:
                ind_items = [
                    (k, v) for k, v in ind.items()
                    if isinstance(v, (int, float)) and not str(k).startswith("_")
                ]
                # Show top 6 indicators (most strategies use 3-6 indicators)
                if ind_items:
                    st.markdown("**指标快照**")
                    rows = []
                    for k, v in ind_items[:8]:
                        rows.append({"指标": k, "数值": f"{v:.4g}"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

            # Other strategies monitoring this symbol — agreement check
            others = [r for r in all_by_symbol[sym] if r["strategy"] != primary["strategy"]]
            if others:
                agree_lines = []
                for r in others:
                    sig_label = SIGNAL_LABEL.get(r["signal"], "—")
                    tag = "(active)" if r["strategy"] == active_map.get(sym, "") else "(monitor)"
                    agree_lines.append(f"{r['strategy']} {tag}: {sig_label}")
                st.markdown("**其它策略观察**: " + "  ·  ".join(agree_lines))

    st.divider()


# ---------------------------------------------------------------------------
# Phase 2.B — simulated position watch (trailing-stop proximity warning)
# ---------------------------------------------------------------------------


def _find_open_simulated_trade(df_sig: pd.DataFrame) -> int | None:
    """Locate the most recent unclosed buy signal in ``df_sig``.

    Scans the Signal column from the end backward. A buy (1) with no
    sell (-1) in between is treated as an open simulated position.
    Returns the bar index of the entry, or None if no open trade.
    """
    if "Signal" not in df_sig.columns:
        return None
    signals = df_sig["Signal"].values
    for i in range(len(signals) - 1, -1, -1):
        if signals[i] == -1:
            return None
        if signals[i] == 1:
            return i
    return None


def render_position_watch(config, target_date, provider):
    """Display 'simulated positions' from each symbol's active strategy and
    flag any within 5% of their trailing stop.

    Uses a Chandelier-style trailing stop estimate
    (``highest_high_since_entry - trail_atr_mult × ATR``) as a conservative
    proxy. Symbols whose latest active-strategy signal is a closed trade
    (i.e. saw a sell after the most recent buy) are excluded.
    """
    st.subheader("假设持仓监控")
    st.caption(
        "按 active 策略最近一次未平 buy 信号假设入场, "
        "用 Chandelier 移动止损估算距离. 仅用于减仓预警, 不替代实盘持仓数据."
    )

    lookback_years = config.get("scanner", {}).get("lookback_years", 3)
    start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()

    rows: list[dict] = []
    for item in config.get("watchlist", []):
        symbol = item["symbol"]
        strat_name = item.get("active", "")
        if not isinstance(strat_name, str) or strat_name not in STRATEGY_MAP:
            continue  # skip ensemble (list) and unknowns

        try:
            df = provider.get_daily(symbol, start=start, end=end)
        except Exception as exc:
            logger.debug("position watch fetch failed for %s: %s", symbol, exc)
            continue
        if df is None or df.empty:
            continue

        params = config.get("strategy", {}).get(strat_name, {})
        try:
            strategy = STRATEGY_MAP[strat_name](**params)
            df_sig = strategy.calculate_indicators(df)
        except Exception as exc:
            logger.debug("position watch indicators failed for %s/%s: %s",
                         symbol, strat_name, exc)
            continue

        idx_entry = _find_open_simulated_trade(df_sig)
        if idx_entry is None:
            continue

        try:
            entry_date = df_sig.index[idx_entry]
            entry_price = float(df_sig["Close"].iloc[idx_entry])
            current_price = float(df_sig["Close"].iloc[-1])
            atr = float(df_sig["ATR"].iloc[-1]) if "ATR" in df_sig.columns else 0.0
        except (KeyError, IndexError):
            continue
        if atr <= 0 or entry_price <= 0:
            continue

        # Chandelier stop — uses High when available, else Close
        track_col = "High" if "High" in df_sig.columns else "Close"
        try:
            highest = float(df_sig[track_col].iloc[idx_entry:].max())
        except (KeyError, ValueError):
            continue

        trail_mult = float(getattr(strategy.params, "trail_atr_mult", 2.5))
        stop_price = highest - trail_mult * atr

        pnl_pct = (current_price / entry_price - 1) * 100
        dist_pct = (current_price - stop_price) / current_price * 100 if current_price > 0 else 0
        days_held = (df_sig.index[-1] - entry_date).days

        rows.append({
            "symbol": symbol,
            "strategy": strat_name,
            "entry_date": entry_date.date().isoformat(),
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "stop_price": stop_price,
            "distance_pct": dist_pct,
            "days_held": days_held,
        })

    if not rows:
        st.info("当前无假设持仓 — active 策略最近的信号均已平仓 (或从未触发买入)")
        st.divider()
        return

    # Sort: closest to stop first (most urgent)
    rows.sort(key=lambda r: r["distance_pct"])

    df_display = pd.DataFrame([{
        "标的": r["symbol"],
        "策略": r["strategy"],
        "入场日": r["entry_date"],
        "入场价": f"${r['entry_price']:.2f}",
        "当前价": f"${r['current_price']:.2f}",
        "浮盈%": f"{r['pnl_pct']:+.2f}%",
        "移动止损": f"${r['stop_price']:.2f}",
        "距止损%": f"{r['distance_pct']:.2f}%",
        "持仓天": r["days_held"],
    } for r in rows])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    danger = [r for r in rows if r["distance_pct"] < 5]
    if danger:
        for r in danger:
            st.warning(
                f"⚠ {r['symbol']} ({r['strategy']}) 距移动止损仅 "
                f"{r['distance_pct']:.2f}% (止损位 ${r['stop_price']:.2f})"
            )
    else:
        st.success(f"全部 {len(rows)} 个假设持仓距移动止损 ≥ 5%, 暂无减仓预警")

    st.divider()
