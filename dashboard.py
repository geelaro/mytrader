"""mytrader Dashboard — Streamlit web UI.

Usage:
    pipenv run streamlit run dashboard.py
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP, SIGNAL_LABEL
from engine.trader import BacktestEngine, plot_result, print_result
from utils import load_toml, get_logger
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility

logger = get_logger("dashboard")

st.set_page_config(page_title="Mytrader", layout="wide")
st.title("Mytrader Dashboard")

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

st.sidebar.header("控制面板")

target_date = st.sidebar.date_input("日期", date.today())

# Load config
config = load_toml("watchlist.toml")
symbols = [item["symbol"] for item in config.get("watchlist", [])]

selected_symbol = st.sidebar.selectbox("标的", symbols)

strategy_options = list(STRATEGY_MAP.keys())
selected_strategy = st.sidebar.selectbox("策略", strategy_options, index=3)

backtest_years = st.sidebar.slider("回测年数", 1, 10, 4)
allocation_mode = st.sidebar.selectbox("组合分配模式", ["equal", "dynamic_equal"], index=0)
pf_strategy = st.sidebar.selectbox("组合策略", strategy_options, index=3,
                                   help="所有标的使用统一策略")

# Initialize data provider (cached)
@st.cache_resource
def get_provider():
    return DataProvider()

@st.cache_resource
def get_cache():
    return CacheManager()

provider = get_provider()
cache = get_cache()

# ---------------------------------------------------------------------------
# Market state
# ---------------------------------------------------------------------------

ms_cfg = config.get("market_state", {})
if ms_cfg.get("enabled", False):
    st.subheader("市场状态")

    proxy_sym = ms_cfg.get("proxy_symbol", "SPY")
    lookback = config.get("default", {}).get("lookback_years", 3)
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

# ---------------------------------------------------------------------------
# Row 1 — Today's signals + Account overview
# ---------------------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"今日信号 ({target_date})")

    from daily import scan_day

    results = scan_day(config, target_date=target_date.isoformat(),
                       provider=provider, cache=cache)

    active_signals = [r for r in results if r["signal"] != 0]
    if active_signals:
        # Group by symbol
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in active_signals:
            grouped[r["symbol"]].append(r)

        for sym, sigs in sorted(grouped.items()):
            parts = []
            has_buy = any(s["signal"] == 1 for s in sigs)
            has_sell = any(s["signal"] == -1 for s in sigs)
            price = sigs[0]["price"]
            for s in sigs:
                tag = "★" if s["strategy"] == next((item.get("active", "") for item in config.get("watchlist", []) if item["symbol"] == sym), "") else ""
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
    # Active strategy count
    active_counts = {}
    for item in config.get("watchlist", []):
        a = item.get("active", "-")
        active_counts[a] = active_counts.get(a, 0) + 1

    for strat, count in sorted(active_counts.items(), key=lambda x: -x[1]):
        st.metric(label=strat, value=f"{count} 个标的")

    st.metric(label="总组合数", value=f"{len(results)} 策略×标的")

# ---------------------------------------------------------------------------
# Tabs: 单标的回测 | 组合回测
# ---------------------------------------------------------------------------

tab_single, tab_portfolio = st.tabs(["单标的回测", "组合回测"])

# ========================
# Tab 1 — 单标的回测
# ========================

with tab_single:

    st.subheader(f"策略回测: {selected_symbol} + {selected_strategy}")

    @st.cache_data(ttl=3600, show_spinner="获取数据中...")
    def _cached_get_daily(symbol, start, end):
        return provider.get_daily(symbol, start=start, end=end)


    start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()

    strategy_cls = STRATEGY_MAP.get(selected_strategy)
    if strategy_cls:
        try:
            df = _cached_get_daily(selected_symbol, start, end)
            if df is not None and not df.empty:
                strategy = strategy_cls()
                df_sig = strategy.calculate_indicators(df)
                engine = BacktestEngine(initial_capital=10000)
                bench = engine.run(strategy, df_sig)
                result = engine.get_result(bench)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("总收益", f"{result.total_return_pct:+.1f}%")
                col2.metric("夏普", f"{result.sharpe_ratio:.2f}")
                col3.metric("最大回撤", f"{result.max_drawdown_pct:.1f}%")
                col4.metric("胜率", f"{result.win_rate_pct:.1f}%")
                col5.metric("交易", result.total_trades)

                # Plot equity curve
                import matplotlib.pyplot as plt

                # Configure Chinese font rendering
                for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
                    try:
                        plt.rcParams["font.sans-serif"] = [font]
                        break
                    except Exception:
                        continue
                plt.rcParams["axes.unicode_minus"] = False

                fig, ax = plt.subplots(figsize=(10, 4))
                eq = result.equity_curve
                ax.plot(eq.index, eq.values,
                        color="#2ca02c", linewidth=1.2, label="策略权益")

                # Benchmark
                if result.buy_hold_return_pct != 0:
                    bh_start = result.initial_capital
                    price_data = df["Close"]
                    price_data = price_data[price_data.index >= eq.index[0]]
                    bh_curve = bh_start * (price_data / price_data.iloc[0])
                    ax.plot(bh_curve.index, bh_curve.values,
                            color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7, label="买入持有")

                # Buy / Sell markers on equity curve
                if result.trades:
                    for t in result.trades:
                        try:
                            pos = eq.index.get_loc(t.entry_date)
                            idx = pos if isinstance(pos, int) else pos.start
                            ax.scatter(eq.index[idx], float(eq.iloc[idx]),
                                       color="limegreen", marker="^", s=50, zorder=5)
                        except (KeyError, IndexError):
                            pass
                        try:
                            pos = eq.index.get_loc(t.exit_date)
                            idx = pos if isinstance(pos, int) else pos.start
                            ax.scatter(eq.index[idx], float(eq.iloc[idx]),
                                       color="red", marker="v", s=50, zorder=5)
                        except (KeyError, IndexError):
                            pass

                ax.axhline(y=10000, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
                ax.set_title(f"{selected_symbol} — {selected_strategy}")
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                st.pyplot(fig)
                plt.close(fig)

                # Indicator preview
                with st.expander("最新指标"):
                    last_row = df_sig.iloc[-1]
                    indicator_cols = [c for c in df_sig.columns
                                      if c not in ("Open", "High", "Low", "Close", "Volume", "Signal")]
                    cols = st.columns(min(len(indicator_cols), 6))
                    for i, col_name in enumerate(indicator_cols[:18]):
                        val = last_row[col_name]
                        if isinstance(val, (float, int)) and not pd.isna(val):
                            cols[i % 6].metric(col_name, f"{float(val):.4f}")

                # Trade details (single-symbol)
                with st.expander(f"交易明细 ({result.total_trades} 笔)"):
                    if result.trades:
                        pf_str = "∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
                        st.caption(
                            f"胜率 {result.win_rate_pct:.1f}%  |  "
                            f"盈亏比 {pf_str}  |  "
                            f"均盈 {result.avg_win_pct:+.2f}%  |  "
                            f"均亏 {result.avg_loss_pct:+.2f}%"
                        )
                        trade_rows = []
                        for t in result.trades:
                            trade_rows.append({
                                "入场日": t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date)[:10],
                                "出场日": t.exit_date.strftime("%Y-%m-%d") if hasattr(t.exit_date, "strftime") else str(t.exit_date)[:10],
                                "数量": t.quantity,
                                "入场价": round(t.entry_price, 2),
                                "出场价": round(t.exit_price, 2),
                                "PnL": round(t.pnl, 0),
                                "PnL%": f"{t.pnl_pct:+.2f}%",
                                "原因": t.exit_reason,
                                "持仓天": t.holding_days,
                            })
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("无交易记录")
            else:
                st.warning(f"无 {selected_symbol} 数据")
        except Exception as e:
            st.error(f"回测失败: {e}")

        # -----------------------------------------------------------------------
        # Row 3 — Signal history + Strategy compare
        # -----------------------------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("近期信号历史")
            since = (date.today() - timedelta(days=7)).isoformat()
            rows = cache.query_signals(scan_date=since)
            if rows:
                df_hist = pd.DataFrame(rows)
                df_hist["signal_label"] = df_hist["signal"].map({1: "买入", -1: "卖出", 0: "—"})
                st.dataframe(
                    df_hist[["scan_date", "symbol", "strategy", "signal_label", "price"]].tail(20),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("近 7 天无信号记录")

        with col2:
            st.subheader("策略 Sharpe 对比")
            strategies_to_compare = ["weekly_macd_kdj", "turtle_trading", "enhanced_macd",
                                     "donchian_breakout", "bollinger_mean_reversion"]
            compare_data = []
            for sn in strategies_to_compare:
                cls = STRATEGY_MAP.get(sn)
                if cls is None:
                    continue
                try:
                    df_cmp = _cached_get_daily(selected_symbol, start, end)
                    if df_cmp is None or df_cmp.empty:
                        continue
                    s = cls()
                    df_s = s.calculate_indicators(df_cmp)
                    eng = BacktestEngine(initial_capital=10000)
                    ben = eng.run(s, df_s)
                    res = eng.get_result(ben)
                    compare_data.append({
                        "策略": sn,
                        "Sharpe": round(res.sharpe_ratio, 2),
                        "收益%": round(res.total_return_pct, 1),
                        "回撤%": round(res.max_drawdown_pct, 1),
                        "胜率%": round(res.win_rate_pct, 1),
                        "交易": res.total_trades,
                    })
                except Exception:
                    pass

            if compare_data:
                df_comp = pd.DataFrame(compare_data).sort_values("Sharpe", ascending=False)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ========================
# Tab 2 — 组合回测
# ========================

with tab_portfolio:

    st.header("组合回测")

    from engine.portfolio import PortfolioBacktest, Leg, PortfolioResult

    # Risk helpers
    def _drawdown_stats(curve):
        rolling_max = curve.expanding().max()
        dd = (curve - rolling_max) / rolling_max * 100
        current_dd = float(dd.iloc[-1])
        is_under = dd < 0
        longest = 0
        streak = 0
        for flag in is_under:
            if flag:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        return current_dd, float(dd.min()), longest

    def _exposure_from_trades(result, curve):
        """Reconstruct daily net exposure via mark-to-market (linear interpolation).

        Each position's value is interpolated from entry_cost to exit_value
        across its holding period, then summed per day.
        """
        if not result.closed_trades or len(curve) < 2:
            return pd.Series(dtype=float), 0.0, {}

        exposure = pd.Series(0.0, index=curve.index)
        symbols = list({t.symbol for t in result.closed_trades})
        by_symbol = {sym: pd.Series(0.0, index=curve.index) for sym in symbols}

        for t in result.closed_trades:
            entry_cost = t.entry_price * t.qty * 1.0004
            exit_value = entry_cost + (t.pnl or 0)

            # Find dates within holding period
            mask = (exposure.index >= t.entry_time) & (exposure.index <= t.exit_time)
            dates_in = exposure.index[mask]

            if len(dates_in) < 2:
                # Single-day trade — use entry_cost
                exposure.loc[mask] += entry_cost
                if t.symbol in by_symbol:
                    by_symbol[t.symbol].loc[mask] += entry_cost
                continue

            # Linear interpolation: entry_cost → exit_value
            total_days = (dates_in[-1] - dates_in[0]).days or 1
            for d in dates_in:
                frac = (d - dates_in[0]).days / total_days
                mtm_value = entry_cost + (exit_value - entry_cost) * frac
                exposure.loc[d] += mtm_value
                if t.symbol in by_symbol:
                    by_symbol[t.symbol].loc[d] += mtm_value

        net_pct = (exposure / curve) * 100
        last_exp = float(net_pct.iloc[-1]) if len(net_pct) > 0 else 0.0
        top = {}
        for sym, ser in sorted(by_symbol.items(), key=lambda x: -x[1].iloc[-1])[:3]:
            top[sym] = round(float(ser.iloc[-1] / curve.iloc[-1] * 100), 1)
        return net_pct, last_exp, top

    @st.cache_data(ttl=3600, show_spinner="运行组合回测...")
    def _cached_portfolio_bt(start, end, alloc, strategy):
        # Build legs from watchlist symbols with the selected strategy
        legs = [Leg(item["symbol"], strategy) for item in config.get("watchlist", [])]
        bt = PortfolioBacktest(
            legs=legs,
            initial_capital=100000,
            allocation=alloc,
        )
        return bt.run(start=start, end=end)

    pf_result = _cached_portfolio_bt(start, end, allocation_mode, pf_strategy)

    # --- Metrics row ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("总收益", f"{pf_result.total_return_pct:+.1f}%")
    m2.metric("夏普", f"{pf_result.sharpe_ratio:.2f}")
    m3.metric("最大回撤", f"{pf_result.max_drawdown_pct:.1f}%")
    m4.metric("交易笔数", pf_result.total_trades)
    m5.metric("胜率", f"{pf_result.win_rate_pct:.1f}%")
    m6.metric("盈亏比", f"{pf_result.profit_factor:.2f}")

    # --- Equity curve + Drawdown ---
    import matplotlib.pyplot as plt

    for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    curve = pf_result.equity_curve
    ax1 = axes[0]
    ax1.plot(curve.index, curve, color="#2ca02c", linewidth=1.2, label="组合权益")
    ax1.axhline(y=pf_result.initial_capital, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("组合权益曲线", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    rolling_max = curve.expanding().max()
    drawdown = (curve - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.4)
    ax2.plot(drawdown.index, drawdown, color="#d62728", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("组合回撤", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Risk dashboard ---
    st.subheader("风险看板")

    curve = pf_result.equity_curve
    current_dd, max_dd_pct, longest_dd_days = _drawdown_stats(curve)
    exposure_series, last_exposure, top_weights = _exposure_from_trades(pf_result, curve)

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("当前回撤", f"{current_dd:.1f}%",
                  delta=f"峰值 {max_dd_pct:.1f}%" if current_dd < -0.1 else None)
    with r2:
        st.metric("最长回撤天数", f"{longest_dd_days} 天")
    with r3:
        st.metric("期末净敞口", f"{last_exposure:.1f}%",
                  delta=f"峰值 {exposure_series.max():.0f}%" if len(exposure_series) > 0 else None)
    with r4:
        avg_exposure = exposure_series.mean() if len(exposure_series) > 0 else 0
        st.metric("平均敞口", f"{avg_exposure:.1f}%")
    with r5:
        active_pct = (exposure_series > 0.1).mean() * 100 if len(exposure_series) > 0 else 0
        st.metric("持仓时间占比", f"{active_pct:.0f}%")

    # Top weights row
    if top_weights:
        st.caption(f"期末 Top 3 标的全重: " +
                   "  |  ".join(f"**{sym}** {wgt:.1f}%" for sym, wgt in top_weights.items()))

    # --- Trade statistics cards ---
    st.subheader("交易统计")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("总笔数", pf_result.total_trades)
    s2.metric("胜率", f"{pf_result.win_rate_pct:.1f}%")
    s3.metric("盈亏比", f"{pf_result.profit_factor:.2f}")
    s4.metric("平均盈利", f"${pf_result.avg_win:,.0f}")
    s5.metric("平均亏损", f"${pf_result.avg_loss:,.0f}")
    s6.metric("平均持仓天", f"{pf_result.avg_hold_days:.1f}")

    # --- PnL Attribution ---
    st.subheader("收益归因")

    if pf_result.closed_trades:
        # Build attribution dataframe
        attr_rows = []
        for t in pf_result.closed_trades:
            pnl = t.pnl or 0
            attr_rows.append({
                "symbol": t.symbol,
                "month": pd.Timestamp(t.entry_time).strftime("%Y-%m"),
                "pnl": pnl,
                "pnl_pct": t.pnl_pct or 0,
                "entry": pd.Timestamp(t.entry_time),
            })
        df_attr = pd.DataFrame(attr_rows)
        total_pnl = df_attr["pnl"].sum()

        # Date filter
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            min_date = df_attr["entry"].min().date()
            max_date = df_attr["entry"].max().date()
            attr_start = st.date_input("筛选起始", min_date, key="attr_start")
        with fc2:
            attr_end = st.date_input("筛选结束", max_date, key="attr_end")

        mask = (df_attr["entry"] >= pd.Timestamp(attr_start)) & (df_attr["entry"] <= pd.Timestamp(attr_end))
        df_filt = df_attr[mask]
        filt_pnl = df_filt["pnl"].sum()

        with fc3:
            st.metric("区间总PnL", f"${filt_pnl:+,.0f}",
                      delta=f"全期 ${total_pnl:+,.0f}" if abs(filt_pnl - total_pnl) > 1 else None)

        # By Symbol + By Month (side by side)
        col_left, col_right = st.columns(2)

        with col_left:
            st.caption("按标的")

            # PnL by symbol bar chart
            by_sym = df_filt.groupby("symbol")["pnl"].sum().sort_values()
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in by_sym]
            fig_sym, ax_sym = plt.subplots(figsize=(6, 2 + len(by_sym) * 0.35))
            ax_sym.barh(by_sym.index, by_sym.values, color=colors, height=0.6)
            ax_sym.axvline(x=0, color="gray", linewidth=0.5)
            ax_sym.set_xlabel("PnL ($)")
            for i, v in enumerate(by_sym.values):
                ax_sym.text(v + (50 if v >= 0 else -50), i, f"${v:+,.0f}",
                            va="center", fontsize=9, ha="left" if v >= 0 else "right")
            ax_sym.grid(axis="x", alpha=0.3)
            st.pyplot(fig_sym)
            plt.close(fig_sym)

            # Contribution table
            by_sym_detail = df_filt.groupby("symbol").agg(
                总PnL=("pnl", "sum"), 笔数=("pnl", "count"),
                胜率=("pnl", lambda x: (x > 0).mean() * 100),
                平均PnL=("pnl", "mean"), 最大=("pnl", "max"), 最小=("pnl", "min"),
            ).sort_values("总PnL", ascending=False)
            by_sym_detail["贡献%"] = (by_sym_detail["总PnL"] / filt_pnl * 100).round(1) if filt_pnl != 0 else 0
            by_sym_detail["总PnL"] = by_sym_detail["总PnL"].round(0).astype(int)
            by_sym_detail["平均PnL"] = by_sym_detail["平均PnL"].round(0).astype(int)
            by_sym_detail["胜率"] = by_sym_detail["胜率"].round(1)
            st.dataframe(by_sym_detail, use_container_width=True)

        with col_right:
            st.caption("按月份")

            # PnL by month bar chart
            by_month = df_filt.groupby("month")["pnl"].sum()
            if len(by_month) > 0:
                colors_m = ["#d62728" if v < 0 else "#2ca02c" for v in by_month]
                fig_mo, ax_mo = plt.subplots(figsize=(6, 2 + len(by_month) * 0.35))
                ax_mo.barh(by_month.index, by_month.values, color=colors_m, height=0.6)
                ax_mo.axvline(x=0, color="gray", linewidth=0.5)
                ax_mo.set_xlabel("PnL ($)")
                for i, v in enumerate(by_month.values):
                    ax_mo.text(v + (50 if v >= 0 else -50), i, f"${v:+,.0f}",
                               va="center", fontsize=9, ha="left" if v >= 0 else "right")
                ax_mo.grid(axis="x", alpha=0.3)
                ax_mo.invert_yaxis()
                st.pyplot(fig_mo)
                plt.close(fig_mo)

                # Monthly detail table
                by_mo_detail = df_filt.groupby("month").agg(
                    总PnL=("pnl", "sum"), 笔数=("pnl", "count"),
                    胜率=("pnl", lambda x: (x > 0).mean() * 100),
                    月均PnL=("pnl_pct", "mean"),
                ).sort_index(ascending=False)
                by_mo_detail["累计PnL"] = by_mo_detail["总PnL"].cumsum()
                by_mo_detail["总PnL"] = by_mo_detail["总PnL"].round(0).astype(int)
                by_mo_detail["胜率"] = by_mo_detail["胜率"].round(1)
                by_mo_detail["月均PnL"] = by_mo_detail["月均PnL"].round(2)
                by_mo_detail["累计PnL"] = by_mo_detail["累计PnL"].round(0).astype(int)
                st.dataframe(by_mo_detail, use_container_width=True)

        # Top contributors / detractors summary
        st.caption("贡献源 & 拖累源")
        c1, c2 = st.columns(2)
        with c1:
            top3 = df_filt.groupby("symbol")["pnl"].sum().nlargest(3)
            for sym, p in top3.items():
                pct = f"{p/filt_pnl*100:.0f}%" if filt_pnl != 0 else "—"
                st.metric(f"↑ {sym}", f"${p:+,.0f}", delta=pct)
        with c2:
            bot3 = df_filt.groupby("symbol")["pnl"].sum().nsmallest(3)
            for sym, p in bot3.items():
                pct = f"{p/filt_pnl*100:.0f}%" if filt_pnl != 0 else "—"
                st.metric(f"↓ {sym}", f"${p:+,.0f}", delta=pct)
    else:
        st.info("无交易记录")

    # --- Filtered trade details ---
    st.subheader("交易明细")

    if pf_result.closed_trades:
        import io
        import base64

        # Build base dataframe
        trade_rows = []
        for t in pf_result.closed_trades:
            entry_str = t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, "strftime") else str(t.entry_time)[:10]
            exit_str = t.exit_time.strftime("%Y-%m-%d") if t.exit_time and hasattr(t.exit_time, "strftime") else (str(t.exit_time)[:10] if t.exit_time else "")
            trade_rows.append({
                "标的": t.symbol,
                "入场日": entry_str,
                "出场日": exit_str,
                "数量": t.qty,
                "入场价": round(t.entry_price, 2),
                "出场价": round(t.exit_price, 2) if t.exit_price else None,
                "PnL": round(t.pnl, 0) if t.pnl else 0,
                "PnL%": round(t.pnl_pct, 2) if t.pnl_pct else 0,
                "原因": t.reason,
                "持仓天": t.hold_days,
                "入场": pd.Timestamp(entry_str),
                "出场": pd.Timestamp(exit_str) if exit_str else pd.NaT,
            })
        df_trades = pd.DataFrame(trade_rows)

        # --- Filters (2 rows) ---
        fr1, fr2, fr3, fr4 = st.columns(4)
        with fr1:
            symbols = sorted(df_trades["标的"].unique().tolist())
            filter_sym = st.multiselect("标的", symbols, default=symbols, key="pf_filter_sym")
        with fr2:
            reasons = sorted(df_trades["原因"].unique().tolist())
            filter_reason = st.multiselect("原因", reasons, default=reasons, key="pf_filter_reason")
        with fr3:
            min_pnl = int(df_trades["PnL"].min())
            max_pnl = int(df_trades["PnL"].max())
            filter_pnl_range = st.slider("PnL ($)", min_pnl, max_pnl,
                                         (min_pnl, max_pnl), step=100, key="pf_filter_pnl_range")
        with fr4:
            max_hold = int(df_trades["持仓天"].max())
            filter_hold = st.slider("持仓天数", 0, max(1, max_hold),
                                    (0, max_hold), step=1, key="pf_filter_hold")

        fr5, fr6, fr7, fr8 = st.columns(4)
        with fr5:
            if not df_trades["入场"].isna().all():
                min_date = df_trades["入场"].min().date()
                max_date = df_trades["出场"].max().date() if not df_trades["出场"].isna().all() else pd.Timestamp.today().date()
                filter_dates = st.date_input("日期区间", value=(min_date, max_date), key="pf_filter_date")

        # Apply filters
        df_filtered = df_trades[df_trades["标的"].isin(filter_sym)]
        df_filtered = df_filtered[df_filtered["原因"].isin(filter_reason)]
        df_filtered = df_filtered[(df_filtered["PnL"] >= filter_pnl_range[0]) &
                                   (df_filtered["PnL"] <= filter_pnl_range[1])]
        df_filtered = df_filtered[(df_filtered["持仓天"] >= filter_hold[0]) &
                                   (df_filtered["持仓天"] <= filter_hold[1])]
        if isinstance(filter_dates, tuple) and len(filter_dates) == 2:
            d1, d2 = pd.Timestamp(filter_dates[0]), pd.Timestamp(filter_dates[1])
            df_filtered = df_filtered[(df_filtered["入场"] >= d1) & (df_filtered["入场"] <= d2)]

        # --- Summary + Export row ---
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.caption(f"共 {len(df_filtered)} 笔（筛选自 {len(df_trades)} 笔）")
        with ec2:
            filtered_pnl = df_filtered["PnL"].sum()
            st.metric("筛选PnL合计", f"${filtered_pnl:+,.0f}")
        with ec3:
            csv_buffer = io.StringIO()
            display_cols = ["标的", "入场日", "出场日", "数量", "入场价", "出场价", "PnL", "PnL%", "原因", "持仓天"]
            df_filtered[display_cols].to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            st.download_button("⬇ CSV", csv_buffer.getvalue(),
                               f"trades_{pf_strategy}_{attr_start}_{attr_end}.csv",
                               "text/csv", key="dl_csv")
        with ec4:
            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                df_filtered[display_cols].to_excel(writer, index=False, sheet_name="交易明细")
            st.download_button("⬇ Excel", xlsx_buffer.getvalue(),
                               f"trades_{pf_strategy}_{attr_start}_{attr_end}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_xlsx")

        # Display table (outside columns for full width)
        st.dataframe(df_filtered[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("无交易记录")

# ---------------------------------------------------------------------------
# 组合相关性矩阵
# ---------------------------------------------------------------------------

with st.expander("持仓行业分布", expanded=False):
    from utils.sectors import get_sector
    from collections import Counter

    watchlist_syms = [item["symbol"] for item in config.get("watchlist", [])]
    sector_counts = Counter()
    for sym in watchlist_syms:
        sector_counts[get_sector(sym)] += 1

    if sector_counts:
        # Group symbols by sector
        sector_symbols = {}
        for sym in watchlist_syms:
            sec = get_sector(sym)
            sector_symbols.setdefault(sec, []).append(sym)

        import matplotlib.pyplot as _plt
        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())
        colors = _plt.cm.Set3(range(len(labels)))
        fig, ax = _plt.subplots(figsize=(4, 3))
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

        # Symbol list per sector
        for sec in sorted(sector_symbols.keys()):
            st.caption(f"{sec}: {', '.join(sector_symbols[sec])}")
    else:
        st.info("无数据")

# ---------------------------------------------------------------------------
# Monte Carlo 风控快照
# ---------------------------------------------------------------------------

with st.expander("Monte Carlo 风控快照", expanded=False):
    from engine.portfolio import PortfolioBacktest, Leg
    import numpy as np

    mc_sym = st.selectbox("标的", symbols, key="mc_sym")
    mc_strat = st.selectbox("策略", strategy_options, index=3, key="mc_strat")
    mc_years = st.slider("回测年数", 2, 8, 4, key="mc_years")

    if st.button("运行 Monte Carlo", key="mc_run"):
        mc_start = (pd.Timestamp(target_date) - pd.DateOffset(years=mc_years)).strftime("%Y-%m-%d")
        mc_end = target_date.isoformat()

        with st.spinner("Monte Carlo 运行中..."):
            legs = [Leg(mc_sym, mc_strat)]
            pf = PortfolioBacktest(legs, initial_capital=10000)
            result = pf.run(start=mc_start, end=mc_end)

            pnl_pcts = np.array([
                t.pnl / result.initial_capital * 100
                for t in result.trades if t.pnl is not None
            ])

            if len(pnl_pcts) >= 5:
                # Monte Carlo: 1000 shuffles
                n_sims = 1000
                max_dds = []
                rng = np.random.default_rng(42)
                for _ in range(n_sims):
                    shuffled = rng.permutation(pnl_pcts)
                    equity = 100.0; peak = 100.0; max_dd = 0.0
                    for pnl in shuffled:
                        equity *= (1 + pnl / 100)
                        if equity > peak: peak = equity
                        dd = (peak - equity) / peak * 100
                        if dd > max_dd: max_dd = dd
                    max_dds.append(max_dd)
                max_dds = np.array(max_dds)

                p95 = np.percentile(max_dds, 95)
                p50 = np.percentile(max_dds, 50)
                win_rate = (pnl_pcts > 0).sum() / len(pnl_pcts) * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("P95 最坏回撤", f"{p95:.1f}%")
                c2.metric("中位回撤", f"{p50:.1f}%")
                c3.metric("交易数", len(pnl_pcts))
                c4.metric("胜率", f"{win_rate:.0f}%")

                if p95 <= 15:
                    st.success("回撤风险: 低 — 策略质量正常")
                elif p95 <= 25:
                    st.warning("回撤风险: 中 — 关注策略表现")
                else:
                    st.error("回撤风险: 高 — 建议检查策略参数或市场状态")
            else:
                st.info(f"交易数量不足 ({len(pnl_pcts)}笔)，至少需要 5 笔")
