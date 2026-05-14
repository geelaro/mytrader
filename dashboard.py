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
from trader import BacktestEngine, plot_result, print_result
from utils import load_toml, get_logger

logger = get_logger("dashboard")

st.set_page_config(page_title="mytrader", layout="wide")
st.title("mytrader Dashboard")

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
# Row 1 — Today's signals + Account overview
# ---------------------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"今日信号 ({target_date})")

    # Run scan
    from daily import scan_day

    results = scan_day(config, target_date=target_date.isoformat(),
                       provider=provider, cache=cache)

    buys = [r for r in results if r["signal"] == 1]
    sells = [r for r in results if r["signal"] == -1]

    if buys or sells:
        for r in buys:
            active_tag = "★" if r["strategy"] == config["watchlist"][0].get("active", "") else ""
            st.success(f"{r['symbol']} — {r['strategy']} {SIGNAL_LABEL[r['signal']]} @ ${r['price']:.2f} {active_tag}")
        for r in sells:
            st.error(f"{r['symbol']} — {r['strategy']} {SIGNAL_LABEL[r['signal']]} @ ${r['price']:.2f}")
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
# Row 2 — Backtest chart
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Row 3 — Signal history + Strategy compare
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Row 4 — Portfolio backtest
# ---------------------------------------------------------------------------

st.divider()
st.header("组合回测")

from portfolio import PortfolioBacktest, DEFAULT_PORTFOLIO


@st.cache_data(ttl=3600, show_spinner="运行组合回测...")
def _cached_portfolio_bt(start, end):
    bt = PortfolioBacktest(
        legs=DEFAULT_PORTFOLIO,
        initial_capital=100000,
        allocation="equal",
    )
    return bt.run(start=start, end=end)


pf_result = _cached_portfolio_bt(start, end)

# Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("总收益", f"{pf_result.total_return_pct:+.1f}%")
m2.metric("夏普", f"{pf_result.sharpe_ratio:.2f}")
m3.metric("最大回撤", f"{pf_result.max_drawdown_pct:.1f}%")
m4.metric("交易笔数", pf_result.total_trades)
m5.metric("胜率", f"{pf_result.win_rate_pct:.1f}%")
m6.metric("盈亏比", f"{pf_result.profit_factor:.2f}")

# Equity curve + Drawdown
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

# Per-symbol breakdown + Trade details
col1, col2 = st.columns(2)

with col1:
    st.subheader("按标的统计")
    if pf_result.closed_trades:
        by_symbol: dict = {}
        for t in pf_result.closed_trades:
            by_symbol.setdefault(t.symbol, []).append(t)
        sym_rows = []
        for sym, sym_trades in sorted(by_symbol.items()):
            n = len(sym_trades)
            wr = sum(1 for t in sym_trades if t.pnl is not None and t.pnl > 0) / n * 100
            total_pnl = sum(t.pnl or 0 for t in sym_trades)
            sym_rows.append({
                "标的": sym, "笔数": n, "胜率%": round(wr, 1),
                "总PnL": round(total_pnl, 0), "平均PnL": round(total_pnl / n, 0),
            })
        st.dataframe(pd.DataFrame(sym_rows), use_container_width=True, hide_index=True)

with col2:
    st.subheader("交易统计")
    st.metric("平均盈利", f"${pf_result.avg_win:,.0f}")
    st.metric("平均亏损", f"${pf_result.avg_loss:,.0f}")
    st.metric("平均持仓天数", f"{pf_result.avg_hold_days:.1f}")

with st.expander(f"交易明细 ({pf_result.total_trades} 笔)"):
    if pf_result.closed_trades:
        trade_rows = []
        for t in pf_result.closed_trades:
            trade_rows.append({
                "标的": t.symbol,
                "入场日": t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time, "strftime") else str(t.entry_time)[:10],
                "出场日": t.exit_time.strftime("%Y-%m-%d") if t.exit_time and hasattr(t.exit_time, "strftime") else (str(t.exit_time)[:10] if t.exit_time else ""),
                "数量": t.qty,
                "入场价": round(t.entry_price, 2),
                "出场价": round(t.exit_price, 2) if t.exit_price else None,
                "PnL": round(t.pnl, 0) if t.pnl else None,
                "PnL%": round(t.pnl_pct, 2) if t.pnl_pct else None,
                "原因": t.reason,
                "持仓天": t.hold_days,
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
