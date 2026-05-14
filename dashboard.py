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

start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
end = target_date.isoformat()

strategy_cls = STRATEGY_MAP.get(selected_strategy)
if strategy_cls:
    try:
        df = provider.get_daily(selected_symbol, start=start, end=end)
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

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(result.equity_curve.index, result.equity_curve.values,
                    color="#2ca02c", linewidth=1.2, label="策略权益")

            # Benchmark
            if result.buy_hold_return_pct != 0:
                bh_start = result.initial_capital
                price_data = df["Close"]
                price_data = price_data[price_data.index >= result.equity_curve.index[0]]
                bh_curve = bh_start * (price_data / price_data.iloc[0])
                ax.plot(bh_curve.index, bh_curve.values,
                        color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7, label="买入持有")

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
            df_cmp = provider.get_daily(selected_symbol, start=start, end=end)
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
