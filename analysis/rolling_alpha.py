"""Rolling-window alpha decay — does strategy performance degrade over time?

Usage:
    pipenv run python analysis/rolling_alpha.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils  # noqa: F401 - triggers env setup before matplotlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import DataProvider
from strategy import STRATEGY_MAP
from engine.trader import BacktestEngine


def run(
    strategy: str = "weekly_macd_kdj",
    symbols: list = None,
    start_range: str = "2018-01-01",
    end_range: str = "2024-06-01",
    window_years: int = 2,
    step_months: int = 3,
    initial_capital: float = 10000,
) -> pd.DataFrame:
    """Run rolling-window alpha decay analysis.

    Parameters
    ----------
    strategy : str
        Strategy name (key of STRATEGY_MAP).
    symbols : list[str] | None
        List of ticker symbols (default: US tech watchlist).
    start_range : str
        Earliest roll start date (YYYY-MM-DD).
    end_range : str
        Latest roll start date (YYYY-MM-DD).
    window_years : int
        Rolling window size in years.
    step_months : int
        Step size between windows in months.
    initial_capital : float
        Capital per symbol backtest.

    Returns
    -------
    pd.DataFrame
        Columns: start, end, strategy_return, benchmark_return, alpha, symbols
    """
    provider = DataProvider()
    strategy_cls = STRATEGY_MAP[strategy]
    strat = strategy_cls()

    if symbols is None:
        symbols = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY"]

    start_dates = pd.bdate_range(start_range, end_range, freq=f"{step_months}MS")

    results = []
    for start_dt in start_dates:
        start = start_dt.strftime("%Y-%m-%d")
        end = (start_dt + pd.DateOffset(years=window_years)).strftime("%Y-%m-%d")

        strat_returns = []
        bench_returns = []
        n = 0
        for sym in symbols:
            df = provider.get_daily(sym, start="2015-01-01", end=end)
            if df is None or len(df) < 200:
                continue
            df_sig = strat.calculate_indicators(df)
            df_sig = df_sig[(df_sig.index >= start) & (df_sig.index <= end)]
            if len(df_sig) < 50:
                continue
            engine = BacktestEngine(initial_capital=initial_capital)
            try:
                bench = engine.run(strat, df_sig)
                r = engine.get_result(bench)
                strat_returns.append(r.total_return_pct)
                bench_returns.append(r.buy_hold_return_pct)
                n += 1
            except ValueError:
                continue

        if n >= 3:
            avg_ret = np.mean(strat_returns)
            avg_bench = np.mean(bench_returns)
            alpha = avg_ret - avg_bench
            results.append({
                "start": start, "end": end,
                "strategy_return": avg_ret,
                "benchmark_return": avg_bench,
                "alpha": alpha,
                "symbols": n,
            })

    if not results:
        print("No results")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(df.to_string(index=False, formatters={
        "strategy_return": "{:+.1f}%".format,
        "benchmark_return": "{:+.1f}%".format,
        "alpha": "{:+.1f}%".format,
    }))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(df["start"], df["strategy_return"], "g-o", label="策略收益", markersize=4)
    ax.plot(df["start"], df["benchmark_return"], color="gray", linestyle="--", label="买入持有", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("收益 %")
    ax.legend(fontsize=9)
    ax.set_title("滚动窗口 α 衰减分析 (2年窗口, 3月步进)", fontsize=11)

    ax = axes[1]
    alphas = df["alpha"].values
    colors = ["#2ca02c" if a >= 0 else "#d62728" for a in alphas]
    ax.bar(range(len(df)), alphas, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["start"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Alpha %")
    # Trend line
    if len(alphas) >= 3:
        z = np.polyfit(range(len(alphas)), alphas, 1)
        trend = np.poly1d(z)
        ax.plot(range(len(alphas)), trend(range(len(alphas))), "b--", linewidth=1,
                label=f"趋势 (slope={z[0]:.2f})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = Path("reports/rolling_alpha.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"\n图表已保存: {path}")

    return df


if __name__ == "__main__":
    run()
