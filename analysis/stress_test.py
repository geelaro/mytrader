"""Stress-test: run current strategy on historical crisis periods.

Usage:
    pipenv run python analysis/stress_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data import DataProvider
from strategy import STRATEGY_MAP
from engine.trader import BacktestEngine
from utils import load_toml


CRISIS_PERIODS = {
    "2008 金融危机":    ("2007-06-01", "2009-06-30"),
    "2020 COVID 崩盘":  ("2019-09-01", "2020-06-30"),
    "2022 加息熊市":    ("2021-06-01", "2023-03-31"),
    "基准 (2023-2025 牛市)": ("2023-01-01", "2025-12-31"),
}


def run():
    config = load_toml("watchlist.toml")
    provider = DataProvider()
    active_strat = "weekly_macd_kdj"
    strat_cls = STRATEGY_MAP[active_strat]
    params = config.get("strategy", {}).get(active_strat, {})
    watchlist = config.get("watchlist", [])

    print(f"\n{'='*70}")
    print(f"  压力测试 — {active_strat}")
    print(f"{'='*70}")

    for period_name, (start, end) in CRISIS_PERIODS.items():
        print(f"\n{'─'*70}")
        print(f"  {period_name} ({start} → {end})")
        print(f"{'─'*70}")

        total_ret = total_sharpe = total_dd = total_trades = 0
        n = 0

        for item in watchlist:
            sym = item["symbol"]
            df = provider.get_daily(sym, start=start, end=end)
            if df is None or len(df) < 100:
                print(f"  {sym:<8s}  数据不足，跳过")
                continue

            strategy = strat_cls(**params)
            df_sig = strategy.calculate_indicators(df)
            engine = BacktestEngine(initial_capital=10000)
            try:
                bench = engine.run(strategy, df_sig)
                r = engine.get_result(bench)
            except ValueError:
                print(f"  {sym:<8s}  无交易信号，跳过")
                continue

            total_ret += r.total_return_pct
            total_sharpe += r.sharpe_ratio
            total_dd += r.max_drawdown_pct
            total_trades += r.total_trades
            n += 1

            print(f"  {sym:<8s}  收益:{r.total_return_pct:+7.1f}%  "
                  f"Sharpe:{r.sharpe_ratio:6.2f}  回撤:{r.max_drawdown_pct:+6.1f}%  "
                  f"交易:{r.total_trades:3d}  胜率:{r.win_rate_pct:5.1f}%")

        if n > 0:
            print(f"  {'─'*70}")
            print(f"  {'平均':<8s}  收益:{total_ret/n:+7.1f}%  "
                  f"Sharpe:{total_sharpe/n:6.2f}  回撤:{total_dd/n:+6.1f}%  "
                  f"交易:{total_trades/n:5.0f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run()
