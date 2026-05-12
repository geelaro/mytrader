"""Backtest runner — KDJ golden cross entry + MACD death cross exit, US stocks."""

import utils  # noqa: F401 — triggers env setup (encoding, matplotlib)

from trader import run_backtest, print_result, plot_result
from strategy import WeeklyMACD_KDJ

symbols = ["AAPL", "QQQ", "SPY", "NVDA", "TSLA", "AMZN", "GOOGL"]

for symbol in symbols:
    print(f"\n{'=' * 60}")
    print(f"  {symbol}")
    print(f"{'=' * 60}")

    result, df = run_backtest(
        symbol=symbol,
        start="2018-01-01",
        initial_capital=10000,
        strategy_cls=WeeklyMACD_KDJ,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        kdj_n=9,
        kdj_k=3,
        kdj_d=3,
    )

    print_result(result)
    plot_result(result, df, symbol=symbol, save_path=f"charts/backtest_{symbol}.png")
