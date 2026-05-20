"""Daily scanner — fetch latest data, run strategies, output today's signals.

Usage:  pipenv run python daily.py          # scan today
        pipenv run python daily.py --date 2026-05-10  # scan specific day
        pipenv run python daily.py --history          # show recent signals
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP, SIGNAL_LABEL
from utils import get_logger, load_toml
from utils.notify import Notifier

logger = get_logger("daily")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "watchlist.toml") -> dict:
    return load_toml(path)


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_day(
    config: dict,
    target_date: str = None,
    provider: DataProvider = None,
    cache: CacheManager = None,
) -> list[dict]:
    """Run all strategies across all watchlist symbols for *target_date*.

    Returns list of signal dicts.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    if provider is None:
        provider = DataProvider()
    if cache is None:
        cache = CacheManager()

    default_lookback = config.get("default", {}).get("lookback_years", 3)
    start = (pd.Timestamp(target_date) - pd.DateOffset(years=default_lookback)).strftime("%Y-%m-%d")
    results = []

    for item in config.get("watchlist", []):
        symbol = item["symbol"]
        name = item.get("name", symbol)
        active_strat = item.get("active", "")
        monitor_list = item.get("monitor", [])
        strategy_names = [active_strat] + monitor_list if active_strat else monitor_list

        # Fetch data once per symbol
        df = provider.get_daily(symbol, start=start, end=target_date)
        if df is None or df.empty:
            logger.warning("数据缺失: %s 无数据", symbol)
            print(f"  ! {symbol} 无数据，跳过")
            continue

        # Data quality checks
        latest = df.index[-1]
        age_days = (pd.Timestamp(target_date) - latest).days
        if age_days > 5:
            logger.warning("数据陈旧: %s 最新K线 %s (%d天前)", symbol, latest.date(), age_days)
            print(f"  ! {symbol} 数据陈旧: 最新 {latest.date()} ({age_days}天前)")
        if len(df) < 50:
            logger.warning("数据不足: %s 仅 %d 根K线", symbol, len(df))
            print(f"  ! {symbol} K线不足: 仅 {len(df)} 根")
        if (df["Close"] <= 0).any():
            logger.warning("异常价格: %s 存在零/负收盘价", symbol)
            print(f"  ! {symbol} 异常价格: 存在零/负值")

        # Ensure target_date is in the data (use latest available if not)
        if target_date not in df.index.strftime("%Y-%m-%d"):
            latest = df.index[-1].strftime("%Y-%m-%d")
            target_bar_date = latest
        else:
            target_bar_date = target_date

        symbol_signals = []  # collect signal lines for this symbol

        for strat_name in strategy_names:
            if strat_name not in STRATEGY_MAP:
                logger.warning("未知策略: %s", strat_name)
                continue

            strat_params = config.get("strategy", {}).get(strat_name, {})
            strategy = STRATEGY_MAP[strat_name](**strat_params)

            try:
                df_sig = strategy.calculate_indicators(df)
            except Exception as e:
                print(f"  ✗ {strat_name}: 计算失败 — {e}")
                continue

            # Get last row with valid signal
            last_idx = -1
            price = float(df_sig["Close"].iloc[last_idx])
            atr = float(df_sig["ATR"].iloc[last_idx]) if "ATR" in df_sig.columns else 0
            signal = int(df_sig["Signal"].iloc[last_idx])

            # Collect indicator snapshot
            indicators = {}
            for col in df_sig.columns:
                if col not in ("Open", "High", "Low", "Close", "Volume", "Signal"):
                    val = df_sig[col].iloc[last_idx]
                    if isinstance(val, (float, int)) and not pd.isna(val):
                        indicators[col] = round(float(val), 4)

            label = SIGNAL_LABEL.get(signal, str(signal))
            if signal != 0:
                tag = " ★" if strat_name == active_strat else "  "
                symbol_signals.append(f"  {strat_name:<20s}  {label:<5s}  价格: {price:.2f}  ATR: {atr:.2f}{tag}")

            # Save to DB
            cache.save_signal(
                scan_date=target_date,
                symbol=symbol,
                strategy=strat_name,
                bar_date=target_bar_date,
                signal=signal,
                price=price,
                atr=atr,
                indicators=json.dumps(indicators, ensure_ascii=False),
            )
            results.append({
                "symbol": symbol,
                "name": name,
                "strategy": strat_name,
                "signal": signal,
                "price": price,
                "atr": atr,
                "bar_date": target_bar_date,
                "indicators": indicators,
            })

        if symbol_signals:
            print(f"\n{'─' * 60}")
            print(f"  {symbol}  {name}  ({target_bar_date})")
            print(f"{'─' * 60}")
            for line in symbol_signals:
                print(line)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a compact summary — one row per symbol, buy/sell columns with wrapping."""

    # Group by symbol
    grouped: dict[str, dict] = {}
    for r in results:
        sym = r["symbol"]
        if sym not in grouped:
            grouped[sym] = {"name": r["name"], "buy": [], "sell": [], "price": None}
        if r["signal"] == 1:
            grouped[sym]["buy"].append(r["strategy"])
            grouped[sym]["price"] = grouped[sym]["price"] or r["price"]
        elif r["signal"] == -1:
            grouped[sym]["sell"].append(r["strategy"])
            grouped[sym]["price"] = grouped[sym]["price"] or r["price"]

    # Filter to symbols with signals
    active = {k: v for k, v in grouped.items() if v["buy"] or v["sell"]}
    if not active:
        print("\n  今日无买入/卖出信号\n")
        return

    buy_total = sum(1 for v in active.values() if v["buy"])
    sell_total = sum(1 for v in active.values() if v["sell"])
    COL_W = 22
    SYM_W = 8

    print(f"\n{'=' * 72}")
    print(f"  每日回溯结果")
    print(f"{'=' * 72}")
    print(f"  {'标的':<{SYM_W}} {'买入信号':<{COL_W}} {'卖出信号':<{COL_W}} {'参考价':>8}")
    print(f"  {'─' * (SYM_W + COL_W * 2 + 10)}")

    for sym in sorted(active.keys()):
        g = active[sym]
        buys = g["buy"] if g["buy"] else ["—"]
        sells = g["sell"] if g["sell"] else ["—"]
        price_str = f"${g['price']:.2f}" if g["price"] else "—"
        max_lines = max(len(buys), len(sells))

        for line_idx in range(max_lines):
            sym_col = sym if line_idx == 0 else ""
            buy_col = buys[line_idx] if line_idx < len(buys) else ""
            sell_col = sells[line_idx] if line_idx < len(sells) else ""
            price_col = price_str if line_idx == 0 else ""
            print(f"  {sym_col:<{SYM_W}} {buy_col:<{COL_W}} {sell_col:<{COL_W}} {price_col:>8}")

    print(f"  {'─' * (SYM_W + COL_W * 2 + 10)}")
    print(f"  买入: {buy_total} 个标的  卖出: {sell_total} 个标的  总计: {len(results)} 个策略-标的组合")
    print()


# ---------------------------------------------------------------------------
# History viewer
# ---------------------------------------------------------------------------

def show_history(cache: CacheManager = None, days: int = 7):
    if cache is None:
        cache = CacheManager()
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = cache.query_signals(scan_date=since)
    if not rows:
        print(f"近 {days} 天无扫描记录")
        return

    print(f"\n{'=' * 80}")
    print(f"  近 {days} 天信号历史")
    print(f"{'=' * 80}")
    print(f"  {'日期':<12} {'标的':<8} {'策略':<22} {'信号':<6} {'价格':>8}")
    print(f"  {'─' * 56}")
    for r in rows:
        label = SIGNAL_LABEL.get(r["signal"], "?")
        print(f"  {r['scan_date']:<12} {r['symbol']:<8} {r['strategy']:<22} {label:<6} {r['price']:>8.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="每日量化回溯扫描")
    parser.add_argument("--date", help="扫描日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--config", default="watchlist.toml", help="配置文件路径")
    parser.add_argument("--history", action="store_true", help="显示近期信号历史")
    parser.add_argument("--days", type=int, default=7, help="历史查询天数 (默认7)")
    parser.add_argument("--notify", action="store_true", help="推送结果到飞书")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    cache = CacheManager()

    if args.history:
        show_history(cache, args.days)
        return

    config = load_config(args.config)
    target_date = args.date if args.date else date.today().isoformat()
    logger.info("每日回溯开始 — %s", target_date)
    print(f"每日回溯 — {target_date}")
    results = scan_day(config, target_date=target_date, cache=cache)
    print_summary(results)

    if args.notify and results:
        notifier = Notifier()
        active = [r for r in results if r["signal"] != 0]
        notifier.signal_card(active, scan_date=target_date)
        buy_n = sum(1 for r in results if r["signal"] == 1)
        sell_n = sum(1 for r in results if r["signal"] == -1)
        notifier.daily_summary(buy_n, sell_n, len(results))
        logger.info("飞书通知已发送")


if __name__ == "__main__":
    main()
