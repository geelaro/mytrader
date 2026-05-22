"""LiveTrader — signal-to-order execution engine.

Flow
----
1. Load watchlist config
2. Fetch latest data → run strategies → get signals
3. Query broker for current positions & account
4. Compare signals vs positions → decide BUY / SELL / HOLD
5. Risk check → submit orders via broker
6. Log everything

Usage
-----
    pipenv run python live_trader.py              # one-shot
    pipenv run python live_trader.py --daemon     # run on schedule (requires cron/scheduler)
    pipenv run python live_trader.py --dry-run    # print orders without submitting
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP, SIGNAL_LABEL
from broker import (
    Broker,
    MockBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from engine.execution import ExecutionConfig, ExecutionModel, ExecutionStyle, ExecutionTiming
from utils import get_logger, load_toml
from utils.risk import RiskLimits  # re-exported for backward compat
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from utils.signal_gate import SignalGate
from utils.signal_scanner import SignalScanner
from utils.notify import Notifier
from live.risk_controller import RiskController
from live.order_manager import OrderManager

logger = get_logger("live")

# ---------------------------------------------------------------------------
# LiveTrader
# ---------------------------------------------------------------------------


class LiveTrader:
    """Signal-to-order bridge.

    Parameters
    ----------
    broker : Broker
        Any broker adapter implementing the Broker interface.
    config_path : str
        Path to watchlist.toml.
    dry_run : bool
        If True, print orders but don't submit.
    """

    def __init__(
        self,
        broker: Broker,
        config_path: str = "watchlist.toml",
        dry_run: bool = False,
        notifier: Optional[Notifier] = None,
    ):
        self.broker = broker
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        self.cache = CacheManager()
        self.provider = DataProvider(cache=self.cache)
        self.risk = RiskLimits.from_config(self.config)
        self.execution_model = self._load_execution_model(self.config)
        self.notifier = notifier or Notifier(dry_run=True)
        self._orphan_strategy = self.config.get("orphan", {}).get("strategy", "")
        self._watchlist_symbols: List[str] = []  # populated in run()
        self._orphan_symbols: set = set()  # symbols in positions but not watchlist
        ms = self.config.get("market_state", {})
        self._ms_proxy = ms.get("proxy_symbol", "SPY")
        self._ms_vol_scalar = ms.get("vol_high_scalar", 0.7)
        self._ms_enabled = ms.get("enabled", False)
        self._market_state = None  # set during run()
        self.risk_ctrl = RiskController(risk=self.risk, cache=self.cache,
                                        broker=self.broker, notifier=self.notifier)
        self._gate = SignalGate(ms_enabled=self._ms_enabled,
                                max_total_exposure_pct=self.risk.max_total_exposure_pct)
        self.order_mgr = OrderManager(
            broker=self.broker,
            cache=self.cache,
            execution_model=self.execution_model,
            notifier=self.notifier,
            gate=self._gate,
            risk_ctrl=self.risk_ctrl,
            dry_run=self.dry_run,
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, target_date: Optional[str] = None) -> List[Order]:
        """Run one trading cycle.

        1. Warmup broker (connections, trade contexts)
        2. Refresh market prices (watchlist + positions)
        3. Get account & positions
        4. Generate signals
        5. Compare & generate orders
        6. Submit orders
        """
        if target_date is None:
            target_date = date.today().isoformat()

        self.cache.enable_batch()

        print(f"\n{'=' * 60}")
        print(f"  LiveTrader — {target_date}")
        print(f"  券商: {self.broker.name}  {'[模拟模式]' if self.dry_run else '[实盘模式]'}")
        print(f"{'=' * 60}")

        # 1. Warmup broker (trade contexts, connections)
        self._watchlist_symbols = [item["symbol"] for item in self.config.get("watchlist", [])]
        self.broker.warmup(self._watchlist_symbols)

        # 2. Prime market prices before broker snapshots.  Some broker
        # adapters compute position market value from last_prices.
        self._refresh_market_prices()

        # 3. Get broker state. If there are orphan positions outside the
        # watchlist, refresh those prices too and re-read broker state so risk
        # checks use the freshest available values.
        account = self.broker.get_account()
        all_positions = self.broker.get_positions()
        if any(p.symbol not in self._watchlist_symbols for p in all_positions):
            self._refresh_market_prices(all_positions)
            account = self.broker.get_account()
            all_positions = self.broker.get_positions()
        positions = {p.symbol: p for p in all_positions}
        self.risk_ctrl.init_risk(account)

        self.risk_ctrl.check_global(account, positions)

        print(f"\n  账户权益: ${account.total_equity:,.0f}  "
              f"可用: ${account.available_cash:,.0f}  "
              f"持仓: {len(positions)} 个")

        if positions:
            # Identify orphans before display
            orphan_syms = {s for s in positions if s not in self._watchlist_symbols}
            print(f"\n  当前持仓:")
            for sym, pos in positions.items():
                tag = " [孤儿]" if sym in orphan_syms else ""
                print(f"    {sym:<8s}  {pos.quantity:>5} 股  "
                      f"均价 ${pos.avg_price:.2f}  市值 ${pos.market_value:,.0f}  "
                      f"浮盈 ${pos.unrealized_pnl:+,.0f}{tag}")

        # 4. Market state classification → build gate
        self._market_state = self._classify_market_state(target_date)
        self._gate = SignalGate(
            ms_enabled=self._ms_enabled,
            market_state=self._market_state,
            trading_paused=self.risk_ctrl.trading_paused,
            pause_reason=self.risk_ctrl.pause_reason,
            max_total_exposure_pct=self.risk.max_total_exposure_pct,
            vol_high_scalar=self._ms_vol_scalar,
        )
        self.order_mgr.gate = self._gate

        # 5. Generate signals
        signals = self._scan_signals(target_date, positions)

        # Real-time price override (FutuBroker)
        for s in signals:
            live_price = self.broker.last_prices.get(s['symbol'], 0)
            if live_price > 0:
                s['price'] = live_price

        # 6. Compare → Orders
        orders = self.order_mgr.generate_orders(signals, positions, account)

        # 7. Submit — process sells first so exposure drops for subsequent buys
        submitted = []
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        for order in sell_orders + buy_orders:
            signal_price = self.broker.last_prices.get(order.symbol, 0)
            if self.dry_run:
                order.status = OrderStatus.FILLED
                fill_price = signal_price
                if fill_price > 0:
                    order.avg_fill_price = fill_price * (1 + 0.0005) if order.side == OrderSide.BUY else fill_price * (1 - 0.0005)
                order.filled_qty = order.quantity
                self.order_mgr.print_order(order)
                submitted.append(order)
                self.notifier.trade_card(order)
            else:
                result = self.order_mgr.submit_and_wait(order)
                self.order_mgr.log_order(result)
                submitted.append(result)
                status = "✓" if result.status == OrderStatus.FILLED else "✗"
                msg = (f"{status} {result.symbol} {result.side.value} "
                       f"{result.filled_qty}股 @ ${result.avg_fill_price:.2f} "
                       f"[{result.order_id}] {result.status.value}")
                print(f"  {msg}")
                logger.info("订单: %s", msg)
                if result.status == OrderStatus.FILLED:
                    self.notifier.trade_card(result)
                    self.order_mgr.record_slippage(result, signal_price)
                elif result.status == OrderStatus.REJECTED:
                    self.notifier.error(f"订单被拒: {result.symbol} {result.side.value}",
                                        str(result.broker_data))
                order = result

            # Track risk state
            if order.status == OrderStatus.FILLED:
                self.risk_ctrl.on_trade_filled(order)
                if order.side == OrderSide.SELL:
                    account = self.broker.get_account()
                    positions_dict = {p.symbol: p for p in self.broker.get_positions()}
                    self.risk_ctrl.check_global(account, positions_dict)

        if not orders:
            print("\n  无新订单 — 信号与持仓一致")
            logger.info("无新订单 — 信号与持仓一致")

        print(f"\n  {'=' * 60}\n")
        self.cache.commit_batch()
        return submitted

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _refresh_market_prices(self, all_positions: list = None):
        """Fetch latest prices for watchlist + existing positions."""
        symbols = [item["symbol"] for item in self.config.get("watchlist", [])]
        if all_positions:
            for p in all_positions:
                if p.symbol not in symbols:
                    symbols.append(p.symbol)

        if not symbols:
            return

        try:
            self.broker.refresh_prices(symbols)
            updated = {s: self.broker.last_prices.get(s, 0)
                       for s in symbols if self.broker.last_prices.get(s, 0) > 0}
            if updated:
                logger.info("行情刷新: %d 个标的", len(updated))
                for s, p in updated.items():
                    print(f"  {s:<8s} ${p:.2f}")
        except Exception as e:
            logger.warning("行情刷新失败: %s", e)

    # ------------------------------------------------------------------
    # Market state
    # ------------------------------------------------------------------

    def _classify_market_state(self, target_date: str):
        """Fetch proxy data, classify market regime + volatility."""
        if not self._ms_enabled:
            return None

        try:
            lookback = self.config.get("scanner", {}).get("lookback_years", 3)
            start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback)).strftime("%Y-%m-%d")
            df = self.provider.get_daily(self._ms_proxy, start=start, end=target_date)
            if df is None or df.empty:
                logger.warning("市场状态: %s 无数据，跳过", self._ms_proxy)
                return None

            classifier = MarketStateClassifier(df)
            state = classifier.classify()
            print(f"\n  市场状态: {state.regime.name}  |  波动率: {state.volatility.name}"
                  f"  (ADX={state.adx:.1f}  BB带宽={state.bb_width_pct:.0f}%)")
            logger.info("市场状态: regime=%s vol=%s adx=%.1f bb_pct=%.0f",
                        state.regime.name, state.volatility.name, state.adx, state.bb_width_pct)
            return state
        except Exception:
            logger.exception("市场状态分类失败")
            return None

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _scan_signals(self, target_date: str, positions: Dict[str, Position]) -> List[dict]:
        """Run strategies across watchlist + orphan positions, return signal dicts."""
        lookback = self.config.get("default", {}).get("lookback_years", 3)

        # --- orphan detection (live-only concern) ---
        self._orphan_symbols = set()
        orphan_positions = []
        if self._orphan_strategy:
            for sym, pos in positions.items():
                if sym not in self._watchlist_symbols and abs(pos.quantity) > 0:
                    self._orphan_symbols.add(sym)
                    orphan_positions.append({
                        "symbol": sym,
                        "name": sym,
                        "strategy": self._orphan_strategy,
                    })
                    # Thin-data fallback: try FutuBroker kline
                    start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback)).strftime("%Y-%m-%d")
                    df = self.provider.get_daily(sym, start=start, end=target_date)
                    if (df is None or len(df) < 50):
                        futu_df = self.broker.get_historical_kline(sym, start, target_date)
                        if not futu_df.empty:
                            self.cache.save(sym, futu_df, source="futu")

        # --- delegate to shared scanner ---
        scanner = SignalScanner(self.provider, lookback_years=lookback)
        results = scanner.scan(
            self.config,
            target_date=target_date,
            orphan_positions=orphan_positions,
            monitors=False,  # live trading only uses active strategy
        )

        # --- seed broker last_prices ---
        for r in results:
            if r["symbol"] not in self.broker.last_prices:
                self.broker.last_prices[r["symbol"]] = r["price"]

        return results

    # ------------------------------------------------------------------
    # Daemon
    # ------------------------------------------------------------------

    def run_daemon(
        self,
        interval_minutes: int = 5,
        market_hours_only: bool = True,
    ):
        """Run trading cycles on a schedule until interrupted.

        Parameters
        ----------
        interval_minutes : int
            Minutes between trading cycles.
        market_hours_only : bool
            If True, only trade during US market hours (Beijing time).
        """
        import signal
        import time as _time

        shutdown = False

        def _handle_signal(sig, frame):
            nonlocal shutdown
            print(f"\n  收到信号 {sig}，安全退出...")
            shutdown = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print(f"\n{'=' * 60}")
        print(f"  LiveTrader 守护进程启动")
        print(f"  券商: {self.broker.name}  {'[模拟]' if self.dry_run else '[实盘]'}")
        print(f"  周期: {interval_minutes} 分钟  "
              f"{'仅交易时段' if market_hours_only else '全天候'}")
        print(f"{'=' * 60}")

        cycle = 0
        while not shutdown:
            cycle += 1
            try:
                if market_hours_only and not self._is_market_open():
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"\n  [{now}] 休市中，等待...")
                    _time.sleep(60 * interval_minutes)
                    continue

                # New day → reset daily trade count (keep circuit breaker across days)
                today = date.today().isoformat()
                if self.risk._date != today:
                    self.risk._date = today
                    self.risk._daily_trade_count = 0
                    self.risk_ctrl.trading_paused = False
                    self.risk_ctrl.pause_reason = ""
                    self.risk_ctrl.alert_sent = False
                    self.risk_ctrl.persist_state()
                    logger.info("新交易日: %s，日交易计数重置", today)

                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] 第 {cycle} 轮")
                self.run()

            except KeyboardInterrupt:
                print("\n  用户中断，退出守护模式")
                break
            except Exception:
                logger.exception("守护进程异常")

            if shutdown:
                break

            # Sleep between cycles
            for _ in range(interval_minutes * 60):
                if shutdown:
                    break
                _time.sleep(1)

        print(f"\n  LiveTrader 守护进程已退出 (共 {cycle} 轮)\n")

    @staticmethod
    def _is_market_open() -> bool:
        """Check if US market is open (Beijing time).

        US regular hours: 9:30-16:00 ET
        Beijing (UTC+8) summer: 21:30-04:00, winter: 22:30-05:00
        """
        now = datetime.now()
        weekday = now.weekday()
        if weekday >= 5:  # Saturday/Sunday
            return False

        hour = now.hour
        minute = now.minute
        t = hour * 100 + minute

        # Rough check: 21:30 - 05:00 next day Beijing time covers US market
        # Winter close is 05:00, so use < 500 (strict) to avoid edge
        return t >= 2130 or t < 500

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        return load_toml(path)

    @staticmethod
    def _load_execution_model(config: dict) -> ExecutionModel:
        ec = config.get("execution", {})
        timing = ExecutionTiming(ec.get("timing", "next_open"))
        style = ExecutionStyle(ec.get("style", "market"))
        return ExecutionModel(ExecutionConfig(
            timing=timing,
            style=style,
            slippage_pct=ec.get("slippage_pct", 0.0005),
            commission_rate=ec.get("commission_rate", 0.0003),
            limit_timeout_seconds=ec.get("limit_timeout_seconds", 300),
            market_timeout_seconds=ec.get("market_timeout_seconds", 60),
        ))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LiveTrader — 实盘信号执行引擎")
    parser.add_argument("--config", default="watchlist.toml", help="配置文件")
    parser.add_argument("--broker", choices=["mock", "futu"], default="mock",
                        help="券商适配器: mock(模拟) / futu(富途OpenD)")
    parser.add_argument("--futu-host", default="127.0.0.1", help="FutuOpenD 地址")
    parser.add_argument("--futu-port", type=int, default=11111, help="FutuOpenD 端口 (模拟盘:11111)")
    parser.add_argument("--initial-cash", type=float, default=100000, help="初始资金 (mock模式)")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际下单")
    parser.add_argument("--date", help="交易日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--notify", action="store_true", help="推送成交/告警到飞书")
    parser.add_argument("--daemon", action="store_true", help="守护进程模式")
    parser.add_argument("--interval", type=int, default=5, help="守护模式轮询间隔(分钟)")
    parser.add_argument("--all-day", action="store_true", help="守护模式下全天候运行")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    if args.broker == "futu":
        from broker.futu import FutuBroker
        broker = FutuBroker(
            host=args.futu_host,
            port=args.futu_port,
            initial_cash=args.initial_cash,
        )
    else:
        broker = MockBroker(initial_cash=args.initial_cash)
    notifier = Notifier() if args.notify else Notifier(dry_run=True)
    trader = LiveTrader(
        broker=broker,
        config_path=args.config,
        dry_run=args.dry_run,
        notifier=notifier,
    )

    if args.daemon:
        trader.run_daemon(
            interval_minutes=args.interval,
            market_hours_only=not args.all_day,
        )
        return

    orders = trader.run(target_date=args.date)
    if orders:
        print(f"提交 {len(orders)} 笔订单:")
        for o in orders:
            print(f"  {o.order_id}  {o.side.value} {o.symbol} {o.filled_qty}股 "
                  f"@ ${o.avg_fill_price:.2f}  [{o.status.value}]")
    else:
        print("无订单")


if __name__ == "__main__":
    main()
