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
from dataclasses import dataclass
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
from utils import get_logger, load_toml
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from utils.signal_gate import SignalGate
from utils.notify import Notifier

logger = get_logger("live")

# ---------------------------------------------------------------------------
# Risk guard
# ---------------------------------------------------------------------------


@dataclass
class RiskLimits:
    """Safety limits — checked before every order submission."""

    max_position_pct: float = 0.30
    max_total_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.05
    min_order_value: float = 500.0
    max_slippage_pct: float = 0.02
    max_consecutive_losses: int = 3
    max_daily_trades: int = 5
    base_risk_pct: float = 0.02
    vol_sensitivity: float = 5.0
    min_vol_scalar: float = 0.3

    # -- runtime state --
    _day_start_equity: float = 0.0
    _date: str = ""
    _consecutive_losses: int = 0
    _daily_trade_count: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "RiskLimits":
        rc = config.get("risk", {})
        return cls(
            max_position_pct=rc.get("max_position_pct", 0.30),
            max_total_exposure_pct=rc.get("max_total_exposure_pct", 0.80),
            max_daily_loss_pct=rc.get("max_daily_loss_pct", 0.05),
            min_order_value=rc.get("min_order_value", 500.0),
            max_slippage_pct=rc.get("max_slippage_pct", 0.02),
            max_consecutive_losses=rc.get("max_consecutive_losses", 3),
            max_daily_trades=rc.get("max_daily_trades", 5),
            base_risk_pct=rc.get("base_risk_pct", 0.02),
            vol_sensitivity=rc.get("vol_sensitivity", 5.0),
            min_vol_scalar=rc.get("min_vol_scalar", 0.3),
        )


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
        self.notifier = notifier or Notifier(dry_run=True)
        self._entry_prices: Dict[str, float] = {}  # for circuit breaker tracking
        self._orphan_strategy = self.config.get("defaults", {}).get("orphan_strategy", "")
        self._watchlist_symbols: List[str] = []  # populated in run()
        self._orphan_symbols: set = set()  # symbols in positions but not watchlist
        self._trading_paused = False
        self._pause_reason = ""
        self._alert_sent = False
        ms = self.config.get("market_state", {})
        self._ms_proxy = ms.get("proxy_symbol", "SPY")
        self._ms_vol_scalar = ms.get("vol_high_scalar", 0.7)
        self._ms_enabled = ms.get("enabled", False)
        self._market_state = None  # set during run()
        self._gate = SignalGate(ms_enabled=self._ms_enabled,
                                max_total_exposure_pct=self.risk.max_total_exposure_pct)
        self._restore_risk_state()

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

        # 2. Get broker state (positions first, so price refresh can include them)
        account = self.broker.get_account()
        all_positions = self.broker.get_positions()
        positions = {p.symbol: p for p in all_positions}

        # 3. Refresh market prices (watchlist + existing positions)
        self._refresh_market_prices(all_positions)
        self._init_risk(account)

        self._check_global_risk(account, positions)

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
            trading_paused=self._trading_paused,
            pause_reason=self._pause_reason,
            max_total_exposure_pct=self.risk.max_total_exposure_pct,
            vol_high_scalar=self._ms_vol_scalar,
        )

        # 5. Generate signals
        signals = self._scan_signals(target_date, positions)

        # Real-time price override (FutuBroker)
        for s in signals:
            live_price = self.broker.last_prices.get(s['symbol'], 0)
            if live_price > 0:
                s['price'] = live_price

        # 6. Compare → Orders
        orders = self._generate_orders(signals, positions, account)

        # 7. Submit (with polling for partial fills + timeout cancellation)
        submitted = []
        for order in orders:
            signal_price = self.broker.last_prices.get(order.symbol, 0)
            if self.dry_run:
                order.status = OrderStatus.FILLED
                fill_price = signal_price
                if fill_price > 0:
                    order.avg_fill_price = fill_price * (1 + 0.0005) if order.side == OrderSide.BUY else fill_price * (1 - 0.0005)
                order.filled_qty = order.quantity
                self._print_order(order)
                submitted.append(order)
                self.notifier.trade_card(order)
            else:
                result = self._submit_and_wait(order)
                self._log_order(result)
                submitted.append(result)
                status = "✓" if result.status == OrderStatus.FILLED else "✗"
                msg = (f"{status} {result.symbol} {result.side.value} "
                       f"{result.filled_qty}股 @ ${result.avg_fill_price:.2f} "
                       f"[{result.order_id}] {result.status.value}")
                print(f"  {msg}")
                logger.info("订单: %s", msg)
                if result.status == OrderStatus.FILLED:
                    self.notifier.trade_card(result)
                    self._record_slippage(result, signal_price)
                elif result.status == OrderStatus.REJECTED:
                    self.notifier.error(f"订单被拒: {result.symbol} {result.side.value}",
                                        str(result.broker_data))
                order = result

            # Track risk state
            if order.status == OrderStatus.FILLED:
                self._update_risk_state(order)

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
            lookback = self.config.get("default", {}).get("lookback_years", 3)
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
        default_lookback = self.config.get("default", {}).get("lookback_years", 3)
        start = (pd.Timestamp(target_date) - pd.DateOffset(years=default_lookback)).strftime("%Y-%m-%d")
        results = []

        # Collect all symbols to scan: watchlist + orphan positions
        scan_items = list(self.config.get("watchlist", []))
        self._orphan_symbols = set()
        if self._orphan_strategy:
            for sym, pos in positions.items():
                if sym not in self._watchlist_symbols and abs(pos.quantity) > 0:
                    self._orphan_symbols.add(sym)
                    scan_items.append({
                        "symbol": sym, "name": sym,
                        "active": self._orphan_strategy,
                        "_orphan": True,
                    })

        for item in scan_items:
            symbol = item["symbol"]
            name = item.get("name", symbol)
            is_orphan = item.get("_orphan", False)
            active_strat = item.get("active", "")
            strategy_names = [active_strat] if active_strat else []

            df = self.provider.get_daily(symbol, start=start, end=target_date)

            # Orphan with thin data: try FutuBroker for historical kline
            if is_orphan and (df is None or len(df) < 50) and hasattr(self.broker, "get_historical_kline"):
                futu_df = self.broker.get_historical_kline(symbol, start, target_date)
                if not futu_df.empty:
                    self.cache.save(symbol, futu_df, source="futu")
                    df = self.provider.get_daily(symbol, start=start, end=target_date)

            if df is None or df.empty:
                continue

            if target_date not in df.index.strftime("%Y-%m-%d"):
                bar_date = df.index[-1].strftime("%Y-%m-%d")
            else:
                bar_date = target_date

            for strat_name in strategy_names:
                cls = STRATEGY_MAP.get(strat_name)
                if cls is None:
                    continue
                params = self.config.get("strategy", {}).get(strat_name, {})
                strategy = cls(**params)
                try:
                    df_sig = strategy.calculate_indicators(df)
                except Exception:
                    logger.exception("策略计算失败: %s %s", symbol, strat_name)
                    continue

                last_idx = -1
                signal = int(df_sig["Signal"].iloc[last_idx])
                price = float(df_sig["Close"].iloc[last_idx])
                atr = float(df_sig["ATR"].iloc[last_idx]) if "ATR" in df_sig.columns else 0

                indicators = {}
                for col in df_sig.columns:
                    if col not in ("Open", "High", "Low", "Close", "Volume", "Signal"):
                        val = df_sig[col].iloc[last_idx]
                        if isinstance(val, (float, int)) and not pd.isna(val):
                            indicators[col] = round(float(val), 4)

                results.append({
                    "symbol": symbol, "name": name, "strategy": strat_name,
                    "signal": signal, "price": price, "atr": atr,
                    "bar_date": bar_date, "indicators": indicators,
                    "orphan": is_orphan,
                })

                if symbol not in self.broker.last_prices:
                    self.broker.last_prices[symbol] = price

        return results

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def _generate_orders(
        self, signals: List[dict], positions: Dict[str, Position], account
    ) -> List[Order]:
        """Compare signals against current positions, generate orders."""
        orders = []

        # Group signals by symbol — use strongest signal (first non-zero)
        symbol_signals: Dict[str, dict] = {}
        for s in signals:
            sym = s["symbol"]
            if sym not in symbol_signals:
                symbol_signals[sym] = s
            # Prefer buy/sell over hold
            if s["signal"] != 0 and symbol_signals[sym]["signal"] == 0:
                symbol_signals[sym] = s

        for sym, sig in symbol_signals.items():
            if sig["signal"] == 0:
                continue

            pos = positions.get(sym)
            has_position = pos is not None and abs(pos.quantity) > 0

            if sig["signal"] == 1 and not has_position:
                sig["_qty"] = self._calc_position_size(sig, account.total_equity)
                if sig["_qty"] <= 0:
                    print(f"  ! {sym} 买入信号但仓位计算为0，跳过")
                    continue

                ok, reason = self._gate.allow_buy(sig, positions, account)
                if not ok:
                    print(f"  ! {sym} {reason}，跳过")
                    self.cache.log_ops("gate_reject", symbol=sym, detail=reason, level="WARN")
                    continue

                qty = self._gate.vol_scaled_qty(sig["_qty"])
                if self._passes_risk(sig, qty, account):
                    orders.append(Order(
                        symbol=sym,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                    ))
                else:
                    print(f"  ! {sym} 风控检查未通过，跳过")

            elif sig["signal"] == -1 and has_position:
                # Sell signal + has position → SELL
                orders.append(Order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=abs(pos.quantity),
                ))

        return orders

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calc_position_size(self, signal: dict, equity: float) -> int:
        """Volatility-adaptive position sizing.

        Scales position down when ATR/price ratio is high (volatile).
        """
        price = signal.get("price", 0)
        atr = signal.get("atr", 0)
        if price <= 0:
            return 0

        r = self.risk
        risk_dollar = equity * r.base_risk_pct

        if atr > 0:
            # Volatility scalar: reduce size when stock is volatile
            vol_ratio = atr / price
            vol_scalar = 1.0 / (1.0 + vol_ratio * r.vol_sensitivity)
            vol_scalar = max(vol_scalar, r.min_vol_scalar)

            qty = int(risk_dollar / (atr * 2) * vol_scalar)
        else:
            qty = int(equity * 0.30 / price)

        max_qty = int(equity * r.max_position_pct / price)
        return max(1, min(qty, max_qty))

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _init_risk(self, account):
        today = date.today().isoformat()
        if self.risk._date != today:
            self.risk._day_start_equity = account.total_equity
            self.risk._date = today
            self._persist_risk_state()

    def _restore_risk_state(self):
        """Restore risk state and entry prices from SQLite after restart."""
        stored_date = self.cache.load_risk_state("date")
        today = date.today().isoformat()
        if stored_date == today:
            cl = self.cache.load_risk_state("consecutive_losses")
            dt = self.cache.load_risk_state("daily_trade_count")
            self.risk._date = today
            self.risk._consecutive_losses = int(cl) if cl else 0
            self.risk._daily_trade_count = int(dt) if dt else 0
        # Entry prices survive across days (needed for circuit breaker)
        self._entry_prices = {
            sym: price for sym, (price, _) in self.cache.load_all_entry_prices().items()
        }

    def _persist_risk_state(self):
        """Write current risk state to SQLite."""
        self.cache.save_risk_state("date", self.risk._date)
        self.cache.save_risk_state("consecutive_losses", str(self.risk._consecutive_losses))
        self.cache.save_risk_state("daily_trade_count", str(self.risk._daily_trade_count))

    def _passes_risk(self, signal: dict, qty: int, account) -> bool:
        """Run all risk checks before order submission."""
        r = self.risk
        equity = account.total_equity

        # Consecutive loss circuit breaker
        if r._consecutive_losses >= r.max_consecutive_losses:
            print(f"  ! 连续亏损熔断 ({r._consecutive_losses}笔)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="consecutive_losses", level="WARN")
            return False

        # Daily trade cap
        if r._daily_trade_count >= r.max_daily_trades:
            print(f"  ! 日内交易次数已达上限 ({r.max_daily_trades}笔)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="daily_trade_cap", level="WARN")
            return False

        # Daily loss limit
        if equity < r._day_start_equity * (1 - r.max_daily_loss_pct):
            print(f"  ! 日内亏损超限 ({r.max_daily_loss_pct*100:.0f}%)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="daily_loss", level="WARN")
            return False

        # Min order value
        order_value = signal.get("price", 0) * qty
        if order_value < r.min_order_value:
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="min_order_value", level="WARN")
            return False

        # Slippage guard — compare signal price vs broker last price
        sym = signal.get("symbol", "")
        signal_price = signal.get("price", 0)
        if sym in self.broker.last_prices:
            last_price = self.broker.last_prices[sym]
            if last_price > 0 and signal_price > 0:
                slippage = abs(signal_price - last_price) / last_price
                if slippage > r.max_slippage_pct:
                    print(f"  ! {sym} 滑点超限 ({slippage*100:.2f}%)，拒绝")
                    self.cache.log_ops("slippage_rejected", symbol=sym,
                                       detail=f"{slippage*100:.2f}%", value=slippage*100)
                    return False

        return True

    def _check_global_risk(self, account, positions: dict):
        """Check global risk thresholds. Pause trading (new BUYs) if breached."""
        r = self.risk
        equity = account.total_equity

        # New day → unpause
        today = date.today().isoformat()
        if r._date != today:
            self._trading_paused = False
            self._pause_reason = ""
            self._alert_sent = False

        # Daily loss limit
        if r._day_start_equity > 0 and equity < r._day_start_equity * (1 - r.max_daily_loss_pct):
            loss_pct = (r._day_start_equity - equity) / r._day_start_equity * 100
            self._trading_paused = True
            self._pause_reason = f"日内亏损超限 ({loss_pct:.1f}% > {r.max_daily_loss_pct*100:.0f}%)"
        elif r._consecutive_losses >= r.max_consecutive_losses:
            self._trading_paused = True
            self._pause_reason = f"连续亏损熔断 ({r._consecutive_losses}/{r.max_consecutive_losses})"
        elif positions:
            total_exposure = sum(p.market_value for p in positions.values() if p.market_value > 0)
            exposure_pct = total_exposure / equity if equity > 0 else 0
            if exposure_pct > r.max_total_exposure_pct:
                self._trading_paused = True
                self._pause_reason = f"总敞口超限 ({exposure_pct*100:.1f}% > {r.max_total_exposure_pct*100:.0f}%)"

        if self._trading_paused:
            print(f"\n  !! 交易暂停: {self._pause_reason}")
            if not self._alert_sent:
                self.notifier.error("交易暂停", self._pause_reason)
                self._alert_sent = True
            self.cache.log_ops("trading_paused", detail=self._pause_reason)
        else:
            self._alert_sent = False

    def _update_risk_state(self, order: Order):
        """Update circuit breaker and daily trade count after a fill."""
        r = self.risk
        sym = order.symbol
        fill_price = order.avg_fill_price

        if order.side == OrderSide.BUY:
            self._entry_prices[sym] = fill_price
            self.cache.save_entry_price(sym, fill_price, date.today().isoformat())
            r._daily_trade_count += 1
        elif order.side == OrderSide.SELL:
            entry = self._entry_prices.pop(sym, None)
            if entry is not None:
                self.cache.delete_entry_price(sym)
                self.cache.save_trade_pnl(
                    sym, "SELL", order.filled_qty, entry, fill_price,
                    date.today().isoformat(), order.order_id,
                )
                if fill_price < entry:
                    r._consecutive_losses += 1
                    logger.warning("连续亏损 %d/%d: %s  PnL=$%.2f",
                                   r._consecutive_losses, r.max_consecutive_losses,
                                   sym, (fill_price - entry) * order.filled_qty)
                else:
                    r._consecutive_losses = 0
                    logger.info("交易PnL: %s  $%.2f", sym, (fill_price - entry) * order.filled_qty)
        self._persist_risk_state()

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
                    self._trading_paused = False
                    self._pause_reason = ""
                    self._alert_sent = False
                    self._persist_risk_state()
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
    def _print_order(order: Order):
        print(f"  [DRY-RUN] {order.side.value} {order.symbol} "
              f"{order.quantity}股 "
              f"{'MARKET' if order.order_type == OrderType.MARKET else f'@ ${order.price:.2f}'}")

    def _log_order(self, order: Order):
        """Persist order to DB for audit trail."""
        self.cache.init_schema()
        self.cache.conn.execute(
            "INSERT INTO order_log VALUES (?,?,?,?,?,?,?)",
            [order.order_id, order.symbol, order.side.value,
             order.filled_qty, order.avg_fill_price,
             order.status.value, order.created_at],
        )
        self.cache._commit()

    def _submit_and_wait(self, order: Order) -> Order:
        """Submit order and poll until filled, cancelled, or timeout."""
        import time as _time

        result = self.broker.submit_order(order)
        if result.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            return result

        # Poll for completion
        is_limit = order.order_type == OrderType.LIMIT
        timeout_s = 300 if is_limit else 60  # 5 min for limit, 60 s for market
        poll_interval = 3
        elapsed = 0

        while elapsed < timeout_s:
            _time.sleep(poll_interval)
            elapsed += poll_interval
            updated = self.broker.get_order(result.order_id)
            if updated is None:
                continue
            result = updated
            if result.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                return result

        # Timeout — cancel if limit order
        if is_limit and result.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            logger.warning("限价单超时，撤单: %s %s", order.symbol, order.order_id)
            self.broker.cancel_order(result.order_id)
            result.status = OrderStatus.CANCELLED

        return result

    def _record_slippage(self, order: Order, signal_price: float):
        """Record slippage stats to DB."""
        if signal_price <= 0 or order.avg_fill_price <= 0:
            return
        slippage = (order.avg_fill_price - signal_price) / signal_price
        self.cache.log_ops("slippage", symbol=order.symbol,
                           detail=f"{slippage*100:+.2f}%", value=slippage*100)
        self.cache.init_schema()
        self.cache.conn.execute(
            "INSERT INTO slippage_log VALUES (?,?,?,?,?,?,?)",
            [order.order_id, order.symbol, order.side.value,
             signal_price, order.avg_fill_price, round(slippage * 100, 4),
             order.created_at],
        )
        self.cache._commit()


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
