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

    _day_start_equity: float = 0.0
    _date: str = ""

    @classmethod
    def from_config(cls, config: dict) -> "RiskLimits":
        rc = config.get("risk", {})
        return cls(
            max_position_pct=rc.get("max_position_pct", 0.30),
            max_total_exposure_pct=rc.get("max_total_exposure_pct", 0.80),
            max_daily_loss_pct=rc.get("max_daily_loss_pct", 0.05),
            min_order_value=rc.get("min_order_value", 500.0),
            max_slippage_pct=rc.get("max_slippage_pct", 0.02),
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
        self.provider = DataProvider()
        self.cache = CacheManager()
        self.risk = RiskLimits.from_config(self.config)
        self.notifier = notifier or Notifier(dry_run=True)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, target_date: Optional[str] = None) -> List[Order]:
        """Run one trading cycle.

        1. Fetch signals for all watchlist symbols
        2. Compare with broker positions
        3. Generate & submit orders
        """
        if target_date is None:
            target_date = date.today().isoformat()

        print(f"\n{'=' * 60}")
        print(f"  LiveTrader — {target_date}")
        print(f"  券商: {self.broker.name}  {'[模拟模式]' if self.dry_run else '[实盘模式]'}")
        print(f"{'=' * 60}")

        # 1. Get broker state
        account = self.broker.get_account()
        positions = {p.symbol: p for p in self.broker.get_positions()}
        self._init_risk(account)

        print(f"\n  账户权益: ${account.total_equity:,.0f}  "
              f"可用: ${account.available_cash:,.0f}  "
              f"持仓: {len(positions)} 个")

        if positions:
            print(f"\n  当前持仓:")
            for sym, pos in positions.items():
                print(f"    {sym:<8s}  {pos.quantity:>5} 股  "
                      f"均价 ${pos.avg_price:.2f}  市值 ${pos.market_value:,.0f}  "
                      f"浮盈 ${pos.unrealized_pnl:+,.0f}")

        # 2. Generate signals
        signals = self._scan_signals(target_date)

        # 3. Compare → Orders
        orders = self._generate_orders(signals, positions, account)

        # 4. Submit
        submitted = []
        for order in orders:
            if self.dry_run:
                self._print_order(order)
                order.status = OrderStatus.FILLED  # pretend for dry-run
            else:
                result = self.broker.submit_order(order)
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
                elif result.status == OrderStatus.REJECTED:
                    self.notifier.error(f"订单被拒: {result.symbol} {result.side.value}",
                                        str(result.broker_data))

        if not orders:
            print("\n  无新订单 — 信号与持仓一致")
            logger.info("无新订单 — 信号与持仓一致")

        print(f"\n  {'=' * 60}\n")
        return submitted

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _scan_signals(self, target_date: str) -> List[dict]:
        """Run strategies across watchlist, return signal dicts."""
        default_lookback = self.config.get("default", {}).get("lookback_years", 3)
        start = (pd.Timestamp(target_date) - pd.DateOffset(years=default_lookback)).strftime("%Y-%m-%d")
        results = []

        for item in self.config.get("watchlist", []):
            symbol = item["symbol"]
            name = item.get("name", symbol)
            strategy_names = item.get("strategies", [])

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

                # Collect indicators
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
                })

                # Update broker last prices for mock
                if hasattr(self.broker, "last_prices"):
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
                # Buy signal + no position → BUY
                qty = self._calc_position_size(sig, account.total_equity)
                if qty <= 0:
                    print(f"  ! {sym} 买入信号但仓位计算为0，跳过")
                    continue
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
        """Basic volatility-adjusted sizing."""
        price = signal.get("price", 0)
        atr = signal.get("atr", 0)
        if price <= 0:
            return 0
        # Risk 2% of equity per trade, stop at 2× ATR
        risk_dollar = equity * 0.02
        if atr > 0:
            qty = int(risk_dollar / (atr * 2))
        else:
            qty = int(equity * 0.30 / price)
        max_qty = int(equity * self.risk.max_position_pct / price)
        return max(1, min(qty, max_qty))

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _init_risk(self, account):
        today = date.today().isoformat()
        if self.risk._date != today:
            self.risk._day_start_equity = account.total_equity
            self.risk._date = today

    def _passes_risk(self, signal: dict, qty: int, account) -> bool:
        """Run all risk checks before order submission."""
        equity = account.total_equity
        # Daily loss limit
        if equity < self.risk._day_start_equity * (1 - self.risk.max_daily_loss_pct):
            print(f"  ! 日内亏损超限 ({self.risk.max_daily_loss_pct*100:.0f}%)，暂停交易")
            return False

        # Min order value
        order_value = signal.get("price", 0) * qty
        if order_value < self.risk.min_order_value:
            return False

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
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
        self.cache.conn.execute(
            "CREATE TABLE IF NOT EXISTS order_log ("
            "  order_id TEXT, symbol TEXT, side TEXT, qty INTEGER,"
            "  price REAL, status TEXT, created_at TEXT"
            ")"
        )
        self.cache.conn.execute(
            "INSERT INTO order_log VALUES (?,?,?,?,?,?,?)",
            [order.order_id, order.symbol, order.side.value,
             order.filled_qty, order.avg_fill_price,
             order.status.value, order.created_at],
        )
        self.cache.conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LiveTrader — 实盘信号执行引擎")
    parser.add_argument("--config", default="watchlist.toml", help="配置文件")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际下单")
    parser.add_argument("--date", help="交易日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--notify", action="store_true", help="推送成交/告警到飞书")
    parser.add_argument("--daemon", action="store_true", help="守护模式 (暂未实现)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    broker = MockBroker(initial_cash=100000)
    notifier = Notifier() if args.notify else Notifier(dry_run=True)
    trader = LiveTrader(
        broker=broker,
        config_path=args.config,
        dry_run=args.dry_run,
        notifier=notifier,
    )
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
