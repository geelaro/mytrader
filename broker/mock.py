"""MockBroker — in-memory simulated broker for testing LiveTrader.

Simulates:
- Market orders fill immediately at last price + slippage
- Limit orders fill at limit price (or better) when price is within range
- Commission charged on fills
- Position tracking across buy/sell
"""

import logging
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import (
    Account,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)


class MockBroker(Broker):
    """In-memory broker — full fill simulation, no network calls.

    Usage:
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}  # update before trading
        order = broker.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, ...))
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        commission_rate: float = 0.0003,
        slippage_pct: float = 0.0005,
        short_margin_ratio: float = 1.5,
        short_borrow_rate: float = 0.08,
        auto_accrue: bool = True,
    ):
        super().__init__()
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._commission_rate = commission_rate
        self._slippage_pct = slippage_pct
        # Reg-T style maintenance margin — short proceeds inflate _cash but
        # are locked at this ratio of notional and not part of available_cash.
        self._short_margin_ratio = short_margin_ratio
        # Annualised borrow fee charged on open short positions. Typical US
        # equity GC rate ≈ 0.25%-2%; hard-to-borrow names can hit 5-30%.
        # 0.08 is a conservative average for the watchlist universe.
        self._short_borrow_rate = short_borrow_rate
        self._auto_accrue = auto_accrue
        self._last_accrual_date: Optional[date] = None
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._realized_pnl = 0.0
        self._short_interest_paid = 0.0

        # External data — must be set before trading
        self.last_prices: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "mock"

    def refresh_prices(self, symbols: List[str]):
        """Mock: retain any existing prices; LiveTrader fills from OHLCV data."""
        logger.info("行情刷新 (Tencent/Sina, %d 个标的)", len(symbols))

    # ------------------------------------------------------------------
    # Account / Positions
    # ------------------------------------------------------------------

    def get_account(self) -> Account:
        positions = self.get_positions()
        # Short proceeds inflate _cash; deduct the maintenance margin so
        # callers cannot re-spend them on long entries.
        short_margin = sum(
            abs(p.market_value) * self._short_margin_ratio
            for p in positions if p.quantity < 0
        )
        avail = max(0.0, self._cash - short_margin)
        # Equity = cash + signed market value (long positive, short negative).
        total = self._cash + sum(p.market_value for p in positions)
        return Account(
            total_equity=total,
            available_cash=avail,
            frozen_margin=short_margin,
            total_pnl=total - self._initial_cash,
        )

    def get_historical_kline(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_positions(self) -> List[Position]:
        # Update market values from last prices
        result = []
        for sym, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            last = self.last_prices.get(sym, pos.avg_price)
            mv = pos.quantity * last
            upnl = (last - pos.avg_price) * pos.quantity
            result.append(Position(
                symbol=sym,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                market_value=mv,
                unrealized_pnl=upnl,
                realized_pnl=pos.realized_pnl,
            ))
        return result

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def accrue_short_interest(self, as_of: Optional[date] = None) -> float:
        """Charge accrued borrow fees on open shorts up to ``as_of``.

        Returns the dollar amount deducted from cash. Days are calendar days
        between the previous accrual and ``as_of``; if this is the first
        call, the accrual baseline is set without charging anything.
        """
        if as_of is None:
            as_of = date.today()
        if self._last_accrual_date is None:
            self._last_accrual_date = as_of
            return 0.0
        days = (as_of - self._last_accrual_date).days
        if days <= 0:
            return 0.0
        interest = 0.0
        for sym, pos in self._positions.items():
            if pos.quantity >= 0:
                continue
            ref_price = self.last_prices.get(sym, pos.avg_price)
            short_notional = abs(pos.quantity) * ref_price
            interest += short_notional * self._short_borrow_rate * days / 365.0
        if interest > 0:
            self._cash -= interest
            self._short_interest_paid += interest
            logger.info("短头借券利息 %d 天: -$%.2f", days, interest)
        self._last_accrual_date = as_of
        return interest

    def submit_order(self, order: Order) -> Order:
        # Charge any pending short-borrow interest before order math so cash
        # / available_cash reflect today's true buying power.
        if self._auto_accrue:
            self.accrue_short_interest()
        order.order_id = str(uuid.uuid4())[:8]
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now().isoformat()
        self._orders[order.order_id] = order

        last = self.last_prices.get(order.symbol)
        if last is None or last <= 0:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now().isoformat()
            return order

        # Simulate fill
        fill_price = self._simulate_fill(order, last)
        if fill_price <= 0:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now().isoformat()
            return order

        self._execute_fill(order, fill_price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now().isoformat()
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate_fill(self, order: Order, last_price: float) -> float:
        """Determine fill price for an order."""
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                return last_price * (1 + self._slippage_pct)
            else:
                return last_price * (1 - self._slippage_pct)

        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and order.price >= last_price:
                return min(order.price, last_price)  # fill at better price
            if order.side == OrderSide.SELL and order.price <= last_price:
                return max(order.price, last_price)  # fill at better price
            return 0  # limit order not reachable

        return 0

    def _execute_fill(self, order: Order, fill_price: float):
        """Update cash + positions after a fill (long, short, cover)."""
        commission = fill_price * order.quantity * self._commission_rate

        if order.side == OrderSide.BUY:
            total_cost = fill_price * order.quantity + commission
            existing = self._positions.get(order.symbol)
            is_cover = existing is not None and existing.quantity < 0
            # Cover spends raw cash (releases short margin). New/added long
            # entries must respect available_cash so locked short margin in
            # other symbols cannot fund them.
            if is_cover:
                spendable = self._cash
            else:
                short_margin = sum(
                    abs(p.quantity) * self.last_prices.get(s, p.avg_price) * self._short_margin_ratio
                    for s, p in self._positions.items() if p.quantity < 0
                )
                spendable = max(0.0, self._cash - short_margin)
            if total_cost > spendable:
                order.status = OrderStatus.REJECTED
                return
            self._cash -= total_cost
            self._update_position(order.symbol, order.quantity, fill_price)
        else:  # SELL
            pos = self._positions.get(order.symbol)
            if pos is not None and pos.quantity > 0:
                # Close/reduce long — must have enough shares
                if pos.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    return
            # else: no position or short position → open/add to short (allowed)
            proceeds = fill_price * order.quantity - commission
            self._cash += proceeds
            self._update_position(order.symbol, -order.quantity, fill_price)

        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = datetime.now().isoformat()

    def _update_position(self, symbol: str, delta: int, price: float):
        """Merge delta into position, tracking realized PnL."""
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol, quantity=0, avg_price=0,
                market_value=0, unrealized_pnl=0, realized_pnl=0,
            )
        pos = self._positions[symbol]
        old_qty = pos.quantity

        if (old_qty > 0 and delta < 0) or (old_qty < 0 and delta > 0):
            # Closing or reducing — realize PnL
            closed = min(abs(old_qty), abs(delta))
            pnl_per_share = price - pos.avg_price
            if old_qty < 0:
                pnl_per_share = -pnl_per_share
            pos.realized_pnl += pnl_per_share * closed

        new_qty = old_qty + delta
        if new_qty == 0:
            pos.avg_price = 0
        elif abs(new_qty) > abs(old_qty):
            # Adding to position
            added = abs(new_qty) - abs(old_qty)
            pos.avg_price = (pos.avg_price * abs(old_qty) + price * added) / abs(new_qty)
        # else: reducing position, avg_price stays

        pos.quantity = new_qty
