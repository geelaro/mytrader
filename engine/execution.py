"""Shared execution semantics for backtest and live trading.

The model separates signal timing from fill mechanics:
- signals can be scheduled for the next open or next close
- market orders apply configured slippage
- limit orders can expire after a bar timeout
- backtests can simulate partial fills via volume participation

Live trading still delegates the actual fill to the broker, but uses the same
order plan to build broker orders and timeout policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from broker import Order, OrderSide, OrderStatus, OrderType


class ExecutionTiming(str, Enum):
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


class ExecutionStyle(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions shared by backtest and live order planning."""

    timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN
    style: ExecutionStyle = ExecutionStyle.MARKET
    slippage_pct: float = 0.0001
    commission_rate: float = 0.0003
    limit_timeout_bars: int = 1
    limit_timeout_seconds: int = 300
    market_timeout_seconds: int = 60
    max_participation_rate: float = 0.0


@dataclass
class ExecutionPlan:
    """Order intent created from a signal and executed later by the model."""

    symbol: str
    side: OrderSide
    quantity: int
    created_index: int
    reason: str = ""
    style: Optional[ExecutionStyle] = None
    limit_price: Optional[float] = None
    timeout_bars: Optional[int] = None

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY


@dataclass
class ExecutionResult:
    """Result of executing an ExecutionPlan on a backtest bar."""

    plan: ExecutionPlan
    date: pd.Timestamp
    requested_qty: int
    filled_qty: int
    fill_price: float
    gross_value: float
    commission: float
    status: OrderStatus
    reason: str = ""

    @property
    def net_cash_delta(self) -> float:
        """Cash delta from the account perspective."""
        if self.plan.side == OrderSide.BUY:
            return -(self.gross_value + self.commission)
        return self.gross_value - self.commission


class ExecutionModel:
    """Apply a consistent execution policy to backtest and live orders."""

    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()

    def make_plan(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        created_index: int,
        reason: str = "",
        limit_price: float | None = None,
        style: ExecutionStyle | None = None,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            symbol=symbol,
            side=side,
            quantity=quantity,
            created_index=created_index,
            reason=reason,
            style=style,
            limit_price=limit_price,
            timeout_bars=self.config.limit_timeout_bars,
        )

    def due(self, plan: ExecutionPlan, current_index: int) -> bool:
        return current_index > plan.created_index

    def expired(self, plan: ExecutionPlan, current_index: int) -> bool:
        timeout = plan.timeout_bars
        if timeout is None:
            timeout = self.config.limit_timeout_bars
        return current_index - plan.created_index > timeout

    def execute_bar(
        self,
        plan: ExecutionPlan,
        row: pd.Series,
        date: pd.Timestamp,
        current_index: int,
        available_qty: int | None = None,
    ) -> ExecutionResult:
        """Execute a pending plan against one OHLCV bar."""
        requested = max(0, int(plan.quantity))
        if requested <= 0:
            return self._result(plan, date, requested, 0, 0.0, OrderStatus.REJECTED, "zero_qty")

        style = plan.style or self.config.style
        if style == ExecutionStyle.LIMIT and self.expired(plan, current_index):
            return self._result(plan, date, requested, 0, 0.0, OrderStatus.CANCELLED, "limit_timeout")

        raw_price = self._raw_price(row)
        if raw_price <= 0:
            return self._result(plan, date, requested, 0, 0.0, OrderStatus.REJECTED, "bad_price")

        fill_price = raw_price
        if style == ExecutionStyle.MARKET:
            fill_price = self._apply_slippage(raw_price, plan.side)
        elif not self._limit_touched(plan, row):
            return self._result(plan, date, requested, 0, 0.0, OrderStatus.SUBMITTED, "limit_not_touched")
        elif plan.limit_price is not None:
            if plan.side == OrderSide.BUY:
                fill_price = min(plan.limit_price, raw_price)
            else:
                fill_price = max(plan.limit_price, raw_price)

        fill_qty = requested
        if available_qty is not None:
            fill_qty = min(fill_qty, max(0, int(available_qty)))
        fill_qty = self._apply_participation(fill_qty, row)
        if fill_qty <= 0:
            return self._result(plan, date, requested, 0, 0.0, OrderStatus.SUBMITTED, "partial_no_volume")

        status = OrderStatus.FILLED if fill_qty >= requested else OrderStatus.PARTIAL
        return self._result(plan, date, requested, fill_qty, fill_price, status, "")

    def to_broker_order(self, plan: ExecutionPlan) -> Order:
        style = plan.style or self.config.style
        order_type = OrderType.MARKET if style == ExecutionStyle.MARKET else OrderType.LIMIT
        return Order(
            symbol=plan.symbol,
            side=plan.side,
            order_type=order_type,
            quantity=plan.quantity,
            price=plan.limit_price,
        )

    def timeout_seconds(self, order: Order) -> int:
        if order.order_type == OrderType.LIMIT:
            return self.config.limit_timeout_seconds
        return self.config.market_timeout_seconds

    def _raw_price(self, row: pd.Series) -> float:
        if self.config.timing == ExecutionTiming.NEXT_CLOSE:
            return float(row.get("Close", 0))
        return float(row.get("Open", row.get("Close", 0)))

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return price * (1 + self.config.slippage_pct)
        return price * (1 - self.config.slippage_pct)

    def _limit_touched(self, plan: ExecutionPlan, row: pd.Series) -> bool:
        if plan.limit_price is None:
            return False
        high = float(row.get("High", row.get("Close", 0)))
        low = float(row.get("Low", row.get("Close", 0)))
        if plan.side == OrderSide.BUY:
            return low <= plan.limit_price
        return high >= plan.limit_price

    def _apply_participation(self, qty: int, row: pd.Series) -> int:
        if self.config.max_participation_rate <= 0:
            return qty
        volume = float(row.get("Volume", 0))
        if volume <= 0:
            return 0
        return min(qty, int(volume * self.config.max_participation_rate))

    def _result(
        self,
        plan: ExecutionPlan,
        date: pd.Timestamp,
        requested_qty: int,
        filled_qty: int,
        fill_price: float,
        status: OrderStatus,
        reason: str,
    ) -> ExecutionResult:
        gross = fill_price * filled_qty
        return ExecutionResult(
            plan=plan,
            date=date,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            fill_price=fill_price,
            gross_value=gross,
            commission=gross * self.config.commission_rate,
            status=status,
            reason=reason,
        )
