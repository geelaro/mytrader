"""Tests for shared ExecutionModel semantics."""

import pandas as pd

from broker import OrderSide, OrderStatus, OrderType
from engine.execution import (
    ExecutionConfig,
    ExecutionModel,
    ExecutionStyle,
    ExecutionTiming,
)


def _bar(open_=100, high=105, low=95, close=102, volume=1000):
    return pd.Series({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def test_market_next_open_applies_slippage():
    model = ExecutionModel(ExecutionConfig(
        timing=ExecutionTiming.NEXT_OPEN,
        slippage_pct=0.01,
        commission_rate=0.001,
    ))
    plan = model.make_plan("AAPL", OrderSide.BUY, 10, created_index=0)
    fill = model.execute_bar(plan, _bar(open_=100), pd.Timestamp("2025-01-02"), 1)

    assert fill.status == OrderStatus.FILLED
    assert fill.fill_price == 101
    assert fill.commission == 1.01


def test_plan_is_not_due_on_signal_bar():
    model = ExecutionModel()
    plan = model.make_plan("AAPL", OrderSide.BUY, 10, created_index=1)
    fill = model.execute_bar(plan, _bar(open_=100), pd.Timestamp("2025-01-02"), 1)
    assert fill.status == OrderStatus.SUBMITTED
    assert fill.reason == "not_due"


def test_market_next_close_uses_close():
    model = ExecutionModel(ExecutionConfig(timing=ExecutionTiming.NEXT_CLOSE))
    plan = model.make_plan("AAPL", OrderSide.SELL, 10, created_index=0)
    fill = model.execute_bar(plan, _bar(open_=100, close=90), pd.Timestamp("2025-01-02"), 1)
    assert round(fill.fill_price, 4) == 89.991


def test_limit_waits_then_times_out():
    model = ExecutionModel(ExecutionConfig(
        style=ExecutionStyle.LIMIT,
        limit_timeout_bars=1,
    ))
    plan = model.make_plan("AAPL", OrderSide.BUY, 10, created_index=0, limit_price=90)

    waiting = model.execute_bar(plan, _bar(low=95), pd.Timestamp("2025-01-02"), 1)
    expired = model.execute_bar(plan, _bar(low=95), pd.Timestamp("2025-01-03"), 2)

    assert waiting.status == OrderStatus.SUBMITTED
    assert expired.status == OrderStatus.CANCELLED


def test_participation_rate_creates_partial_fill():
    model = ExecutionModel(ExecutionConfig(max_participation_rate=0.1))
    plan = model.make_plan("AAPL", OrderSide.BUY, 100, created_index=0)
    fill = model.execute_bar(plan, _bar(volume=250), pd.Timestamp("2025-01-02"), 1)

    assert fill.status == OrderStatus.PARTIAL
    assert fill.filled_qty == 25


def test_to_broker_order_uses_limit_style():
    model = ExecutionModel(ExecutionConfig(style=ExecutionStyle.LIMIT))
    plan = model.make_plan("AAPL", OrderSide.BUY, 10, created_index=0, limit_price=99)
    order = model.to_broker_order(plan)

    assert order.order_type == OrderType.LIMIT
    assert order.price == 99
