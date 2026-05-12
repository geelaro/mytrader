"""Tests for broker/ — MockBroker and data types."""

import pytest
from broker import (
    Broker,
    MockBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Account,
)


class TestOrderDataclass:
    def test_defaults(self):
        o = Order(symbol="AAPL", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=10)
        assert o.order_id == ""
        assert o.status == OrderStatus.PENDING
        assert o.price is None

    def test_limit_order_has_price(self):
        o = Order(symbol="AAPL", side=OrderSide.BUY,
                  order_type=OrderType.LIMIT, quantity=10, price=195.0)
        assert o.price == 195.0


class TestAccountDataclass:
    def test_defaults(self):
        a = Account(total_equity=100000, available_cash=50000,
                    frozen_margin=50000, total_pnl=1000)
        assert a.currency == "USD"


# ===================================================================
# MockBroker
# ===================================================================


class TestMockBrokerInit:
    def test_name(self, mock_broker):
        assert mock_broker.name == "mock"

    def test_initial_account(self, mock_broker):
        acct = mock_broker.get_account()
        assert acct.total_equity == 100_000
        assert acct.available_cash == 100_000
        assert acct.frozen_margin == 0

    def test_no_initial_positions(self, mock_broker):
        assert mock_broker.get_positions() == []


class TestMarketOrder:
    def test_buy_fills_immediately(self, mock_broker):
        order = Order(symbol="AAPL", side=OrderSide.BUY,
                      order_type=OrderType.MARKET, quantity=10)
        result = mock_broker.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 10
        assert result.avg_fill_price > 0
        assert result.order_id != ""

    def test_buy_reduces_cash(self, mock_broker):
        order = Order(symbol="AAPL", side=OrderSide.BUY,
                      order_type=OrderType.MARKET, quantity=100)
        mock_broker.submit_order(order)
        acct = mock_broker.get_account()
        assert acct.available_cash < 100_000

    def test_sell_rejected_without_position(self, mock_broker):
        order = Order(symbol="AAPL", side=OrderSide.SELL,
                      order_type=OrderType.MARKET, quantity=10)
        result = mock_broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_sell_reduces_position(self, mock_broker):
        # Buy first
        buy = Order(symbol="AAPL", side=OrderSide.BUY,
                    order_type=OrderType.MARKET, quantity=50)
        mock_broker.submit_order(buy)
        # Then sell
        sell = Order(symbol="AAPL", side=OrderSide.SELL,
                     order_type=OrderType.MARKET, quantity=30)
        result = mock_broker.submit_order(sell)
        assert result.status == OrderStatus.FILLED
        positions = mock_broker.get_positions()
        assert positions[0].quantity == 20  # 50 - 30

    def test_full_sell_clears_position(self, mock_broker):
        mock_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        ))
        mock_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=10,
        ))
        positions = mock_broker.get_positions()
        assert len(positions) == 0


class TestPositionTracking:
    def test_avg_price_on_add(self, mock_broker):
        mock_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        ))
        mock_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        ))
        pos = mock_broker.get_positions()[0]
        assert pos.quantity == 20
        assert pos.avg_price > 0

    def test_unrealized_pnl(self, mock_broker):
        mock_broker.last_prices["AAPL"] = 195.0
        mock_broker.submit_order(Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        ))
        # Push price up
        mock_broker.last_prices["AAPL"] = 200.0
        pos = mock_broker.get_positions()[0]
        assert pos.unrealized_pnl > 0


class TestLimitOrder:
    def test_limit_buy_fills_at_limit(self, mock_broker):
        mock_broker.last_prices["AAPL"] = 190.0
        order = Order(symbol="AAPL", side=OrderSide.BUY,
                      order_type=OrderType.LIMIT, quantity=10, price=195.0)
        result = mock_broker.submit_order(order)
        # Last=190, limit=195 — should fill at 195
        assert result.status == OrderStatus.FILLED

    def test_limit_buy_not_reachable(self, mock_broker):
        mock_broker.last_prices["AAPL"] = 200.0
        order = Order(symbol="AAPL", side=OrderSide.BUY,
                      order_type=OrderType.LIMIT, quantity=10, price=195.0)
        result = mock_broker.submit_order(order)
        # Last=200, limit=195 — can't reach, rejected
        assert result.status == OrderStatus.REJECTED


class TestCancelOrder:
    def test_cancel_pending(self, mock_broker):
        order = Order(symbol="AAPL", side=OrderSide.BUY,
                      order_type=OrderType.MARKET, quantity=10)
        result = mock_broker.submit_order(order)
        # Already filled for market order, can't cancel
        assert mock_broker.cancel_order(result.order_id) is False
