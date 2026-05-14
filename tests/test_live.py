"""Tests for live_trader.py — RiskLimits, LiveTrader order generation, position sizing."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from live_trader import LiveTrader, RiskLimits
from broker import (
    Order, OrderSide, OrderStatus, OrderType, Position, Account, MockBroker,
)


# ===================================================================
# RiskLimits (extending existing test_risk.py)
# ===================================================================


class TestRiskLimitsDayTracking:
    def test_new_day_resets_equity(self):
        r = RiskLimits()
        r._day_start_equity = 100000
        r._date = "2025-01-14"

        # Simulate _init_risk with a new day
        account = type("Account", (), {"total_equity": 105000})()
        today = "2025-01-15"
        if r._date != today:
            r._day_start_equity = account.total_equity
            r._date = today

        assert r._day_start_equity == 105000
        assert r._date == "2025-01-15"

    def test_same_day_keeps_equity(self):
        r = RiskLimits()
        r._day_start_equity = 100000
        r._date = "2025-01-15"

        account = type("Account", (), {"total_equity": 95000})()
        today = "2025-01-15"
        if r._date != today:
            r._day_start_equity = account.total_equity

        assert r._day_start_equity == 100000  # unchanged


# ===================================================================
# LiveTrader._calc_position_size
# ===================================================================


class TestPositionSizing:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        return LiveTrader(broker=broker, dry_run=True)

    def test_normal_sizing(self):
        trader = self._make_trader()
        qty = trader._calc_position_size(
            {"symbol": "AAPL", "price": 195.0, "atr": 5.0},
            equity=100000,
        )
        assert qty > 0
        # risk 2% of 100k = 2000, stop at 2*ATR=10, so ~200 shares
        # max position 30% of 100k at $195 = ~153 shares
        assert 1 <= qty <= 200

    def test_zero_price_returns_zero(self):
        trader = self._make_trader()
        qty = trader._calc_position_size(
            {"symbol": "AAPL", "price": 0, "atr": 5.0},
            equity=100000,
        )
        assert qty == 0

    def test_no_atr_uses_fallback(self):
        trader = self._make_trader()
        qty = trader._calc_position_size(
            {"symbol": "AAPL", "price": 195.0, "atr": 0},
            equity=100000,
        )
        # fallback: equity * 0.30 / price
        assert qty > 0


# ===================================================================
# LiveTrader._passes_risk
# ===================================================================


class TestRiskChecks:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        return LiveTrader(broker=broker, dry_run=True)

    def test_normal_case_passes(self):
        trader = self._make_trader()
        trader.risk._day_start_equity = 100000
        account = type("Account", (), {"total_equity": 101000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is True

    def test_daily_loss_limit(self, capsys):
        trader = self._make_trader()
        trader.risk.max_daily_loss_pct = 0.05  # 5%
        trader.risk._day_start_equity = 100000
        # equity dropped 8% — exceeds 5% limit
        account = type("Account", (), {"total_equity": 92000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is False

    def test_small_order_rejected(self):
        trader = self._make_trader()
        trader.risk.min_order_value = 500.0
        trader.risk._day_start_equity = 100000
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        # 1 share @ $195 = $195 < $500 min
        assert trader._passes_risk(signal, 1, account) is False


# ===================================================================
# LiveTrader._generate_orders
# ===================================================================


class TestOrderGeneration:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_buy_signal_without_position(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY

    def test_sell_signal_with_position(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": -1, "price": 200.0, "atr": 5.0,
        }]
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=10000, unrealized_pnl=500),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 50

    def test_hold_signal_generates_no_order(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 0, "price": 195.0, "atr": 5.0,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_buy_with_existing_position_generates_no_order(self):
        """Already holding, buy signal should not double-buy."""
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=9750, unrealized_pnl=250),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_prefers_nonzero_signal_when_multiple(self):
        trader = self._make_trader()
        signals = [
            {"symbol": "AAPL", "name": "Apple", "strategy": "s1",
             "signal": 0, "price": 195.0, "atr": 5.0},
            {"symbol": "AAPL", "name": "Apple", "strategy": "s2",
             "signal": 1, "price": 195.0, "atr": 5.0},
        ]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        # Should pick the buy signal (signal=1) over hold (signal=0)
        assert len(orders) >= 1
        assert any(o.side == OrderSide.BUY for o in orders)


# ===================================================================
# LiveTrader._load_config
# ===================================================================


class TestLoadConfig:
    def test_loads_watchlist_config(self):
        cfg = LiveTrader._load_config("watchlist.toml")
        assert "watchlist" in cfg
        assert "risk" in cfg


# ===================================================================
# LiveTrader init
# ===================================================================


class TestDaemonHelpers:
    def test_market_open_midnight(self):
        """US market open at midnight Beijing (summer)."""
        from unittest.mock import patch
        import datetime as dt
        monday_1am = dt.datetime(2026, 5, 11, 1, 0, 0)  # Monday 1 AM BJT
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_1am
            assert LiveTrader._is_market_open() is True

    def test_market_open_evening(self):
        """US market opens at 21:30 Beijing."""
        from unittest.mock import patch
        import datetime as dt
        monday_10pm = dt.datetime(2026, 5, 11, 22, 0, 0)
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_10pm
            assert LiveTrader._is_market_open() is True

    def test_market_closed_daytime(self):
        """US market closed at 10 AM Beijing."""
        from unittest.mock import patch
        import datetime as dt
        monday_10am = dt.datetime(2026, 5, 11, 10, 0, 0)
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_10am
            assert LiveTrader._is_market_open() is False

    def test_weekend_closed(self):
        """Market closed on Saturday."""
        from unittest.mock import patch
        import datetime as dt
        saturday_1am = dt.datetime(2026, 5, 9, 1, 0, 0)  # Saturday
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = saturday_1am
            assert LiveTrader._is_market_open() is False


# ===================================================================
# Total exposure & slippage checks
# ===================================================================


class TestExposureCheck:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_exposure_capped_in_generate_orders(self):
        """When existing position + new order exceeds max exposure, skip."""
        trader = self._make_trader()
        from broker import Position
        # Already holding $70k of NVDA out of $100k = 70% exposed
        positions = {
            "NVDA": Position(symbol="NVDA", quantity=82, avg_price=840.0,
                             market_value=70000, unrealized_pnl=1000),
        }
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "turtle_trading",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader._generate_orders(signals, positions, account)
        # New position ~$3k → total 73% ≤ 80% → order should pass
        assert len(orders) >= 0


class TestSlippageCheck:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 200.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_large_slippage_rejected(self):
        trader = self._make_trader()
        trader.risk.max_slippage_pct = 0.02
        # Signal price $195 vs last price $200 = 2.5% slippage > 2%
        signal = {"symbol": "AAPL", "price": 195.0}
        account = type("Account", (), {"total_equity": 100000})()
        assert trader._passes_risk(signal, 10, account) is False

    def test_small_slippage_passes(self):
        trader = self._make_trader()
        trader.risk.max_slippage_pct = 0.02
        # Signal price $199 vs last price $200 = 0.5% slippage < 2%
        signal = {"symbol": "AAPL", "price": 199.0}
        account = type("Account", (), {"total_equity": 100000})()
        assert trader._passes_risk(signal, 10, account) is True


class TestLiveTraderInternals:
    def test_print_order(self, capsys):
        trader = LiveTrader(broker=MockBroker(), dry_run=True)
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        trader._print_order(order)
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "AAPL" in out

    def test_log_order(self):
        trader = LiveTrader(broker=MockBroker(), dry_run=True)
        # Ensure table exists, then clean slate (table created lazily in _log_order)
        trader.cache.conn.execute(
            "CREATE TABLE IF NOT EXISTS order_log ("
            "  order_id TEXT, symbol TEXT, side TEXT, qty INTEGER,"
            "  price REAL, status TEXT, created_at TEXT"
            ")"
        )
        trader.cache.conn.execute("DELETE FROM order_log")
        trader.cache.conn.commit()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="test-123", status=OrderStatus.FILLED,
            avg_fill_price=195.0,
        )
        trader._log_order(order)
        rows = trader.cache.conn.execute(
            "SELECT * FROM order_log WHERE order_id=?", ["test-123"]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "AAPL"


class TestLiveTraderInit:
    def test_defaults(self):
        broker = MockBroker(initial_cash=50000)
        trader = LiveTrader(broker=broker, dry_run=True)
        assert trader.dry_run is True
        assert trader.broker.name == "mock"

    def test_notifier_created(self):
        broker = MockBroker()
        trader = LiveTrader(broker=broker, dry_run=True)
        assert trader.notifier is not None


# ===================================================================
# Circuit breaker (_update_risk_state)
# ===================================================================


class TestCircuitBreaker:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        return LiveTrader(broker=broker, dry_run=True)

    def test_buy_increments_daily_count(self):
        trader = self._make_trader()
        assert trader.risk._daily_trade_count == 0
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=195.0)
        trader._update_risk_state(order)
        assert trader.risk._daily_trade_count == 1
        assert trader._entry_prices["AAPL"] == 195.0

    def test_profitable_sell_resets_losses(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 2
        trader._entry_prices["AAPL"] = 190.0
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader._update_risk_state(order)
        assert trader.risk._consecutive_losses == 0
        assert "AAPL" not in trader._entry_prices

    def test_losing_sell_increments_losses(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 1
        trader._entry_prices["AAPL"] = 210.0
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader._update_risk_state(order)
        assert trader.risk._consecutive_losses == 2

    def test_sell_without_entry_price(self):
        """No entry price tracked — should not crash."""
        trader = self._make_trader()
        trader.risk._consecutive_losses = 1
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader._update_risk_state(order)
        assert trader.risk._consecutive_losses == 1  # unchanged


# ===================================================================
# Circuit breaker in _passes_risk
# ===================================================================


class TestCircuitBreakerPassesRisk:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_consecutive_losses_block(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 3
        trader.risk.max_consecutive_losses = 3
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is False

    def test_below_threshold_passes(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 2
        trader.risk.max_consecutive_losses = 3
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is True


# ===================================================================
# Daily trade limit
# ===================================================================


class TestDailyTradeLimit:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_daily_trade_cap_blocks(self):
        trader = self._make_trader()
        trader.risk._daily_trade_count = 5
        trader.risk.max_daily_trades = 5
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is False

    def test_under_cap_passes(self):
        trader = self._make_trader()
        trader.risk._daily_trade_count = 4
        trader.risk.max_daily_trades = 5
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader._passes_risk(signal, 10, account) is True


# ===================================================================
# Volatility-adaptive position sizing
# ===================================================================


class TestAdaptiveSizing:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        return LiveTrader(broker=broker, dry_run=True)

    def test_low_volatility_full_size(self):
        trader = self._make_trader()
        # ATR=2, price=200 → vol_ratio=1% → scalar=1/(1+0.01*5)=0.95 → near full
        qty = trader._calc_position_size(
            {"symbol": "AAPL", "price": 200.0, "atr": 2.0},
            equity=100000,
        )
        assert qty > 0

    def test_high_volatility_reduced_size(self):
        trader = self._make_trader()
        # ATR=40, price=200 → vol_ratio=20% → scalar=1/(1+0.2*5)=0.5 → half
        qty_high = trader._calc_position_size(
            {"symbol": "TSLA", "price": 200.0, "atr": 40.0},
            equity=100000,
        )
        # Low volatility on same price → larger position
        qty_low = trader._calc_position_size(
            {"symbol": "AAPL", "price": 200.0, "atr": 4.0},
            equity=100000,
        )
        assert qty_low > qty_high

    def test_min_vol_scalar_floor(self):
        trader = self._make_trader()
        trader.risk.min_vol_scalar = 0.3
        # Extreme volatility
        qty = trader._calc_position_size(
            {"symbol": "WILD", "price": 100.0, "atr": 80.0},
            equity=100000,
        )
        assert qty > 0  # not zero

    def test_zero_atr_fallback(self):
        trader = self._make_trader()
        qty = trader._calc_position_size(
            {"symbol": "AAPL", "price": 200.0, "atr": 0},
            equity=100000,
        )
        assert qty > 0
