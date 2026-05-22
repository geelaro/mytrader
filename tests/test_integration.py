"""Integration smoke tests — signal-to-order flow with LiveTrader."""

from unittest.mock import MagicMock, patch

import pytest

from broker import MockBroker, OrderSide
from live_trader import LiveTrader


class TestLiveTraderSmoke:
    """Minimal end-to-end: signals → LiveTrader → orders → no crash."""

    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader._watchlist_symbols = ["AAPL"]
        trader.risk._day_start_equity = 100000
        trader.cache.init_schema()
        trader.cache.conn.execute("DELETE FROM risk_state")
        trader.cache.conn.execute("DELETE FROM entry_prices")
        trader.cache.conn.commit()
        trader._entry_prices = {}
        return trader

    def test_signal_to_order_flow_no_crash(self, ohlcv):
        """Scan signals, feed to LiveTrader, verify run() completes clean."""
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        trader._classify_market_state = lambda target_date: None

        orders = trader.run(target_date="2021-03-01")
        assert isinstance(orders, list)
