"""FutuBroker — 富途 OpenAPI 券商适配器.

Supports simulated trading (模拟盘) via FutuOpenD gateway.

Setup
-----
1. Install FutuOpenD (https://www.futunn.com/download/openAPI)
2. Launch FutuOpenD, log in with your Futu account
3. In FutuOpenD settings, enable "模拟盘" trading
4. Default simulated trading port: 11111

Usage
-----
    from broker.futu import FutuBroker
    from live_trader import LiveTrader

    broker = FutuBroker(host='127.0.0.1', port=11111, initial_cash=10000)
    trader = LiveTrader(broker=broker)
    trader.run()
"""

from typing import Dict, List, Optional
import time

from futu import (
    OpenSecTradeContext,
    OpenQuoteContext,
    TrdEnv,
    TrdMarket,
    TrdSide,
    OrderType as FutuOrderType,
    OrderStatus as FutuOrderStatus,
    ModifyOrderOp,
)

from .base import (
    Account,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


# ---------------------------------------------------------------------------
# Symbol mapping: mytrader format ↔ Futu format
# ---------------------------------------------------------------------------

def _to_futu_symbol(symbol: str) -> str:
    """Convert mytrader symbol to Futu format (US.AAPL, HK.00700, SH.510300)."""
    s = symbol.upper().strip()
    if s.startswith("US."):
        return s
    if s.startswith("HK."):
        return s
    if s[:2] in ("SH", "SZ"):
        if "." not in s:
            return f"{s[:2]}.{s[2:]}"
        return s
    if s.isdigit() and len(s) == 6:
        # A-share code
        prefix = "SH" if s[0] in ("5", "6", "9") else "SZ"
        return f"{prefix}.{s}"
    # Default: US stock
    return f"US.{s}"


def _to_market(symbol: str) -> int:
    """Map Futu symbol to TrdMarket enum."""
    s = _to_futu_symbol(symbol)
    prefix = s.split(".")[0].upper()
    if prefix in ("US", "USO", "USN"):
        return TrdMarket.US
    if prefix == "HK":
        return TrdMarket.HK
    if prefix in ("SH", "SZ"):
        return TrdMarket.CN
    return TrdMarket.US  # default


def _from_futu_side(futu_side) -> OrderSide:
    return OrderSide.BUY if futu_side == TrdSide.BUY else OrderSide.SELL


def _to_futu_side(side: OrderSide):
    return TrdSide.BUY if side == OrderSide.BUY else TrdSide.SELL


def _to_futu_order_type(ot: OrderType):
    return FutuOrderType.MARKET if ot == OrderType.MARKET else FutuOrderType.NORMAL


def _from_futu_status(futu_status) -> OrderStatus:
    """Map Futu order status to our OrderStatus enum."""
    mapping = {
        FutuOrderStatus.WAITING_SUBMIT: OrderStatus.PENDING,
        FutuOrderStatus.SUBMITTING: OrderStatus.SUBMITTED,
        FutuOrderStatus.SUBMITTED: OrderStatus.SUBMITTED,
        FutuOrderStatus.FILLED_PART: OrderStatus.PARTIAL,
        FutuOrderStatus.FILLED_ALL: OrderStatus.FILLED,
        FutuOrderStatus.CANCELLED_PART: OrderStatus.CANCELLED,
        FutuOrderStatus.CANCELLED_ALL: OrderStatus.CANCELLED,
        FutuOrderStatus.FAILED: OrderStatus.REJECTED,
        FutuOrderStatus.DISABLED: OrderStatus.REJECTED,
        FutuOrderStatus.DELETED: OrderStatus.CANCELLED,
    }
    return mapping.get(futu_status, OrderStatus.REJECTED)


# ---------------------------------------------------------------------------
# FutuBroker
# ---------------------------------------------------------------------------


class FutuBroker(Broker):
    """Futu OpenAPI broker adapter — simulated trading.

    Parameters
    ----------
    host : str
        FutuOpenD host (default 127.0.0.1).
    port : int
        FutuOpenD port (default 11111 for simulated trading).
    initial_cash : float
        Reference initial cash for P&L tracking (account ground truth from Futu).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        initial_cash: float = 10000.0,
    ):
        self._host = host
        self._port = port
        self._initial_cash = initial_cash
        self._quote_ctx: Optional[OpenQuoteContext] = None
        self._trade_ctxs: Dict[int, OpenSecTradeContext] = {}
        self.last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Establish connections to FutuOpenD."""
        self._quote_ctx = OpenQuoteContext(host=self._host, port=self._port)
        self._quote_ctx.start()

    def disconnect(self):
        """Close all connections."""
        if self._quote_ctx is not None:
            self._quote_ctx.stop()
            self._quote_ctx = None
        for ctx in self._trade_ctxs.values():
            ctx.close()
        self._trade_ctxs.clear()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def _get_trade_ctx(self, symbol: str):
        """Get or create a trade context for the symbol's market.

        Returns None if connection fails (FutuOpenD not running).
        """
        market = _to_market(symbol)
        if market not in self._trade_ctxs:
            try:
                ctx = OpenSecTradeContext(
                    host=self._host,
                    port=self._port,
                    filter_trdmarket=market,
                )
                self._trade_ctxs[market] = ctx
            except Exception:
                return None
        return self._trade_ctxs[market]

    # ------------------------------------------------------------------
    # Broker identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "futu"

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Account:
        """Query account info from Futu — aggregates across all markets."""
        equity_parts = []
        cash_parts = []
        frozen = 0.0
        has_data = False

        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.accinfo_query()
                if ret == 0 and not data.empty:
                    has_data = True
                    row = data.iloc[0]
                    equity_parts.append(float(row.get("total_assets", 0)))
                    cash_parts.append(float(row.get("cash", 0)))
                    frozen += float(row.get("frozen_cash", 0))
            except Exception:
                pass

        if has_data:
            total_equity = sum(equity_parts)
            available_cash = sum(cash_parts)
        else:
            total_equity = self._initial_cash
            available_cash = self._initial_cash

        total_pnl = total_equity - self._initial_cash if has_data else 0.0

        return Account(
            total_equity=total_equity,
            available_cash=available_cash,
            frozen_margin=frozen,
            total_pnl=total_pnl,
            currency="USD",
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        """Query current positions from Futu."""
        results = []
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.position_list_query()
                if ret != 0 or data.empty:
                    continue
                for _, row in data.iterrows():
                    futu_sym = row.get("code", "")
                    sym = futu_sym.split(".")[-1]  # US.AAPL → AAPL
                    qty = int(row.get("qty", 0))
                    if qty == 0:
                        continue
                    avg_price = float(row.get("cost_price", 0))
                    last = self.last_prices.get(sym, avg_price)
                    mv = qty * last
                    upnl = (last - avg_price) * qty
                    results.append(Position(
                        symbol=sym,
                        quantity=qty,
                        avg_price=avg_price,
                        market_value=mv,
                        unrealized_pnl=upnl,
                    ))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Futu. Returns updated Order."""
        futu_sym = _to_futu_symbol(order.symbol)
        ctx = self._get_trade_ctx(order.symbol)
        if ctx is None:
            order.status = OrderStatus.REJECTED
            order.broker_data = {"error": "FutuOpenD not connected"}
            return order

        try:
            ret, data = ctx.place_order(
                price=order.price or 0.0,
                qty=order.quantity,
                code=futu_sym,
                trd_side=_to_futu_side(order.side),
                order_type=_to_futu_order_type(order.order_type),
                trd_env=TrdEnv.SIMULATE,
            )
            if ret == 0 and not data.empty:
                row = data.iloc[0]
                order.order_id = str(row.get("order_id", ""))
                order.status = _from_futu_status(row.get("order_status", FutuOrderStatus.SUBMITTED))
                order.filled_qty = int(row.get("dealt_qty", 0))
                order.avg_fill_price = float(row.get("dealt_avg_price", 0))
                order.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                order.broker_data = row.to_dict()
            else:
                order.status = OrderStatus.REJECTED
                order.broker_data = {"error": str(data) if data else "unknown"}
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.broker_data = {"error": str(e)}

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.cancel_order(
                    order_id=order_id,
                    trd_env=TrdEnv.SIMULATE,
                )
                if ret == 0:
                    return True
            except Exception:
                continue
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Query order status from Futu."""
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.order_list_query(
                    order_id=order_id,
                    trd_env=TrdEnv.SIMULATE,
                )
                if ret == 0 and not data.empty:
                    row = data.iloc[0]
                    futu_sym = row.get("code", "")
                    sym = futu_sym.split(".")[-1]
                    return Order(
                        symbol=sym,
                        side=_from_futu_side(row.get("trd_side", TrdSide.BUY)),
                        order_type=OrderType.MARKET,
                        quantity=int(row.get("qty", 0)),
                        order_id=order_id,
                        status=_from_futu_status(row.get("order_status", FutuOrderStatus.SUBMITTED)),
                        filled_qty=int(row.get("dealt_qty", 0)),
                        avg_fill_price=float(row.get("dealt_avg_price", 0)),
                        broker_data=row.to_dict(),
                    )
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def refresh_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Update last_prices from real-time quotes."""
        if self._quote_ctx is None:
            self.connect()

        futu_symbols = [_to_futu_symbol(s) for s in symbols]
        try:
            ret, data = self._quote_ctx.get_market_snapshot(futu_symbols)
            if ret == 0 and not data.empty:
                for _, row in data.iterrows():
                    futu_sym = row.get("code", "")
                    sym = futu_sym.split(".")[-1]
                    price = float(row.get("last_price", 0))
                    if price > 0:
                        self.last_prices[sym] = price
        except Exception:
            pass
        return self.last_prices
