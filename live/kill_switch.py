"""Emergency portfolio liquidation — manual one-button kill switch.

Manual-only by design.  Empirical study (CBOE VIX history 1990+, see
project journal 2026-06-02) shows VIX > 50 historically marks
*bottoms*, not continuation lower — auto-liquidating at VIX > 50 would
have locked in losses right before SPY rallied ~45% over the next year.
So no automatic threshold triggers; the user clicks the button when
they decide.

On trigger
----------
1. Snapshot all current positions via broker.get_positions().
2. For each non-zero position, submit an opposing MARKET order
   (SELL for longs, BUY for shorts).
3. Set ``risk_ctrl.trading_paused = True`` so the daemon doesn't open
   new positions until the user explicitly resets.
4. Persist active flag + reason + timestamp via StateStore.
5. Append to alert_history (audit trail).
6. Push Feishu RED card via notifier.

Reset
-----
``reset()`` clears the active flag and unpauses trading.  Idempotent —
re-triggering while active is a no-op.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from broker import Broker
    from live.risk_controller import RiskController
    from data.cache import CacheManager
    from utils.notify import Notifier

logger = logging.getLogger(__name__)


_KEY_ACTIVE = "kill_switch:active"
_KEY_REASON = "kill_switch:reason"
_KEY_TRIGGERED_AT = "kill_switch:triggered_at"


class KillSwitch:
    """Manual emergency liquidation.

    Construction
    ------------
    All four dependencies are injected so the class is testable without
    spinning up a real LiveTrader.
    """

    def __init__(
        self,
        broker: "Broker",
        risk_ctrl: "RiskController",
        notifier: "Notifier",
        cache: "CacheManager",
    ):
        self.broker = broker
        self.risk_ctrl = risk_ctrl
        self.notifier = notifier
        self.cache = cache

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.cache.load_risk_state(_KEY_ACTIVE) == "1"

    def get_state(self) -> dict:
        """Snapshot of current kill switch state."""
        return {
            "active": self.is_active,
            "reason": self.cache.load_risk_state(_KEY_REASON) or "",
            "triggered_at": self.cache.load_risk_state(_KEY_TRIGGERED_AT) or "",
        }

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def trigger(self, reason: str, dry_run: bool = False) -> dict:
        """Liquidate all open positions immediately.

        Parameters
        ----------
        reason : str
            Free-text reason (audit + notification body).  Required.
        dry_run : bool
            If True, records the intended orders but doesn't actually
            submit them.  Used by dashboard preview / tests.

        Returns
        -------
        dict::

            {
                "status": "triggered" | "already_active" | "no_positions",
                "reason": str,
                "n_positions": int,
                "orders": list[dict],   # one per attempted sell
                "errors": list[dict],   # per-symbol errors
                "dry_run": bool,
            }

        Idempotent: re-triggering while already active is a no-op and
        returns ``status="already_active"`` (no fresh orders submitted).
        """
        if not reason or not reason.strip():
            raise ValueError("Kill Switch requires a non-empty reason")

        if self.is_active:
            logger.warning("KillSwitch already active, skipping re-trigger")
            return {
                "status": "already_active",
                "reason": reason,
                "n_positions": 0,
                "orders": [],
                "errors": [],
                "dry_run": dry_run,
            }

        # Import here to avoid circular import at module load
        from broker import Order, OrderSide, OrderType

        positions = list(self.broker.get_positions() or [])
        orders: list = []
        errors: list = []

        for pos in positions:
            qty = int(getattr(pos, "quantity", 0) or 0)
            if qty == 0:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            sell_qty = abs(qty)
            try:
                order = Order(
                    symbol=pos.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=sell_qty,
                )
                if dry_run:
                    orders.append({
                        "symbol": pos.symbol, "side": side.value,
                        "qty": sell_qty, "status": "DRY_RUN",
                        "order_id": "",
                    })
                else:
                    result = self.broker.submit_order(order)
                    orders.append({
                        "symbol": pos.symbol,
                        "side": result.side.value,
                        "qty": result.quantity,
                        "status": result.status.value,
                        "order_id": result.order_id,
                    })
            except Exception as exc:
                logger.exception("KillSwitch: failed to liquidate %s", pos.symbol)
                errors.append({"symbol": pos.symbol, "error": str(exc)})

        ts = datetime.now().isoformat(timespec="seconds")

        # Pause trading so the daemon doesn't open new positions
        try:
            self.risk_ctrl.trading_paused = True
            self.risk_ctrl.pause_reason = f"Kill Switch: {reason}"
            if hasattr(self.risk_ctrl, "persist_state"):
                self.risk_ctrl.persist_state()
        except Exception:
            logger.exception("KillSwitch: failed to pause risk_ctrl")

        # Persist active flag (idempotency depends on this)
        self.cache.save_risk_state(_KEY_ACTIVE, "1")
        self.cache.save_risk_state(_KEY_REASON, reason)
        self.cache.save_risk_state(_KEY_TRIGGERED_AT, ts)

        # Audit trail
        payload = {
            "reason": reason,
            "n_positions": len(positions),
            "orders": orders,
            "errors": errors,
            "dry_run": dry_run,
            "triggered_at": ts,
        }
        try:
            self.cache.record_alert("kill_switch", payload)
        except Exception:
            logger.exception("KillSwitch: audit log write failed")

        # Push Feishu card (best-effort)
        self._notify(reason, len(positions), orders, errors, dry_run)

        status = "no_positions" if not positions else "triggered"
        return {"status": status, **payload}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, reason: str = "Manual reset by user") -> None:
        """Clear the active flag and unpause trading.

        After triggering, the daemon stays paused until this is called.
        The user is expected to manually review positions, confirm safe
        state, then reset.
        """
        if not self.is_active:
            return
        self.cache.save_risk_state(_KEY_ACTIVE, "0")
        try:
            self.risk_ctrl.trading_paused = False
            self.risk_ctrl.pause_reason = ""
            if hasattr(self.risk_ctrl, "persist_state"):
                self.risk_ctrl.persist_state()
        except Exception:
            logger.exception("KillSwitch: failed to unpause risk_ctrl")
        try:
            self.cache.record_alert("kill_switch_reset", {
                "reason": reason,
                "reset_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception:
            logger.exception("KillSwitch: reset audit log failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _notify(self, reason: str, n_positions: int, orders: list,
                errors: list, dry_run: bool) -> None:
        """Push a RED Feishu card describing the trigger."""
        if not getattr(self.notifier, "available", False):
            logger.warning("KillSwitch: notifier unavailable, alert dropped")
            return
        try:
            self.notifier.kill_switch_card(reason, n_positions, orders,
                                           errors, dry_run)
        except Exception:
            logger.exception("KillSwitch: failed to send Feishu card")
