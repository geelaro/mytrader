"""Risk alerter — push Feishu notifications on threshold crossings.

Stateful alerter: only fires on transitions, not every tick.  State persisted
to :class:`StateStore` so daemon restarts do not re-fire stale alerts.

Three alert types
-----------------
1. Risk light → RED (from analysis.risk_monitor.RiskState)
   Fires on transition into RED.  Recovery to YELLOW/GREEN re-arms the next
   RED.  Persistent RED ticks do NOT re-fire.

2. VIX spike (Stage 2 — not yet implemented)
3. Position approaching stop (Stage 3 — not yet implemented)

Design
------
``RiskAlerter.check_risk_light(state)`` is idempotent given the same input
state: calling twice in a row with the same RED state fires once.  This makes
the daemon-tick integration trivial — just call on every tick.

Usage
-----
    from live.risk_alerts import RiskAlerter, AlertConfig
    alerter = RiskAlerter(notifier, state_store, AlertConfig.from_dict(cfg))
    alerter.check_risk_light(risk_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from analysis.risk_monitor import RiskState
    from data.cache import StateStore
    from utils.notify import Notifier

logger = logging.getLogger(__name__)


def _incr_alert(alert_type: str) -> None:
    """Best-effort metrics increment — never raises."""
    try:
        from utils import metrics_server
        metrics_server.incr("risk_alert_fired_total", {"type": alert_type})
    except Exception:
        pass


# State-store key prefix; keep separate from existing risk_state keys.
_KEY_RISK_LIGHT = "alert:last_risk_level"
_KEY_VIX_ALERTED = "alert:vix_alerted"
_KEY_POS_ALERTED_FMT = "alert:pos_alerted:{symbol}"


@dataclass
class AlertConfig:
    """Alert toggles and thresholds, loaded from ``config.yaml [alerts]``."""

    enabled: bool = True
    risk_light_enabled: bool = True
    vix_spike_enabled: bool = True
    vix_alert_threshold: float = 30.0
    vix_clear_threshold: float = 27.0
    position_alert_enabled: bool = True
    position_distance_alert_pct: float = 5.0
    position_distance_clear_pct: float = 7.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AlertConfig":
        """Build from a config-section dict.  Unknown keys are ignored."""
        if not data:
            return cls()
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


class RiskAlerter:
    """Stateful alerter; only fires on transitions."""

    def __init__(
        self,
        notifier: "Notifier",
        state_store: "StateStore",
        config: Optional[AlertConfig] = None,
    ):
        self.notifier = notifier
        self.store = state_store
        self.config = config or AlertConfig()

    # ------------------------------------------------------------------
    # Risk light
    # ------------------------------------------------------------------

    def check_risk_light(self, state: "RiskState") -> bool:
        """Fire if risk-light level transitioned into RED.

        Returns True iff an alert was sent.  Always persists the new level
        so subsequent calls see the transition correctly.
        """
        if not (self.config.enabled and self.config.risk_light_enabled):
            return False

        prev = self.store.load_risk_state(_KEY_RISK_LIGHT)
        current = state.level.value
        # Persist current level for the *next* tick to compare against.
        self.store.save_risk_state(_KEY_RISK_LIGHT, current)

        if current == "red" and prev != "red":
            self._notify_risk_light(state)
            _incr_alert("risk_light")
            return True
        return False

    # ------------------------------------------------------------------
    # VIX spike
    # ------------------------------------------------------------------

    def check_vix(self, vix_value: float) -> bool:
        """Fire if VIX crossed ``vix_alert_threshold`` from below.

        Uses hysteresis: once alerted, only re-arms when VIX falls below
        ``vix_clear_threshold`` to avoid flicker around the boundary.
        Returns True iff a new alert was sent this call.
        """
        if not (self.config.enabled and self.config.vix_spike_enabled):
            return False
        if vix_value is None or vix_value <= 0:
            return False

        was_alerted = self.store.load_risk_state(_KEY_VIX_ALERTED) == "1"

        if not was_alerted and vix_value >= self.config.vix_alert_threshold:
            self.store.save_risk_state(_KEY_VIX_ALERTED, "1")
            self._notify_vix(vix_value)
            _incr_alert("vix_spike")
            return True
        if was_alerted and vix_value < self.config.vix_clear_threshold:
            self.store.save_risk_state(_KEY_VIX_ALERTED, "0")
        return False

    # ------------------------------------------------------------------
    # Positions approaching stop
    # ------------------------------------------------------------------

    def check_positions(self, positions: list) -> int:
        """For each position, fire if distance-to-stop crossed below alert pct.

        Each ``positions`` item is a dict with at least::

            {"symbol": str, "current_price": float, "stop_price": float}

        Optional keys like ``shares`` / ``strategy`` are passed through to
        the notification card if present.

        Per-symbol hysteresis: once alerted, re-arms only when distance
        recovers above ``position_distance_clear_pct``.  Returns the count
        of fresh alerts fired this call.
        """
        if not (self.config.enabled and self.config.position_alert_enabled):
            return 0
        if not positions:
            return 0

        fired = 0
        for pos in positions:
            symbol = pos.get("symbol")
            current = pos.get("current_price", 0)
            stop = pos.get("stop_price", 0)
            if not symbol or current <= 0 or stop <= 0:
                continue

            distance_pct = (current - stop) / current * 100
            key = _KEY_POS_ALERTED_FMT.format(symbol=symbol)
            was_alerted = self.store.load_risk_state(key) == "1"

            if not was_alerted and distance_pct < self.config.position_distance_alert_pct:
                self.store.save_risk_state(key, "1")
                self._notify_position(pos, distance_pct)
                _incr_alert("position_stop")
                fired += 1
            elif was_alerted and distance_pct > self.config.position_distance_clear_pct:
                self.store.save_risk_state(key, "0")
        return fired

    # ------------------------------------------------------------------
    # Combined daemon hook
    # ------------------------------------------------------------------

    def tick(
        self,
        risk_state: Optional["RiskState"] = None,
        vix_value: Optional[float] = None,
        positions: Optional[list] = None,
    ) -> dict:
        """Run all enabled checks in one call — intended for daemon ticks.

        Each input is optional: pass ``None`` (or empty list) to skip that
        check.  Returns a dict summarising what fired::

            {"risk_light": bool, "vix": bool, "positions": int}
        """
        summary = {"risk_light": False, "vix": False, "positions": 0}
        if risk_state is not None:
            summary["risk_light"] = self.check_risk_light(risk_state)
        if vix_value is not None:
            summary["vix"] = self.check_vix(vix_value)
        if positions:
            summary["positions"] = self.check_positions(positions)
        return summary

    # ------------------------------------------------------------------
    # Notification renderers
    # ------------------------------------------------------------------

    def _record_history(self, alert_type: str, payload: dict) -> None:
        """Best-effort write to alert_history; never crashes the alerter."""
        try:
            self.store.record_alert(alert_type, payload)
        except Exception:
            logger.exception("RiskAlerter: failed to record %s history", alert_type)

    def _notify_risk_light(self, state: "RiskState"):
        """Record + push a RED risk-light alert.

        History is recorded BEFORE notifier dispatch so it survives notifier
        failures (offline, webhook down, etc.) — the audit trail must be
        independent of message delivery.
        """
        self._record_history("risk_light", {
            "level": state.level.value,
            "reasons": list(state.reasons),
            "indicators": dict(state.indicators or {}),
        })
        if not self.notifier.available:
            logger.warning("RiskAlerter: notifier unavailable, RED alert dropped")
            return
        try:
            self.notifier.risk_alert_card(state)
        except Exception:
            logger.exception("RiskAlerter: failed to send RED alert")

    def _notify_vix(self, value: float):
        """Record + push a VIX spike alert."""
        self._record_history("vix_spike", {
            "value": float(value),
            "threshold": float(self.config.vix_alert_threshold),
        })
        if not self.notifier.available:
            logger.warning("RiskAlerter: notifier unavailable, VIX alert dropped")
            return
        try:
            self.notifier.vix_alert_card(value, self.config.vix_alert_threshold)
        except Exception:
            logger.exception("RiskAlerter: failed to send VIX alert")

    def _notify_position(self, position: dict, distance_pct: float):
        """Record + push a position-approaching-stop alert."""
        self._record_history("position_stop", {
            "symbol": position.get("symbol"),
            "current_price": position.get("current_price"),
            "stop_price": position.get("stop_price"),
            "distance_pct": float(distance_pct),
            "strategy": position.get("strategy"),
            "shares": position.get("shares"),
        })
        if not self.notifier.available:
            logger.warning(
                "RiskAlerter: notifier unavailable, position alert dropped for %s",
                position.get("symbol"),
            )
            return
        try:
            self.notifier.position_alert_card(position, distance_pct)
        except Exception:
            logger.exception(
                "RiskAlerter: failed to send position alert for %s",
                position.get("symbol"),
            )
