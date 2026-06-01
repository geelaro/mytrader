"""Tests for live/risk_alerts.py — RiskAlerter state machine."""

from unittest.mock import MagicMock

import pytest

from analysis.risk_monitor import RiskLevel, RiskState
from live.risk_alerts import (
    AlertConfig,
    RiskAlerter,
    _KEY_POS_ALERTED_FMT,
    _KEY_RISK_LIGHT,
    _KEY_VIX_ALERTED,
)


def _make_state(level: RiskLevel, reasons=None, indicators=None) -> RiskState:
    return RiskState(
        level=level,
        reasons=reasons or [],
        indicators=indicators or {},
    )


def _make_notifier(available: bool = True) -> MagicMock:
    nf = MagicMock()
    nf.available = available
    nf.risk_alert_card = MagicMock(return_value=True)
    nf.vix_alert_card = MagicMock(return_value=True)
    nf.position_alert_card = MagicMock(return_value=True)
    return nf


def _pos(symbol: str, current: float, stop: float, **extra) -> dict:
    return {"symbol": symbol, "current_price": current, "stop_price": stop, **extra}


# ===================================================================
# AlertConfig
# ===================================================================


class TestAlertConfig:
    def test_defaults(self):
        cfg = AlertConfig()
        assert cfg.enabled is True
        assert cfg.risk_light_enabled is True
        assert cfg.position_distance_alert_pct == 5.0
        assert cfg.position_distance_clear_pct == 7.0

    def test_from_dict_filters_unknown_keys(self):
        cfg = AlertConfig.from_dict({
            "enabled": False,
            "risk_light_enabled": False,
            "bogus_field": "ignored",
        })
        assert cfg.enabled is False
        assert cfg.risk_light_enabled is False
        # Defaults preserved for unspecified
        assert cfg.vix_alert_threshold == 30.0

    def test_from_dict_none_returns_defaults(self):
        assert AlertConfig.from_dict(None) == AlertConfig()
        assert AlertConfig.from_dict({}) == AlertConfig()


# ===================================================================
# Risk light state machine
# ===================================================================


class TestRiskLightTransitions:
    def test_first_red_fires(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        fired = alerter.check_risk_light(_make_state(RiskLevel.RED, ["test"]))
        assert fired is True
        notifier.risk_alert_card.assert_called_once()

    def test_consecutive_red_no_repeat(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        state = _make_state(RiskLevel.RED)
        assert alerter.check_risk_light(state) is True
        assert alerter.check_risk_light(state) is False
        assert alerter.check_risk_light(state) is False
        assert notifier.risk_alert_card.call_count == 1

    def test_red_to_green_to_red_rearms(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_risk_light(_make_state(RiskLevel.RED)) is True
        assert alerter.check_risk_light(_make_state(RiskLevel.GREEN)) is False
        # Re-entering RED fires again
        assert alerter.check_risk_light(_make_state(RiskLevel.RED)) is True
        assert notifier.risk_alert_card.call_count == 2

    def test_yellow_to_red_fires(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_risk_light(_make_state(RiskLevel.YELLOW))
        assert alerter.check_risk_light(_make_state(RiskLevel.RED)) is True
        assert notifier.risk_alert_card.call_count == 1

    def test_green_yellow_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        for level in (RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.GREEN, RiskLevel.YELLOW):
            assert alerter.check_risk_light(_make_state(level)) is False
        notifier.risk_alert_card.assert_not_called()

    def test_level_persisted_each_call(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_risk_light(_make_state(RiskLevel.YELLOW))
        assert temp_cache.load_risk_state(_KEY_RISK_LIGHT) == "yellow"
        alerter.check_risk_light(_make_state(RiskLevel.RED))
        assert temp_cache.load_risk_state(_KEY_RISK_LIGHT) == "red"


class TestRiskLightDisabled:
    def test_globally_disabled_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache, AlertConfig(enabled=False))
        assert alerter.check_risk_light(_make_state(RiskLevel.RED)) is False
        notifier.risk_alert_card.assert_not_called()

    def test_risk_light_only_disabled_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(
            notifier, temp_cache,
            AlertConfig(enabled=True, risk_light_enabled=False),
        )
        assert alerter.check_risk_light(_make_state(RiskLevel.RED)) is False
        notifier.risk_alert_card.assert_not_called()


class TestVixSpike:
    def test_first_cross_above_30_fires(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(25.0) is False
        assert alerter.check_vix(31.5) is True
        notifier.vix_alert_card.assert_called_once_with(31.5, 30.0)

    def test_at_exact_threshold_fires(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(30.0) is True

    def test_consecutive_high_vix_no_repeat(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(35.0) is True
        assert alerter.check_vix(38.0) is False
        assert alerter.check_vix(33.0) is False
        assert notifier.vix_alert_card.call_count == 1

    def test_hysteresis_no_rearm_until_clear_threshold(self, temp_cache):
        """VIX 30→28 (between clear and alert thresholds) should NOT re-arm."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(32.0) is True
        # Drops to 28 — above clear (27) — does NOT re-arm
        assert alerter.check_vix(28.0) is False
        # Goes back to 35 — still no fire because never cleared
        assert alerter.check_vix(35.0) is False
        assert notifier.vix_alert_card.call_count == 1

    def test_rearm_after_clear(self, temp_cache):
        """VIX 32 → 26 (below clear) → 35 should fire twice."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(32.0) is True
        assert alerter.check_vix(26.0) is False  # cleared
        assert temp_cache.load_risk_state(_KEY_VIX_ALERTED) == "0"
        assert alerter.check_vix(35.0) is True   # re-armed → fires
        assert notifier.vix_alert_card.call_count == 2

    def test_invalid_vix_value_ignored(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_vix(None) is False
        assert alerter.check_vix(0) is False
        assert alerter.check_vix(-5) is False
        notifier.vix_alert_card.assert_not_called()


class TestVixDisabled:
    def test_globally_disabled_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache, AlertConfig(enabled=False))
        assert alerter.check_vix(40.0) is False
        notifier.vix_alert_card.assert_not_called()

    def test_vix_only_disabled_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(
            notifier, temp_cache,
            AlertConfig(enabled=True, vix_spike_enabled=False),
        )
        assert alerter.check_vix(40.0) is False
        notifier.vix_alert_card.assert_not_called()


class TestPositionAlert:
    def test_within_5pct_fires(self, temp_cache):
        """price=100, stop=96 → distance 4% < 5% → fire."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        n = alerter.check_positions([_pos("AAPL", 100, 96, shares=10)])
        assert n == 1
        notifier.position_alert_card.assert_called_once()

    def test_above_5pct_no_fire(self, temp_cache):
        """price=100, stop=90 → distance 10% → no fire."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        n = alerter.check_positions([_pos("AAPL", 100, 90)])
        assert n == 0
        notifier.position_alert_card.assert_not_called()

    def test_consecutive_close_no_repeat(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        pos = _pos("AAPL", 100, 97)  # 3% distance
        assert alerter.check_positions([pos]) == 1
        assert alerter.check_positions([pos]) == 0
        assert alerter.check_positions([pos]) == 0
        assert notifier.position_alert_card.call_count == 1

    def test_per_symbol_independent_state(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        # AAPL close, MSFT far → fire only AAPL
        positions = [_pos("AAPL", 100, 97), _pos("MSFT", 200, 180)]
        assert alerter.check_positions(positions) == 1
        # Next call MSFT also gets close → fire MSFT this time
        positions = [_pos("AAPL", 100, 97), _pos("MSFT", 200, 195)]
        assert alerter.check_positions(positions) == 1
        assert notifier.position_alert_card.call_count == 2

    def test_hysteresis_no_rearm_in_band(self, temp_cache):
        """distance 4% (fire) → 6% (above alert, below clear) → 3% (still alerted)."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_positions([_pos("AAPL", 100, 96)]) == 1
        # Recovers to 6% — above 5% alert but below 7% clear → still latched
        assert alerter.check_positions([_pos("AAPL", 100, 94)]) == 0
        # Drops back close — should NOT re-fire (never cleared)
        assert alerter.check_positions([_pos("AAPL", 100, 97)]) == 0
        assert notifier.position_alert_card.call_count == 1

    def test_rearm_after_clear(self, temp_cache):
        """distance 4% (fire) → 8% (cleared) → 3% (re-fires)."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_positions([_pos("AAPL", 100, 96)]) == 1
        # Recovers to 8% — above clear → re-arm
        assert alerter.check_positions([_pos("AAPL", 100, 92)]) == 0
        assert temp_cache.load_risk_state(_KEY_POS_ALERTED_FMT.format(symbol="AAPL")) == "0"
        # Drops back close → fires again
        assert alerter.check_positions([_pos("AAPL", 100, 97)]) == 1
        assert notifier.position_alert_card.call_count == 2

    def test_empty_positions_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        assert alerter.check_positions([]) == 0
        assert alerter.check_positions(None) == 0

    def test_invalid_positions_skipped(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        bad = [
            _pos("AAPL", 0, 95),         # current=0
            _pos("MSFT", 100, 0),        # stop=0
            _pos("", 100, 97),           # no symbol
            _pos("GOOG", -10, 5),        # negative price
        ]
        assert alerter.check_positions(bad) == 0
        notifier.position_alert_card.assert_not_called()

    def test_disabled_no_fire(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(
            notifier, temp_cache,
            AlertConfig(enabled=True, position_alert_enabled=False),
        )
        assert alerter.check_positions([_pos("AAPL", 100, 97)]) == 0
        notifier.position_alert_card.assert_not_called()


class TestTickCombined:
    def test_tick_fires_all_three(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        summary = alerter.tick(
            risk_state=_make_state(RiskLevel.RED),
            vix_value=35.0,
            positions=[_pos("AAPL", 100, 97)],
        )
        assert summary == {"risk_light": True, "vix": True, "positions": 1}
        notifier.risk_alert_card.assert_called_once()
        notifier.vix_alert_card.assert_called_once()
        notifier.position_alert_card.assert_called_once()

    def test_tick_skips_none_inputs(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        summary = alerter.tick(risk_state=None, vix_value=None, positions=None)
        assert summary == {"risk_light": False, "vix": False, "positions": 0}
        notifier.risk_alert_card.assert_not_called()
        notifier.vix_alert_card.assert_not_called()
        notifier.position_alert_card.assert_not_called()

    def test_tick_partial(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        summary = alerter.tick(vix_value=35.0)
        assert summary == {"risk_light": False, "vix": True, "positions": 0}
        notifier.risk_alert_card.assert_not_called()
        notifier.vix_alert_card.assert_called_once()


class TestAlertHistoryRecording:
    """Stage B: verify every fired alert is persisted to alert_history."""

    def test_red_alert_recorded(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        state = _make_state(
            RiskLevel.RED,
            reasons=["VIX > 30"],
            indicators={"vix": 35.0, "spy_close": 400.0},
        )
        alerter.check_risk_light(state)
        rows = temp_cache.load_alert_history(days=1, alert_type="risk_light")
        assert len(rows) == 1
        assert rows[0]["payload"]["level"] == "red"
        assert rows[0]["payload"]["reasons"] == ["VIX > 30"]
        assert rows[0]["payload"]["indicators"]["vix"] == 35.0

    def test_vix_alert_recorded(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_vix(33.5)
        rows = temp_cache.load_alert_history(days=1, alert_type="vix_spike")
        assert len(rows) == 1
        assert rows[0]["payload"]["value"] == 33.5
        assert rows[0]["payload"]["threshold"] == 30.0

    def test_position_alert_recorded(self, temp_cache):
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_positions([_pos(
            "AAPL", current=100, stop=97, shares=10, strategy="weekly_macd_kdj",
        )])
        rows = temp_cache.load_alert_history(days=1, alert_type="position_stop")
        assert len(rows) == 1
        p = rows[0]["payload"]
        assert p["symbol"] == "AAPL"
        assert p["distance_pct"] == pytest.approx(3.0, abs=0.01)
        assert p["strategy"] == "weekly_macd_kdj"
        assert p["shares"] == 10

    def test_consecutive_red_records_only_once(self, temp_cache):
        """State machine de-dupes — history shouldn't get duplicate rows."""
        notifier = _make_notifier()
        alerter = RiskAlerter(notifier, temp_cache)
        state = _make_state(RiskLevel.RED)
        alerter.check_risk_light(state)
        alerter.check_risk_light(state)
        alerter.check_risk_light(state)
        rows = temp_cache.load_alert_history(days=1)
        assert len(rows) == 1

    def test_history_survives_notifier_failure(self, temp_cache):
        """Notifier crash must not lose the audit trail."""
        notifier = _make_notifier()
        notifier.risk_alert_card.side_effect = RuntimeError("webhook 500")
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_risk_light(_make_state(RiskLevel.RED))
        rows = temp_cache.load_alert_history(days=1)
        assert len(rows) == 1
        assert rows[0]["alert_type"] == "risk_light"

    def test_history_survives_notifier_unavailable(self, temp_cache):
        notifier = _make_notifier(available=False)
        alerter = RiskAlerter(notifier, temp_cache)
        alerter.check_risk_light(_make_state(RiskLevel.RED))
        alerter.check_vix(35.0)
        alerter.check_positions([_pos("AAPL", 100, 97)])
        rows = temp_cache.load_alert_history(days=1)
        # All three recorded even though notifier was unavailable
        types = sorted(r["alert_type"] for r in rows)
        assert types == ["position_stop", "risk_light", "vix_spike"]


class TestRiskLightRobustness:
    def test_notifier_unavailable_does_not_crash(self, temp_cache):
        notifier = _make_notifier(available=False)
        alerter = RiskAlerter(notifier, temp_cache)
        # Still returns True (transition detected) and persists state,
        # only the actual notification is dropped.
        fired = alerter.check_risk_light(_make_state(RiskLevel.RED))
        assert fired is True
        notifier.risk_alert_card.assert_not_called()

    def test_notifier_raises_does_not_crash(self, temp_cache):
        notifier = _make_notifier()
        notifier.risk_alert_card.side_effect = RuntimeError("network down")
        alerter = RiskAlerter(notifier, temp_cache)
        # Exception swallowed and logged, no propagation
        fired = alerter.check_risk_light(_make_state(RiskLevel.RED))
        assert fired is True
