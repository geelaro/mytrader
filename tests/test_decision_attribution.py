"""Tests for analysis.decision_attribution."""

from __future__ import annotations

import pytest

from analysis.decision_attribution import (
    decision_attribution_summary,
    hit_rate_by_group,
    join_decision_pnl,
)


def _dec(ts, dtype, sym, risk_light=None, order_id=None):
    return {
        "ts": ts,
        "decision_type": dtype,
        "symbol": sym,
        "risk_light": risk_light,
        "payload": {"order_id": order_id} if order_id else {},
    }


def _pnl(symbol, pnl, exit_date, order_id=None, pnl_pct=None):
    return {
        "symbol": symbol,
        "pnl": pnl,
        "pnl_pct": pnl_pct if pnl_pct is not None else pnl / 100.0,
        "exit_date": exit_date,
        "order_id": order_id,
    }


class TestJoin:
    def test_match_by_order_id(self):
        """Close-side decisions match exactly via payload.order_id."""
        decisions = [_dec("2026-01-01", "trade_sell", "NVDA", order_id="X1")]
        pnls = [_pnl("NVDA", 500, "2026-01-02", order_id="X1")]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] == 500
        assert out[0]["matched_by"] == "order_id"

    def test_match_buy_to_next_close(self):
        """trade_buy matches the next-after close on the same symbol."""
        decisions = [_dec("2026-01-01", "trade_buy", "NVDA")]
        pnls = [_pnl("NVDA", 300, "2026-01-15", order_id="X9")]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] == 300
        assert out[0]["matched_by"] == "symbol"

    def test_buy_does_not_match_earlier_exit(self):
        """A buy decision on Jan 5 doesn't match a Jan 1 exit (from prior trade)."""
        decisions = [_dec("2026-01-05", "trade_buy", "NVDA")]
        pnls = [_pnl("NVDA", 999, "2026-01-01", order_id="OLD")]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] is None
        assert out[0]["matched_by"] is None

    def test_buy_matches_earliest_subsequent_exit(self):
        """Multiple subsequent exits → match the earliest one."""
        decisions = [_dec("2026-01-01", "trade_buy", "NVDA")]
        pnls = [
            _pnl("NVDA", 100, "2026-01-10", order_id="A"),
            _pnl("NVDA", 999, "2026-02-15", order_id="B"),
        ]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] == 100   # earliest after entry

    def test_order_id_beats_symbol_match(self):
        """When both modes could match, order_id wins."""
        decisions = [_dec("2026-01-01", "trade_sell", "NVDA", order_id="EXACT")]
        pnls = [
            _pnl("NVDA", 700, "2026-01-02", order_id="EXACT"),
            _pnl("NVDA", 999, "2026-01-03", order_id="OTHER"),
        ]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] == 700
        assert out[0]["matched_by"] == "order_id"

    def test_unmatched_decision_has_none_pnl(self):
        decisions = [_dec("2026-01-01", "trade_buy", "META")]
        pnls = [_pnl("NVDA", 100, "2026-02-01", order_id="X")]  # different sym
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] is None

    def test_kill_switch_does_not_force_match(self):
        """kill_switch is symbol=None — no symbol-based match attempted."""
        decisions = [{
            "ts": "2026-01-01", "decision_type": "kill_switch",
            "symbol": None, "risk_light": "red", "payload": {},
        }]
        pnls = [_pnl("NVDA", 999, "2026-02-01", order_id="X")]
        out = join_decision_pnl(decisions, pnls)
        assert out[0]["pnl"] is None

    def test_empty_inputs(self):
        assert join_decision_pnl([], []) == []
        assert join_decision_pnl(None, None) == []


class TestHitRate:
    def test_groups_by_risk_light(self):
        joined = [
            {"risk_light": "green", "pnl": 200},
            {"risk_light": "green", "pnl": -50},
            {"risk_light": "red",   "pnl": -300},
            {"risk_light": "red",   "pnl": -100},
            {"risk_light": "red",   "pnl": 50},
        ]
        out = hit_rate_by_group(joined, group_by="risk_light")
        green = out["groups"]["green"]
        red = out["groups"]["red"]
        assert green["n"] == 2
        assert green["n_wins"] == 1
        assert green["win_rate"] == 0.5
        assert green["total_pnl"] == 150
        assert red["n"] == 3
        assert red["n_wins"] == 1
        assert red["win_rate"] == pytest.approx(1 / 3)
        assert red["total_pnl"] == -350

    def test_unmatched_excluded_by_default(self):
        joined = [
            {"risk_light": "green", "pnl": 100},
            {"risk_light": "green", "pnl": None},  # unmatched
        ]
        out = hit_rate_by_group(joined, group_by="risk_light")
        assert out["groups"]["green"]["n"] == 1
        assert out["n_matched"] == 1
        assert out["n_unmatched"] == 1

    def test_none_bucket(self):
        joined = [
            {"risk_light": None, "pnl": 100},
            {"risk_light": "",   "pnl": -50},
        ]
        out = hit_rate_by_group(joined, group_by="risk_light")
        assert "(unknown)" in out["groups"]
        assert out["groups"]["(unknown)"]["n"] == 2

    def test_median_correct(self):
        joined = [{"risk_light": "g", "pnl": p} for p in [-100, 50, 200]]
        out = hit_rate_by_group(joined, group_by="risk_light")
        assert out["groups"]["g"]["median_pnl"] == 50

    def test_group_by_decision_type(self):
        joined = [
            {"decision_type": "trade_buy",  "pnl": 100},
            {"decision_type": "trade_buy",  "pnl": 200},
            {"decision_type": "trade_sell", "pnl": -50},
        ]
        out = hit_rate_by_group(joined, group_by="decision_type")
        assert out["groups"]["trade_buy"]["n"] == 2
        assert out["groups"]["trade_sell"]["n"] == 1


class TestSummary:
    def test_end_to_end(self):
        decisions = [
            _dec("2026-01-01", "trade_buy", "NVDA", risk_light="green"),
            _dec("2026-01-05", "trade_buy", "TSLA", risk_light="red"),
        ]
        pnls = [
            _pnl("NVDA", 500, "2026-01-10"),
            _pnl("TSLA", -300, "2026-01-12"),
        ]
        out = decision_attribution_summary(decisions, pnls, group_by="risk_light")
        assert out["group_by"] == "risk_light"
        assert out["n_matched"] == 2
        assert out["groups"]["green"]["win_rate"] == 1.0
        assert out["groups"]["red"]["win_rate"] == 0.0
