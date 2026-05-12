"""Tests for RiskLimits in live_trader.py."""

import pytest
from live_trader import RiskLimits


class TestRiskLimitsDefaults:
    def test_default_values(self):
        r = RiskLimits()
        assert r.max_position_pct == 0.30
        assert r.max_total_exposure_pct == 0.80
        assert r.max_daily_loss_pct == 0.05
        assert r.min_order_value == 500.0
        assert r.max_slippage_pct == 0.02

    def test_override(self):
        r = RiskLimits(max_position_pct=0.25, min_order_value=1000)
        assert r.max_position_pct == 0.25
        assert r.min_order_value == 1000
        # Others stay default
        assert r.max_total_exposure_pct == 0.80

    def test_day_tracking_initialised(self):
        r = RiskLimits()
        assert r._day_start_equity == 0.0
        assert r._date == ""


class TestRiskLimitsFromConfig:
    def test_full_config(self):
        config = {
            "risk": {
                "max_position_pct": 0.25,
                "max_total_exposure_pct": 0.70,
                "max_daily_loss_pct": 0.03,
                "min_order_value": 1000.0,
                "max_slippage_pct": 0.01,
            }
        }
        r = RiskLimits.from_config(config)
        assert r.max_position_pct == 0.25
        assert r.max_total_exposure_pct == 0.70
        assert r.max_daily_loss_pct == 0.03
        assert r.min_order_value == 1000.0
        assert r.max_slippage_pct == 0.01

    def test_partial_config(self):
        config = {"risk": {"max_position_pct": 0.20}}
        r = RiskLimits.from_config(config)
        assert r.max_position_pct == 0.20
        assert r.max_total_exposure_pct == 0.80  # default

    def test_empty_config(self):
        r = RiskLimits.from_config({})
        assert r.max_position_pct == 0.30  # all defaults

    def test_no_risk_section(self):
        config = {"watchlist": []}
        r = RiskLimits.from_config(config)
        assert r.max_position_pct == 0.30  # all defaults


class TestRiskLimitsBoundary:
    def test_zero_values(self):
        r = RiskLimits(max_position_pct=0, min_order_value=0)
        assert r.max_position_pct == 0

    def test_high_values(self):
        r = RiskLimits(max_position_pct=1.0, max_total_exposure_pct=2.0)
        assert r.max_position_pct == 1.0
        assert r.max_total_exposure_pct == 2.0
