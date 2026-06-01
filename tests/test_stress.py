"""Tests for analysis/stress.py — historical scenario replay."""

import math

import numpy as np
import pandas as pd
import pytest

from analysis.stress import (
    SCENARIOS,
    replay_custom,
    replay_scenario,
    run_scenarios,
)


def _prices_with_drop(symbols, start, end, drop_pct):
    """Build a price panel where the window has a linear drop of ``drop_pct``."""
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n < 2:
        raise ValueError("need at least 2 dates")
    end_factor = 1 - drop_pct / 100
    data = {}
    for s in symbols:
        # Linear path from 100 → 100*end_factor
        path = np.linspace(100, 100 * end_factor, n)
        data[s] = path
    return pd.DataFrame(data, index=dates)


# ===================================================================
# Scenario library sanity
# ===================================================================


class TestScenarioLibrary:
    def test_all_scenarios_have_required_fields(self):
        for key, cfg in SCENARIOS.items():
            for field in ("name", "start", "end", "description"):
                assert field in cfg, f"{key} missing {field}"
            # Dates parseable + start < end
            start = pd.Timestamp(cfg["start"])
            end = pd.Timestamp(cfg["end"])
            assert start < end, f"{key}: start >= end"

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            replay_scenario(pd.DataFrame(), {}, "no_such_scenario")


# ===================================================================
# Replay correctness
# ===================================================================


class TestReplayCorrectness:
    def test_linear_drop_recovered_exactly(self):
        """Synthetic 30% drop → portfolio return ≈ -30%, MaxDD ≈ -30%."""
        prices = _prices_with_drop(["AAPL"], "2020-02-19", "2020-03-23", 30.0)
        r = replay_scenario(prices, {"AAPL": 1.0}, "2020_covid")
        assert r["return_pct"] == pytest.approx(-30.0, abs=0.1)
        # Monotonic drop: MaxDD equals total drop
        assert r["max_dd_pct"] == pytest.approx(-30.0, abs=0.1)
        assert r["n_days"] > 0
        assert r["by_symbol"]["AAPL"] == pytest.approx(-30.0, abs=0.1)
        assert r["missing_symbols"] == []

    def test_two_symbols_equal_weight_daily_rebalanced(self):
        """Portfolio return is daily-rebalanced weighted avg, not B&H avg.

        Linear paths AAPL 100→70 and MSFT 100→110: daily rebalance keeps
        adding to the loser, so cumulative return is slightly worse than
        the simple ((-30 + 10) / 2 = -10%) buy-and-hold average — closer
        to -12% with these paths.  This is the industry-standard convention
        (Aladdin / Bloomberg use daily-rebalanced weights).
        """
        prices = _prices_with_drop(["AAPL", "MSFT"], "2020-02-19", "2020-03-23", 0)
        prices["AAPL"] = np.linspace(100, 70, len(prices))
        prices["MSFT"] = np.linspace(100, 110, len(prices))
        r = replay_scenario(prices, {"AAPL": 1, "MSFT": 1}, "2020_covid")
        # Daily-rebalanced portfolio return on these linear paths ≈ -12%
        assert -14.0 < r["return_pct"] < -10.0
        # Per-symbol total return is still pure buy-and-hold for context
        assert r["by_symbol"]["AAPL"] == pytest.approx(-30.0, abs=0.1)
        assert r["by_symbol"]["MSFT"] == pytest.approx(10.0, abs=0.1)

    def test_weights_normalised(self):
        prices = _prices_with_drop(["AAPL", "MSFT"], "2020-02-19", "2020-03-23", 0)
        prices["AAPL"] = np.linspace(100, 70, len(prices))
        prices["MSFT"] = np.linspace(100, 100, len(prices))
        r1 = replay_scenario(prices, {"AAPL": 1, "MSFT": 1}, "2020_covid")
        r2 = replay_scenario(prices, {"AAPL": 10, "MSFT": 10}, "2020_covid")
        assert r1["return_pct"] == pytest.approx(r2["return_pct"], abs=0.001)

    def test_max_dd_tracks_intra_window_drop(self):
        """V-shape: -30% then back to start.  Total return ≈ 0, MaxDD ≈ -30%."""
        dates = pd.bdate_range("2020-02-19", "2020-03-23")
        n = len(dates)
        half = n // 2
        path = np.concatenate([
            np.linspace(100, 70, half),
            np.linspace(70, 100, n - half),
        ])
        prices = pd.DataFrame({"AAPL": path}, index=dates)
        r = replay_scenario(prices, {"AAPL": 1}, "2020_covid")
        assert abs(r["return_pct"]) < 1.0       # back near 0
        assert r["max_dd_pct"] < -25.0           # dropped below -25 at trough


# ===================================================================
# Missing data handling
# ===================================================================


class TestMissingData:
    def test_empty_prices_returns_nans(self):
        r = replay_scenario(pd.DataFrame(), {"AAPL": 1}, "2020_covid")
        assert math.isnan(r["return_pct"])
        assert math.isnan(r["max_dd_pct"])
        assert r["missing_symbols"] == ["AAPL"]

    def test_no_data_in_window(self):
        """Symbol exists but data is outside scenario window."""
        prices = _prices_with_drop(["AAPL"], "2025-01-01", "2025-12-31", 10)
        r = replay_scenario(prices, {"AAPL": 1}, "2020_covid")
        assert "AAPL" in r["missing_symbols"]
        assert math.isnan(r["return_pct"])

    def test_partial_coverage_uses_available(self):
        """One symbol has data, another doesn't — uses what's available."""
        prices = _prices_with_drop(["AAPL"], "2020-02-19", "2020-03-23", 30)
        r = replay_scenario(prices, {"AAPL": 1, "TSLA": 1}, "2020_covid")
        # TSLA missing → portfolio = 100% AAPL
        assert r["return_pct"] == pytest.approx(-30.0, abs=0.5)
        assert "TSLA" in r["missing_symbols"]
        assert "AAPL" not in r["missing_symbols"]

    def test_zero_weight_skipped(self):
        """Symbol with weight=0 doesn't show up as missing or used."""
        prices = _prices_with_drop(["AAPL", "MSFT"], "2020-02-19", "2020-03-23", 10)
        r = replay_scenario(prices, {"AAPL": 1, "MSFT": 0}, "2020_covid")
        assert "MSFT" not in r["missing_symbols"]
        assert "MSFT" not in r["by_symbol"]


# ===================================================================
# Custom scenario + batch run
# ===================================================================


class TestCustomAndBatch:
    def test_replay_custom_window(self):
        prices = _prices_with_drop(["AAPL"], "2024-01-01", "2024-06-30", 15)
        r = replay_custom(prices, {"AAPL": 1}, "2024-01-01", "2024-06-30", name="测试窗口")
        assert r["name"] == "测试窗口"
        assert r["scenario"] == "custom"
        assert r["return_pct"] == pytest.approx(-15.0, abs=0.5)

    def test_run_scenarios_returns_all(self):
        prices = _prices_with_drop(
            ["AAPL"], "2008-01-01", "2024-12-31", 0,
        )
        # All symbols flat 100 throughout history — every scenario should
        # see 0% return.  No errors thrown.
        results = run_scenarios(prices, {"AAPL": 1})
        assert set(results.keys()) == set(SCENARIOS.keys())
        for sid, r in results.items():
            if not math.isnan(r["return_pct"]):
                assert abs(r["return_pct"]) < 0.5  # flat

    def test_run_scenarios_subset(self):
        prices = _prices_with_drop(["AAPL"], "2008-01-01", "2024-12-31", 0)
        results = run_scenarios(prices, {"AAPL": 1}, scenarios=["2020_covid", "2022_rate_hike"])
        assert set(results.keys()) == {"2020_covid", "2022_rate_hike"}
