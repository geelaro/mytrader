"""Tests for analysis/brinson.py — BHB attribution math + data helpers."""

import numpy as np
import pandas as pd
import pytest

from analysis.brinson import (
    SECTOR_ETF,
    brinson_attribution,
    compute_period_returns,
    portfolio_sector_breakdown,
)


# ===================================================================
# brinson_attribution — math correctness
# ===================================================================


class TestBrinsonMath:
    def test_zero_active_when_identical(self):
        """If portfolio == benchmark, all effects are zero."""
        w = {"Tech": 0.5, "Fin": 0.5}
        r = {"Tech": 0.10, "Fin": 0.05}
        result = brinson_attribution(w, r, w, r)
        totals = result["totals"]
        assert abs(totals["active_return"]) < 1e-12
        assert abs(totals["allocation"]) < 1e-12
        assert abs(totals["selection"]) < 1e-12
        assert abs(totals["interaction"]) < 1e-12

    def test_pure_allocation_effect(self):
        """Same per-sector returns as benchmark, but different weights:
        all effect should be allocation, no selection."""
        w_p = {"Tech": 0.8, "Fin": 0.2}
        w_b = {"Tech": 0.5, "Fin": 0.5}
        # Identical sector returns
        r = {"Tech": 0.10, "Fin": 0.00}
        result = brinson_attribution(w_p, r, w_b, r)
        df = result["by_sector"]
        totals = result["totals"]
        # Selection = w_b * (r_p - r_b) = 0 because r_p == r_b in each sector
        assert abs(totals["selection"]) < 1e-12
        # Interaction = (w_p - w_b) * (r_p - r_b) = 0 (same)
        assert abs(totals["interaction"]) < 1e-12
        # Allocation = (w_p - w_b) * r_b ≠ 0
        assert totals["allocation"] != 0
        # Overweight Tech (10%) → positive allocation in Tech
        assert df.loc["Tech", "allocation"] > 0

    def test_pure_selection_effect(self):
        """Same weights, different per-sector returns → all effect is selection."""
        w = {"Tech": 0.5, "Fin": 0.5}
        r_p = {"Tech": 0.15, "Fin": 0.05}
        r_b = {"Tech": 0.10, "Fin": 0.00}
        result = brinson_attribution(w, r_p, w, r_b)
        totals = result["totals"]
        # Allocation = (w_p - w_b) * r_b = 0 because weights are identical
        assert abs(totals["allocation"]) < 1e-12
        # Interaction = (w_p - w_b) * ... = 0
        assert abs(totals["interaction"]) < 1e-12
        # Selection > 0 since portfolio outperformed in both sectors
        assert totals["selection"] > 0

    def test_effects_sum_to_active_return(self):
        """BHB identity: alloc + sel + inter == portfolio_return - benchmark_return."""
        w_p = {"Tech": 0.6, "Fin": 0.3, "Energy": 0.1}
        w_b = {"Tech": 0.3, "Fin": 0.4, "Energy": 0.3}
        r_p = {"Tech": 0.12, "Fin": 0.05, "Energy": -0.03}
        r_b = {"Tech": 0.10, "Fin": 0.04, "Energy": -0.05}
        result = brinson_attribution(w_p, r_p, w_b, r_b)
        totals = result["totals"]
        sum_effects = totals["allocation"] + totals["selection"] + totals["interaction"]
        assert sum_effects == pytest.approx(totals["active_return"], abs=1e-12)

    def test_per_sector_total_matches_sum(self):
        """For each sector, allocation + selection + interaction == total."""
        w_p = {"Tech": 0.6, "Fin": 0.4}
        w_b = {"Tech": 0.4, "Fin": 0.6}
        r_p = {"Tech": 0.10, "Fin": 0.05}
        r_b = {"Tech": 0.08, "Fin": 0.03}
        df = brinson_attribution(w_p, r_p, w_b, r_b)["by_sector"]
        for sec in df.index:
            row = df.loc[sec]
            assert row["allocation"] + row["selection"] + row["interaction"] == pytest.approx(
                row["total"], abs=1e-12,
            )

    def test_missing_sector_in_portfolio_is_negative_allocation(self):
        """Sector in benchmark but not in portfolio with positive r_b:
        you missed the upside → negative allocation effect."""
        w_p = {"Tech": 1.0}
        w_b = {"Tech": 0.5, "Fin": 0.5}
        # Fin did well in benchmark, portfolio missed it
        r_p = {"Tech": 0.10}
        r_b = {"Tech": 0.10, "Fin": 0.10}
        df = brinson_attribution(w_p, r_p, w_b, r_b)["by_sector"]
        # Fin: w_p=0, w_b=0.5 → alloc = (0 - 0.5) * 0.10 = -0.05
        assert df.loc["Fin", "allocation"] == pytest.approx(-0.05, abs=1e-9)

    def test_weights_normalised_internally(self):
        """Inputs that don't sum to 1 still work — normalised internally."""
        w_p = {"Tech": 5, "Fin": 5}  # un-normalised
        w_b = {"Tech": 2, "Fin": 2}
        r = {"Tech": 0.10, "Fin": 0.00}
        result = brinson_attribution(w_p, r, w_b, r)
        # Both normalise to 50/50 → zero active
        assert abs(result["totals"]["active_return"]) < 1e-12

    def test_empty_weights_returns_zero(self):
        result = brinson_attribution({}, {}, {}, {})
        assert result["by_sector"].empty
        for v in result["totals"].values():
            assert v == 0

    def test_negative_weights_dropped(self):
        """Short positions (negative weight) are excluded — long-only design."""
        w_p = {"Tech": 1.0, "Fin": -0.5}
        w_b = {"Tech": 0.5, "Fin": 0.5}
        r = {"Tech": 0.10, "Fin": 0.05}
        result = brinson_attribution(w_p, r, w_b, r)
        # Portfolio normalised: Tech=1.0, Fin dropped
        assert "Fin" in result["by_sector"].index  # in benchmark
        df = result["by_sector"]
        assert df.loc["Tech", "w_p"] == pytest.approx(1.0)
        assert df.loc["Fin", "w_p"] == pytest.approx(0.0)


# ===================================================================
# portfolio_sector_breakdown
# ===================================================================


class TestSectorBreakdown:
    def test_equal_weight_aggregation(self):
        symbols = ["AAPL", "MSFT", "JPM"]
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Fin"}
        returns = {"AAPL": 0.10, "MSFT": 0.20, "JPM": 0.05}
        weights, sec_returns = portfolio_sector_breakdown(
            symbols, sector_map, returns,
        )
        # Equal weight: each symbol 1/3 → Tech 2/3, Fin 1/3
        assert weights["Tech"] == pytest.approx(2 / 3, abs=1e-9)
        assert weights["Fin"] == pytest.approx(1 / 3, abs=1e-9)
        # Tech return = avg(AAPL 0.10, MSFT 0.20) = 0.15
        assert sec_returns["Tech"] == pytest.approx(0.15, abs=1e-9)
        assert sec_returns["Fin"] == pytest.approx(0.05, abs=1e-9)

    def test_unknown_symbol_bucketed(self):
        symbols = ["AAPL", "NEWCO"]
        sector_map = {"AAPL": "Tech"}
        returns = {"AAPL": 0.10, "NEWCO": -0.05}
        weights, sec_returns = portfolio_sector_breakdown(
            symbols, sector_map, returns,
        )
        assert "Tech" in weights
        assert "Unknown" in weights

    def test_missing_returns_drops_symbol(self):
        symbols = ["AAPL", "MSFT", "JPM"]
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Fin"}
        returns = {"AAPL": 0.10}  # MSFT, JPM missing
        weights, sec_returns = portfolio_sector_breakdown(
            symbols, sector_map, returns,
        )
        # Only AAPL counted → 100% Tech
        assert weights == {"Tech": 1.0}
        assert sec_returns == {"Tech": pytest.approx(0.10, abs=1e-9)}

    def test_custom_weights(self):
        symbols = ["AAPL", "JPM"]
        sector_map = {"AAPL": "Tech", "JPM": "Fin"}
        returns = {"AAPL": 0.10, "JPM": 0.05}
        weights, sec_returns = portfolio_sector_breakdown(
            symbols, sector_map, returns,
            weights={"AAPL": 3, "JPM": 1},
        )
        # 3:1 → 0.75 Tech, 0.25 Fin
        assert weights["Tech"] == pytest.approx(0.75, abs=1e-9)
        assert weights["Fin"] == pytest.approx(0.25, abs=1e-9)

    def test_empty_inputs(self):
        assert portfolio_sector_breakdown([], {}, {}) == ({}, {})


# ===================================================================
# compute_period_returns
# ===================================================================


class TestComputePeriodReturns:
    @pytest.fixture
    def prices(self):
        dates = pd.bdate_range("2025-01-01", periods=100)
        return pd.DataFrame({
            "AAPL": np.linspace(100, 120, 100),  # +20%
            "JPM":  np.linspace(50, 45, 100),    # -10%
        }, index=dates)

    def test_full_window_returns(self, prices):
        out = compute_period_returns(prices, "2025-01-01", "2025-12-31")
        assert out["AAPL"] == pytest.approx(0.20, abs=0.01)
        assert out["JPM"] == pytest.approx(-0.10, abs=0.01)

    def test_partial_window(self, prices):
        """Mid-period slice produces partial return."""
        out = compute_period_returns(prices, "2025-02-01", "2025-03-01")
        # Both still proportional within the slice
        assert -1 < out["AAPL"] < 1
        assert -1 < out["JPM"] < 1

    def test_window_too_short(self, prices):
        """Less than 2 bars in window → drop symbol."""
        out = compute_period_returns(prices, "2025-01-01", "2025-01-01")
        assert out == {}

    def test_empty_prices(self):
        assert compute_period_returns(pd.DataFrame(), "2025-01-01", "2025-12-31") == {}

    def test_skips_zero_starting_price(self):
        dates = pd.bdate_range("2025-01-01", periods=10)
        df = pd.DataFrame({"BAD": [0] * 5 + [10] * 5}, index=dates)
        out = compute_period_returns(df, "2025-01-01", "2025-12-31")
        assert out == {}


# ===================================================================
# Sector ETF map sanity
# ===================================================================


class TestSectorEtfMap:
    def test_all_etfs_are_strings(self):
        for sec, etf in SECTOR_ETF.items():
            assert isinstance(etf, str)
            assert etf.startswith("X")  # all SPDR sectors start with X

    def test_consumer_and_auto_share_xly(self):
        """Auto sits inside Consumer Discretionary."""
        assert SECTOR_ETF["Consumer"] == SECTOR_ETF["Automotive"]


# ===================================================================
# Realistic end-to-end with hypothetical traderbridge watchlist
# ===================================================================


class TestEndToEnd:
    def test_tech_heavy_watchlist_attribution(self):
        """Portfolio is 90% Tech + 10% Fin, benchmark equal-weighted.
        Tech up 15% (portfolio + benchmark same), Fin up 5%.
        Portfolio out-allocates to a winning sector."""
        w_p = {"Tech": 0.9, "Fin": 0.1}
        w_b = {"Tech": 0.5, "Fin": 0.5}
        # Equal returns per sector — pure allocation play
        r = {"Tech": 0.15, "Fin": 0.05}
        result = brinson_attribution(w_p, r, w_b, r)
        totals = result["totals"]
        # Portfolio return = 0.9*0.15 + 0.1*0.05 = 0.140
        # Benchmark return = 0.5*0.15 + 0.5*0.05 = 0.100
        # Active = 0.040
        assert totals["portfolio_return"] == pytest.approx(0.14, abs=1e-9)
        assert totals["benchmark_return"] == pytest.approx(0.10, abs=1e-9)
        assert totals["active_return"] == pytest.approx(0.04, abs=1e-9)
        # All from allocation (since per-sector returns are equal)
        assert totals["allocation"] == pytest.approx(0.04, abs=1e-9)
        assert abs(totals["selection"]) < 1e-12
        assert abs(totals["interaction"]) < 1e-12
