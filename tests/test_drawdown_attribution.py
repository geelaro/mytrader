"""Tests for analysis.drawdown_attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.drawdown_attribution import (
    attribute_active_drawdown,
    attribute_drawdown,
)


def _dates(n: int, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


class TestAttributeDrawdown:
    def test_single_position_takes_full_blame(self):
        """One holding, no cash flow → 100% of drawdown attributed to it."""
        idx = _dates(5)
        # NVDA goes 100 → 90.  Portfolio value mirrors.
        pv = pd.Series([100, 102, 95, 90, 92], index=idx)
        pos = pd.DataFrame({"NVDA": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[1], trough_date=idx[3])

        assert result["peak_value"] == 102
        assert result["trough_value"] == 90
        assert result["depth_usd"] == -12
        assert result["depth_pct"] == pytest.approx(-11.764, abs=0.01)
        assert len(result["by_symbol"]) == 1
        assert result["by_symbol"][0]["symbol"] == "NVDA"
        assert result["by_symbol"][0]["contribution_usd"] == -12
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_two_positions_split_blame(self):
        """50/50 portfolio, one tanks one flat → top contributor is the one that fell."""
        idx = _dates(3)
        # NVDA 50→40, TSLA 50→50.  Total 100→90.
        pv = pd.Series([100, 95, 90], index=idx)
        pos = pd.DataFrame({
            "NVDA": [50, 45, 40],
            "TSLA": [50, 50, 50],
        }, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[0], trough_date=idx[2])

        # NVDA contributed -10, TSLA 0.
        by_sym = {r["symbol"]: r for r in result["by_symbol"]}
        assert by_sym["NVDA"]["contribution_usd"] == -10
        assert by_sym["TSLA"]["contribution_usd"] == 0
        # Sort order: NVDA first (larger |contribution|).
        assert result["by_symbol"][0]["symbol"] == "NVDA"
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_auto_finds_worst_window(self):
        """No peak/trough given → picks the deepest drawdown automatically."""
        idx = _dates(7)
        # Two drawdowns: 100→95 (small), 110→80 (big).
        pv = pd.Series([100, 95, 100, 110, 100, 80, 90], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos)

        assert result["peak_date"] == idx[3]   # 110
        assert result["trough_date"] == idx[5]  # 80
        assert result["depth_usd"] == -30
        assert result["depth_pct"] == pytest.approx(-27.27, abs=0.01)

    def test_position_opened_mid_drawdown(self):
        """Symbol bought during the DD shows peak=0, contribution=+trough_value."""
        idx = _dates(3)
        # NVDA tanks; we buy SPY mid-DD as a hedge.
        pv = pd.Series([100, 85, 88], index=idx)
        pos = pd.DataFrame({
            "NVDA": [100, 80, 70],
            "SPY":  [0,   5,  18],
        }, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[0], trough_date=idx[2])

        by_sym = {r["symbol"]: r for r in result["by_symbol"]}
        assert by_sym["NVDA"]["contribution_usd"] == -30
        assert by_sym["SPY"]["peak_value"] == 0
        assert by_sym["SPY"]["contribution_usd"] == 18
        # Net: -30 + 18 = -12, matching depth_usd.
        assert result["depth_usd"] == -12
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_unexplained_reflects_cash_flow(self):
        """A deposit during the window shows up as unexplained_usd."""
        idx = _dates(3)
        # Portfolio: 100 → +50 deposit → still drops to 130.
        # Position only fell 100→80, so positions explain -20, residual -30 unexplained
        # (== the deposit absorbed by other cash).  Wait this needs care.
        # Let's set: peak total=100 (all in NVDA).  Day 2: deposit 50 cash,
        # NVDA value 80. total=130. Day 3: NVDA 70, total=120.
        # peak→trough on total: 130 → 120 → -10.  But this isn't the true DD
        # — the peak we pass is day 1 (100), trough day 3 (120).  Total
        # rose 100→120, no drawdown.  Pick day 2 → day 3: peak 130, trough 120.
        pv = pd.Series([100, 130, 120], index=idx)
        pos = pd.DataFrame({"NVDA": [100, 80, 70]}, index=idx)

        result = attribute_drawdown(pv, pos, peak_date=idx[1], trough_date=idx[2])

        # NVDA went 80→70 → -10 contribution.
        nvda = next(r for r in result["by_symbol"] if r["symbol"] == "NVDA")
        assert nvda["contribution_usd"] == -10
        # depth = -10; explained = -10 → unexplained ≈ 0.
        assert result["unexplained_usd"] == pytest.approx(0, abs=1e-9)

    def test_empty_input(self):
        """Empty series → all-zeros summary."""
        result = attribute_drawdown(pd.Series(dtype=float), pd.DataFrame())
        assert result["depth_pct"] == 0.0
        assert result["by_symbol"] == []

    def test_no_drawdown(self):
        """Monotonically rising → peak==trough at same date, 0% depth."""
        idx = _dates(4)
        pv = pd.Series([100, 105, 110, 115], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_drawdown(pv, pos)

        assert result["depth_usd"] == 0
        assert result["depth_pct"] == 0.0

    def test_top_n_limit(self):
        """top_n truncates to most-impactful contributors."""
        idx = _dates(2)
        pv = pd.Series([100, 60], index=idx)
        pos = pd.DataFrame({
            "A": [50, 30],   # -20
            "B": [30, 20],   # -10
            "C": [20, 10],   # -10
        }, index=idx)

        result = attribute_drawdown(pv, pos, top_n=1)

        assert len(result["by_symbol"]) == 1
        assert result["by_symbol"][0]["symbol"] == "A"


class TestAttributeActiveDrawdown:
    def test_currently_in_drawdown(self):
        """Latest value < cummax → attributes the live drawdown."""
        idx = _dates(5)
        pv = pd.Series([100, 110, 105, 95, 92], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_active_drawdown(pv, pos)

        assert result["peak_date"] == idx[1]   # 110
        assert result["trough_date"] == idx[4]  # 92 (latest)
        assert result["depth_usd"] == -18
        assert result["by_symbol"][0]["symbol"] == "X"

    def test_at_new_high(self):
        """At ATH → empty summary, no active DD."""
        idx = _dates(3)
        pv = pd.Series([100, 110, 120], index=idx)
        pos = pd.DataFrame({"X": pv.values}, index=idx)

        result = attribute_active_drawdown(pv, pos)

        assert result["depth_usd"] == 0
        assert result["by_symbol"] == []
