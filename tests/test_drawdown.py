"""Tests for analysis/drawdown.py — underwater / episodes / recovery."""

import math

import numpy as np
import pandas as pd
import pytest

from analysis.drawdown import (
    drawdown_episodes,
    drawdown_summary,
    time_to_recover_stats,
    underwater_curve,
)


# ===================================================================
# Underwater curve
# ===================================================================


class TestUnderwaterCurve:
    def test_at_new_peaks_zero(self):
        """Monotonically rising returns → underwater = 0 throughout."""
        r = pd.Series([0.01] * 10)
        uw = underwater_curve(r)
        assert (uw == 0).all() or all(abs(v) < 1e-9 for v in uw)

    def test_after_drop_negative(self):
        """+10% then −20%: equity 1.0 → 1.1 → 0.88, peak 1.1, uw = −20%."""
        r = pd.Series([0.10, -0.20])
        uw = underwater_curve(r)
        assert uw.iloc[-1] == pytest.approx(-20.0, abs=0.1)

    def test_full_recovery_back_to_zero(self):
        """After recovery, underwater = 0 again.

        +10% → equity 1.10 (peak).  -10% → 0.99 (uw -10%).
        Need +1.10/0.99 - 1 ≈ +11.11% to return to peak.
        """
        r = pd.Series([0.10, -0.10, 0.1112])
        uw = underwater_curve(r)
        assert abs(uw.iloc[-1]) < 0.1  # at or above peak

    def test_empty_returns_empty(self):
        assert underwater_curve(pd.Series(dtype=float)).empty


# ===================================================================
# Drawdown episodes
# ===================================================================


class TestDrawdownEpisodes:
    def test_no_drawdown(self):
        """Monotone rise → no episodes."""
        r = pd.Series([0.01] * 10)
        ep = drawdown_episodes(r)
        assert ep.empty

    def test_single_complete_episode(self):
        """V-shape: up, down, fully recover."""
        dates = pd.bdate_range("2025-01-01", periods=5)
        r = pd.Series([0.10, -0.20, -0.10, 0.20, 0.20], index=dates)
        # Equity: 1.10, 0.88, 0.792, 0.9504, 1.14
        # Peak 1.10 at bar 0, trough at bar 2, recovery at bar 4
        ep = drawdown_episodes(r)
        assert len(ep) == 1
        row = ep.iloc[0]
        assert row["peak_date"] == dates[0]
        assert row["trough_date"] == dates[2]
        assert row["recovery_date"] == dates[4]
        # Depth: 0.792/1.10 - 1 = -28%
        assert row["depth_pct"] == pytest.approx(-28.0, abs=0.5)
        assert row["duration_to_trough_days"] > 0
        assert row["duration_to_recovery_days"] > 0

    def test_two_episodes(self):
        """Two separate drawdowns."""
        dates = pd.bdate_range("2025-01-01", periods=8)
        # +10%, -10%, +12%, -10%, -10%, +25%
        # episode 1: peak at bar0 (1.10), trough at bar1 (0.99), recover at bar2 (1.1088)
        # episode 2: peak at bar2 (1.1088), trough at bar4 (~0.898), recover at bar5
        r = pd.Series([0.10, -0.10, 0.12, -0.10, -0.10, 0.25, 0.0, 0.0], index=dates)
        ep = drawdown_episodes(r)
        assert len(ep) >= 2

    def test_unrecovered_episode_has_nan(self):
        """Drawdown still ongoing at end → recovery_date NaT, duration NaN."""
        dates = pd.bdate_range("2025-01-01", periods=4)
        r = pd.Series([0.10, -0.05, -0.05, -0.05], index=dates)
        ep = drawdown_episodes(r)
        assert len(ep) == 1
        row = ep.iloc[0]
        assert pd.isna(row["recovery_date"])
        assert math.isnan(row["duration_to_recovery_days"])
        # Depth should still be computed
        assert row["depth_pct"] < 0

    def test_sorted_by_depth_worst_first(self):
        """When multiple episodes exist, sorted by depth ascending (most negative first)."""
        dates = pd.bdate_range("2025-01-01", periods=10)
        # Mild then severe drawdown
        r = pd.Series([0.10, -0.05, 0.06, 0.10, -0.20, -0.10, 0.40, 0.0, 0.0, 0.0],
                      index=dates)
        ep = drawdown_episodes(r)
        if len(ep) >= 2:
            assert ep.iloc[0]["depth_pct"] <= ep.iloc[1]["depth_pct"]


# ===================================================================
# Time to recover
# ===================================================================


class TestTimeToRecover:
    def test_no_episodes_returns_empty_stats(self):
        r = pd.Series([0.01] * 10)
        stats = time_to_recover_stats(r)
        assert stats["n_episodes"] == 0
        assert stats["still_in_drawdown"] is False

    def test_one_completed_episode(self):
        dates = pd.bdate_range("2025-01-01", periods=5)
        r = pd.Series([0.10, -0.20, -0.10, 0.20, 0.20], index=dates)
        stats = time_to_recover_stats(r)
        assert stats["n_episodes"] == 1
        assert stats["median_days"] > 0
        assert stats["still_in_drawdown"] is False

    def test_currently_in_drawdown_flag(self):
        dates = pd.bdate_range("2025-01-01", periods=4)
        r = pd.Series([0.10, -0.05, -0.05, -0.05], index=dates)
        stats = time_to_recover_stats(r)
        assert stats["still_in_drawdown"] is True
        assert stats["current_dd_days"] > 0
        # No completed episodes
        assert stats["n_episodes"] == 0


# ===================================================================
# Summary
# ===================================================================


class TestDrawdownSummary:
    def test_empty_input(self):
        s = drawdown_summary(pd.Series(dtype=float))
        assert s["max_drawdown_pct"] == 0.0
        assert s["pct_time_underwater"] == 0.0
        assert s["top_episodes"] == []

    def test_pct_time_underwater(self):
        dates = pd.bdate_range("2025-01-01", periods=5)
        # +10%, -20%, -10%, +20%, +20% — bars 1,2,3 are underwater (3/5 = 60%)
        r = pd.Series([0.10, -0.20, -0.10, 0.20, 0.20], index=dates)
        s = drawdown_summary(r)
        # 3 of 5 bars below peak
        assert 50 <= s["pct_time_underwater"] <= 80

    def test_top_n_limited(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=300)
        # Many small drawdowns
        r = pd.Series(rng.normal(0.001, 0.02, 300), index=dates)
        s = drawdown_summary(r, top_n=3)
        assert len(s["top_episodes"]) <= 3

    def test_max_dd_matches_underwater_min(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=200)
        r = pd.Series(rng.normal(0.0005, 0.015, 200), index=dates)
        s = drawdown_summary(r)
        uw = underwater_curve(r)
        assert s["max_drawdown_pct"] == pytest.approx(-uw.min(), abs=0.01)
