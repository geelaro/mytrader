"""Tests for analysis.macro_calendar + analysis.historical_analog."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.macro_calendar import (
    is_cpi_release, is_fomc_release, is_nfp_release, macro_tag,
)
from analysis.historical_analog import (
    analog_summary_by_tag,
    find_closest_analogs,
    find_drop_events,
)


# ─── macro_calendar tests ─────────────────────────────────────────────


class TestNFP:
    def test_first_friday_is_nfp(self):
        # 2026-01-02 is the first Friday of January 2026
        assert is_nfp_release("2026-01-02") is True
        # 2025-12-05 is the first Friday of December 2025
        assert is_nfp_release("2025-12-05") is True

    def test_non_friday_is_not_nfp(self):
        assert is_nfp_release("2026-01-05") is False  # Monday
        assert is_nfp_release("2026-01-01") is False  # Thursday

    def test_second_friday_is_not_nfp(self):
        # 2026-01-09 is second Friday (day > 7)
        assert is_nfp_release("2026-01-09") is False

    def test_garbage_input_returns_false(self):
        assert is_nfp_release("not a date") is False
        assert is_nfp_release(None) is False


class TestCPI:
    def test_second_week_midweek_is_cpi(self):
        # 2026-01-14 is Wednesday, day 14 — typical CPI day
        assert is_cpi_release("2026-01-14") is True
        # 2026-02-12 is Thursday day 12
        assert is_cpi_release("2026-02-12") is True

    def test_friday_in_window_not_cpi(self):
        # 2026-01-09 is Friday day 9 (NFP-ish, not CPI)
        assert is_cpi_release("2026-01-09") is False

    def test_out_of_window_not_cpi(self):
        assert is_cpi_release("2026-01-08") is False  # day 8
        assert is_cpi_release("2026-01-16") is False  # day 16

    def test_monday_not_cpi(self):
        assert is_cpi_release("2026-01-12") is False  # Mon, day 12


class TestFOMC:
    def test_stub_returns_false(self):
        assert is_fomc_release("2026-01-28") is False


class TestMacroTag:
    def test_nfp_first_friday(self):
        assert macro_tag("2026-01-02") == "NFP"

    def test_cpi_midweek_midmonth(self):
        assert macro_tag("2026-01-14") == "CPI"

    def test_random_day_returns_none(self):
        assert macro_tag("2026-01-20") is None

    def test_unparseable_returns_none(self):
        assert macro_tag("garbage") is None


# ─── find_drop_events tests ───────────────────────────────────────────


def _series(closes, start="2026-01-01"):
    """Build a daily-indexed close series."""
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.Series(closes, index=idx)


class TestFindDropEvents:
    def test_detects_threshold_drop(self):
        # 100 → 96 = -4%, below -3% threshold
        s = _series([100, 96, 95, 94, 93, 92, 91, 90, 89, 88])
        events = find_drop_events(s, drop_threshold_pct=-3.0)
        assert len(events) == 1
        assert events.iloc[0]["drop_pct"] == pytest.approx(-4.0)

    def test_no_drops(self):
        s = _series([100, 100.5, 101, 101.5, 102])
        events = find_drop_events(s)
        assert events.empty

    def test_threshold_boundary_inclusive(self):
        # Exactly -3.0% should be included
        s = _series([100, 97, 95, 93, 91])
        events = find_drop_events(s, drop_threshold_pct=-3.0)
        assert len(events) >= 1  # 100→97 = -3.0% included

    def test_forward_returns_computed(self):
        # 100 → 95 (-5%) → 96 → 97 → 98 → 99 → 100 → 101 → 102 → 103
        s = _series([100, 95, 96, 97, 98, 99, 100, 101, 102, 103])
        events = find_drop_events(s, drop_threshold_pct=-3.0,
                                   horizons=(3, 5))
        assert len(events) == 1
        e = events.iloc[0]
        # From close=95, +3 days → 98, +5 days → 100
        assert e["fwd_3d"] == pytest.approx((98 / 95 - 1) * 100)
        assert e["fwd_5d"] == pytest.approx((100 / 95 - 1) * 100)

    def test_fwd_nan_when_insufficient_future(self):
        # drop at index 1; need index 1+3=4 for fwd_3d → 5 points minimum
        s = _series([100, 96, 97, 98, 99])
        events = find_drop_events(s, drop_threshold_pct=-3.0,
                                   horizons=(3, 60))
        e = events.iloc[0]
        assert pd.notna(e["fwd_3d"])         # 60 missing
        assert pd.isna(e["fwd_60d"])

    def test_macro_tag_assigned(self):
        # 2026-01-02 (Fri, first Friday) — NFP
        idx = pd.DatetimeIndex(["2025-12-31", "2026-01-02"])
        s = pd.Series([100, 95], index=idx)
        events = find_drop_events(s, drop_threshold_pct=-3.0)
        assert events.iloc[0]["macro_tag"] == "NFP"

    def test_empty_input(self):
        assert find_drop_events(pd.Series(dtype=float)).empty
        assert find_drop_events(None).empty


# ─── analog_summary_by_tag tests ──────────────────────────────────────


class TestAnalogSummary:
    def test_groups_by_tag(self):
        events = pd.DataFrame({
            "drop_pct": [-3.5, -4.0, -3.2, -5.0],
            "macro_tag": ["NFP", "NFP", "CPI", "OTHER"],
            "fwd_3d": [1.0, -2.0, 0.5, 3.0],
            "fwd_10d": [2.0, 1.0, -1.5, 5.0],
        })
        out = analog_summary_by_tag(events, horizons=(3, 10))
        nfp = out[out["tag"] == "NFP"].iloc[0]
        assert nfp["n"] == 2
        assert nfp["3d_mean"] == pytest.approx(-0.5)
        assert nfp["10d_mean"] == pytest.approx(1.5)
        assert nfp["3d_winrate"] == pytest.approx(50.0)

    def test_sort_by_n_desc(self):
        events = pd.DataFrame({
            "drop_pct": [-3.0] * 3,
            "macro_tag": ["A", "B", "B"],
            "fwd_3d": [1, 2, 3],
        })
        out = analog_summary_by_tag(events, horizons=(3,))
        assert out.iloc[0]["tag"] == "B"
        assert out.iloc[0]["n"] == 2

    def test_empty_input(self):
        assert analog_summary_by_tag(pd.DataFrame()).empty


# ─── find_closest_analogs tests ───────────────────────────────────────


class TestClosestAnalogs:
    def test_tag_match_wins_over_magnitude(self):
        events = pd.DataFrame({
            "drop_pct": [-3.5, -3.6, -4.0],
            "macro_tag": ["OTHER", "OTHER", "NFP"],
            "fwd_3d": [1, 2, 3],
        }, index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]))
        out = find_closest_analogs(events, today_drop_pct=-3.5,
                                    today_macro_tag="NFP", top_n=1)
        # NFP wins even though magnitudes are further
        assert out.iloc[0]["macro_tag"] == "NFP"

    def test_within_tag_closest_magnitude(self):
        events = pd.DataFrame({
            "drop_pct": [-3.0, -4.0, -5.0],
            "macro_tag": ["NFP", "NFP", "NFP"],
            "fwd_3d": [1, 2, 3],
        }, index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]))
        out = find_closest_analogs(events, today_drop_pct=-3.9,
                                    today_macro_tag="NFP", top_n=2)
        # -4.0 closest, then -3.0 (diff 0.9 vs -5.0 diff 1.1)
        assert out.iloc[0]["drop_pct"] == pytest.approx(-4.0)

    def test_none_tag_ranks_by_magnitude(self):
        events = pd.DataFrame({
            "drop_pct": [-3.0, -4.0, -5.0],
            "macro_tag": ["A", "B", "C"],
            "fwd_3d": [1, 2, 3],
        }, index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]))
        out = find_closest_analogs(events, today_drop_pct=-3.9,
                                    today_macro_tag=None, top_n=1)
        assert out.iloc[0]["drop_pct"] == pytest.approx(-4.0)

    def test_empty_input(self):
        out = find_closest_analogs(pd.DataFrame(), today_drop_pct=-3.0)
        assert out.empty

    def test_top_n_limits(self):
        events = pd.DataFrame({
            "drop_pct": [-3.0, -3.5, -4.0, -4.5, -5.0],
            "macro_tag": ["A"] * 5,
            "fwd_3d": [1, 2, 3, 4, 5],
        }, index=pd.bdate_range("2026-01-01", periods=5))
        out = find_closest_analogs(events, today_drop_pct=-3.5,
                                    today_macro_tag=None, top_n=3)
        assert len(out) == 3
