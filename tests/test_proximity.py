"""Tests for analysis.proximity."""

from __future__ import annotations

import pytest

from analysis.proximity import (
    proximity_summary,
    proximity_summary_by_symbol,
    proximity_warnings,
)


def _result(symbol="X", strategy="s", price=100, signal=0, **indicators):
    return {
        "symbol": symbol, "strategy": strategy,
        "signal": signal, "price": price,
        "indicators": indicators,
    }


class TestMACDNearCross:
    def test_alert_when_very_close(self):
        # MACD=10, signal=10.3 → diff=-0.3, rel=0.3/10=3% → alert
        out = proximity_warnings(_result(MACD=10, MACD_signal=10.3))
        macd = [w for w in out if w["type"] == "macd_near_cross"]
        assert len(macd) == 1
        assert macd[0]["level"] == "alert"
        assert macd[0]["direction"] == "bullish"  # diff < 0 means approaching golden

    def test_warn_when_moderately_close(self):
        # diff=1.0, magnitude=10 → 10%, warn
        out = proximity_warnings(_result(MACD=10, MACD_signal=9.0))
        macd = [w for w in out if w["type"] == "macd_near_cross"]
        assert len(macd) == 1
        assert macd[0]["level"] == "warn"
        assert macd[0]["direction"] == "bearish"  # currently above, may cross down

    def test_no_warning_when_far(self):
        # diff=5, magnitude=10 → 50%
        out = proximity_warnings(_result(MACD=10, MACD_signal=5.0))
        assert not [w for w in out if w["type"] == "macd_near_cross"]

    def test_no_warning_when_indicator_missing(self):
        out = proximity_warnings(_result(K=50, D=50))  # no MACD
        assert not [w for w in out if w["type"] == "macd_near_cross"]


class TestKDJ:
    def test_kd_near_cross_alert(self):
        out = proximity_warnings(_result(K=50.5, D=50.0, J=51.5))
        kd = [w for w in out if w["type"] == "kdj_kd_near_cross"]
        assert kd and kd[0]["level"] == "alert"

    def test_kd_near_cross_warn(self):
        out = proximity_warnings(_result(K=52, D=50, J=56))
        kd = [w for w in out if w["type"] == "kdj_kd_near_cross"]
        assert kd and kd[0]["level"] == "warn"

    def test_j_extreme_overbought(self):
        out = proximity_warnings(_result(K=80, D=80, J=115))
        j = [w for w in out if w["type"] == "kdj_overbought_extreme"]
        assert j and j[0]["level"] == "alert"
        assert j[0]["direction"] == "bearish"

    def test_j_overbought_warn(self):
        out = proximity_warnings(_result(K=85, D=85, J=105))
        j = [w for w in out if w["type"] == "kdj_overbought_extreme"]
        assert j and j[0]["level"] == "warn"

    def test_j_extreme_oversold(self):
        out = proximity_warnings(_result(K=10, D=10, J=-15))
        j = [w for w in out if w["type"] == "kdj_oversold_extreme"]
        assert j and j[0]["level"] == "alert"
        assert j[0]["direction"] == "bullish"

    def test_j_neutral_no_warn(self):
        out = proximity_warnings(_result(K=50, D=50, J=50))
        # No extreme warning, but KD gap=0 may trigger near_cross
        j = [w for w in out if w["type"] == "kdj_overbought_extreme"]
        assert not j


class TestDonchian:
    def test_near_upper_alert(self):
        # 99.5/100 = 0.995 > 0.99 → alert
        out = proximity_warnings(_result(price=99.5, Donchian_upper=100,
                                          Donchian_lower=80))
        dn = [w for w in out if w["type"] == "donchian_near_upper"]
        assert dn and dn[0]["level"] == "alert"

    def test_near_upper_warn(self):
        # 98.5/100 = 0.985 → warn
        out = proximity_warnings(_result(price=98.5, Donchian_upper=100,
                                          Donchian_lower=80))
        dn = [w for w in out if w["type"] == "donchian_near_upper"]
        assert dn and dn[0]["level"] == "warn"

    def test_near_lower(self):
        # 80.5/80 = 1.006 < 1.01 → alert (bearish)
        out = proximity_warnings(_result(price=80.5, Donchian_upper=120,
                                          Donchian_lower=80))
        dn = [w for w in out if w["type"] == "donchian_near_lower"]
        assert dn and dn[0]["level"] == "alert"
        assert dn[0]["direction"] == "bearish"


class TestMA:
    def test_near_breakdown(self):
        # Close 100 / MA 99.5 → +0.5% → alert; rel_dist > 0 → bearish (about to break down)
        out = proximity_warnings(_result(price=100, MA=99.5))
        ma = [w for w in out if w["type"] == "ma_near_breakdown"]
        assert ma and ma[0]["level"] == "alert"

    def test_near_breakout(self):
        # Close 99.5 / MA 100 → -0.5% → alert; bullish (about to break up)
        out = proximity_warnings(_result(price=99.5, MA=100))
        ma = [w for w in out if w["type"] == "ma_near_breakout"]
        assert ma and ma[0]["level"] == "alert"

    def test_far_from_ma_no_warning(self):
        out = proximity_warnings(_result(price=120, MA=100))
        assert not [w for w in out if w["type"].startswith("ma_near")]


class TestNDayHigh:
    def test_near_high(self):
        out = proximity_warnings(_result(price=99.5, N_day_high=100))
        n = [w for w in out if w["type"] == "near_n_day_high"]
        assert n and n[0]["level"] == "alert"


class TestSummary:
    def test_ranks_alerts_above_warns(self):
        scans = [
            # symbol A: only warn
            _result(symbol="A", strategy="s1", MACD=10, MACD_signal=9.0),
            # symbol B: alert
            _result(symbol="B", strategy="s2", K=50.5, D=50.0, J=51.5),
        ]
        out = proximity_summary(scans)
        assert out[0]["symbol"] == "B"  # alert first
        assert out[1]["symbol"] == "A"

    def test_excludes_empty(self):
        scans = [
            _result(symbol="A", MACD=10, MACD_signal=5.0),  # far, no warn
            _result(symbol="B", J=115),                      # extreme alert
        ]
        out = proximity_summary(scans)
        syms = [r["symbol"] for r in out]
        assert "A" not in syms
        assert "B" in syms

    def test_n_warnings_breaks_tie(self):
        scans = [
            _result(symbol="A", strategy="s",  # 1 warn
                    MACD=10, MACD_signal=9.0),
            _result(symbol="B", strategy="s",  # 2 warns
                    MACD=10, MACD_signal=9.0, K=52, D=50, J=56),
        ]
        out = proximity_summary(scans)
        assert out[0]["symbol"] == "B"
        assert out[0]["n_warnings"] == 2

    def test_empty_input(self):
        assert proximity_summary([]) == []
        assert proximity_summary(None) == []


class TestSummaryBySymbol:
    def test_collapses_multiple_strategies_per_symbol(self):
        scans = [
            _result(symbol="QQQ", strategy="daily_macd_kdj",
                    MACD=10, MACD_signal=10.3),     # alert macd cross
            _result(symbol="QQQ", strategy="weekly_macd_kdj", J=115),  # alert J extreme
            _result(symbol="QQQ", strategy="spy_ma_breakout",
                    price=99.5, N_day_high=100),    # alert near high
        ]
        out = proximity_summary_by_symbol(scans)
        assert len(out) == 1
        row = out[0]
        assert row["symbol"] == "QQQ"
        assert row["n_strategies"] == 3
        assert row["n_warnings"] == 3
        assert set(row["by_strategy"].keys()) == {
            "daily_macd_kdj", "weekly_macd_kdj", "spy_ma_breakout",
        }
        assert row["max_level"] == "alert"

    def test_max_level_is_highest_across_strategies(self):
        scans = [
            _result(symbol="X", strategy="a", MACD=10, MACD_signal=9.0),   # warn
            _result(symbol="X", strategy="b", J=115),                       # alert
        ]
        out = proximity_summary_by_symbol(scans)
        assert out[0]["max_level"] == "alert"

    def test_sort_alerts_above_warns(self):
        scans = [
            _result(symbol="A", strategy="s", MACD=10, MACD_signal=9.0),   # warn only
            _result(symbol="B", strategy="s", J=115),                       # alert
        ]
        out = proximity_summary_by_symbol(scans)
        assert [r["symbol"] for r in out] == ["B", "A"]

    def test_sort_breaks_tie_by_total_n(self):
        scans = [
            _result(symbol="A", strategy="s", MACD=10, MACD_signal=10.3),   # alert + 1
            _result(symbol="B", strategy="s1", MACD=10, MACD_signal=10.3),  # alert
            _result(symbol="B", strategy="s2", J=115),                       # +alert
        ]
        out = proximity_summary_by_symbol(scans)
        assert out[0]["symbol"] == "B"   # 2 warnings ranks above 1 at same level
        assert out[0]["n_warnings"] == 2

    def test_excludes_clean_symbols(self):
        scans = [
            _result(symbol="A", MACD=10, MACD_signal=5),    # far, no warn
            _result(symbol="B", J=115),                      # alert
        ]
        out = proximity_summary_by_symbol(scans)
        assert [r["symbol"] for r in out] == ["B"]

    def test_empty_input(self):
        assert proximity_summary_by_symbol([]) == []
        assert proximity_summary_by_symbol(None) == []


class TestEdgeCases:
    def test_nan_indicator_ignored(self):
        out = proximity_warnings(_result(MACD=float("nan"), MACD_signal=5))
        assert not [w for w in out if w["type"] == "macd_near_cross"]

    def test_zero_price_skips_price_based(self):
        out = proximity_warnings(_result(price=0, Donchian_upper=100,
                                          Donchian_lower=80, MA=90))
        assert not [w for w in out if w["type"].startswith(("donchian", "ma_", "near_n_day"))]
