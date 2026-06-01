"""Tests for analysis/concentration.py — HHI / sector / correlation metrics."""

import numpy as np
import pandas as pd
import pytest

from analysis.concentration import (
    concentration_summary,
    correlation_hhi,
    effective_n,
    hhi,
    hhi_label,
    sector_exposure,
    sector_hhi,
    top_n_weight,
)


# ===================================================================
# HHI
# ===================================================================


class TestHHI:
    def test_single_position_hhi_10000(self):
        assert hhi({"AAPL": 1.0}) == pytest.approx(10000, abs=0.1)

    def test_equal_weight_n_positions(self):
        """N equally-weighted positions → HHI = 10000 / N."""
        for n in (2, 4, 5, 10):
            weights = {f"S{i}": 1 for i in range(n)}
            assert hhi(weights) == pytest.approx(10000 / n, abs=0.1)

    def test_normalisation(self):
        """HHI invariant to scaling — only relative weights matter."""
        assert hhi({"A": 1, "B": 1}) == pytest.approx(5000, abs=0.1)
        assert hhi({"A": 100, "B": 100}) == pytest.approx(5000, abs=0.1)
        assert hhi({"A": 0.5, "B": 0.5}) == pytest.approx(5000, abs=0.1)

    def test_skewed_higher_than_equal(self):
        equal = hhi({"A": 1, "B": 1, "C": 1, "D": 1})
        skewed = hhi({"A": 4, "B": 2, "C": 1, "D": 1})  # same n, more concentrated
        assert skewed > equal

    def test_empty_returns_zero(self):
        assert hhi({}) == 0.0
        assert hhi(None) == 0.0
        assert hhi({"A": 0, "B": 0}) == 0.0

    def test_negative_weights_excluded(self):
        """Short positions intentionally excluded — HHI is for gross exposure."""
        result = hhi({"A": 1.0, "B": -0.5})
        # Only A counted → single-position HHI
        assert result == pytest.approx(10000, abs=0.1)


class TestHHILabel:
    def test_thresholds(self):
        assert hhi_label(1000) == "分散"
        assert hhi_label(1499) == "分散"
        assert hhi_label(1500) == "中等集中"
        assert hhi_label(2499) == "中等集中"
        assert hhi_label(2500) == "高度集中"
        assert hhi_label(10000) == "高度集中"


# ===================================================================
# Effective N
# ===================================================================


class TestEffectiveN:
    def test_equal_weight_recovers_n(self):
        for n in (3, 5, 10):
            weights = {f"S{i}": 1 for i in range(n)}
            assert effective_n(weights) == pytest.approx(n, abs=0.01)

    def test_dominated_lt_count(self):
        """One position dominating → effective_n < raw count."""
        weights = {"A": 0.8, "B": 0.1, "C": 0.1}
        en = effective_n(weights)
        # 1 / (0.64 + 0.01 + 0.01) ≈ 1.51
        assert 1 < en < 2

    def test_empty_returns_zero(self):
        assert effective_n({}) == 0.0


# ===================================================================
# Top-N weight
# ===================================================================


class TestTopNWeight:
    def test_top_3_largest(self):
        weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.05, "E": 0.05}
        assert top_n_weight(weights, n=3) == pytest.approx(0.9, abs=0.001)

    def test_top_n_capped_at_size(self):
        """Asking for top 10 when only 3 positions exist → returns 1.0."""
        weights = {"A": 1, "B": 1, "C": 1}
        assert top_n_weight(weights, n=10) == pytest.approx(1.0, abs=0.001)

    def test_zero_n(self):
        assert top_n_weight({"A": 1}, n=0) == 0.0


# ===================================================================
# Sector exposure
# ===================================================================


class TestSectorExposure:
    def test_aggregates_by_sector(self):
        weights = {"AAPL": 1, "MSFT": 1, "TSLA": 1, "JPM": 1}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "TSLA": "Auto", "JPM": "Fin"}
        exp = sector_exposure(weights, sectors)
        # Equal weight 4 names → 0.25 each → Tech 0.5, Auto 0.25, Fin 0.25
        assert exp["Tech"] == pytest.approx(0.5, abs=0.001)
        assert exp["Auto"] == pytest.approx(0.25, abs=0.001)
        assert exp["Fin"] == pytest.approx(0.25, abs=0.001)

    def test_unknown_symbol_bucketed(self):
        weights = {"AAPL": 1, "NEWCO": 1}
        sectors = {"AAPL": "Tech"}
        exp = sector_exposure(weights, sectors, unknown_label="Other")
        assert exp["Tech"] == pytest.approx(0.5, abs=0.001)
        assert exp["Other"] == pytest.approx(0.5, abs=0.001)

    def test_sorted_descending(self):
        weights = {"AAPL": 0.1, "TSLA": 0.5, "JPM": 0.4}
        sectors = {"AAPL": "Tech", "TSLA": "Auto", "JPM": "Fin"}
        exp = sector_exposure(weights, sectors)
        keys = list(exp.keys())
        # Auto biggest (0.5), Fin (0.4), Tech (0.1)
        assert keys == ["Auto", "Fin", "Tech"]


class TestSectorHHI:
    def test_all_same_sector_max(self):
        """11 names all in Tech → sector HHI = 10000 (max concentration)."""
        weights = {f"T{i}": 1 for i in range(11)}
        sectors = {k: "Tech" for k in weights}
        assert sector_hhi(weights, sectors) == pytest.approx(10000, abs=0.1)

    def test_split_two_sectors(self):
        weights = {"A": 1, "B": 1, "C": 1, "D": 1}
        sectors = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        # 50/50 sector split → HHI = 0.5² + 0.5² = 0.5 → 5000
        assert sector_hhi(weights, sectors) == pytest.approx(5000, abs=0.1)


# ===================================================================
# Correlation-adjusted HHI
# ===================================================================


class TestCorrelationHHI:
    def test_zero_correlation_equals_plain_hhi(self):
        weights = {"A": 1, "B": 1}
        corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                            index=["A", "B"], columns=["A", "B"])
        result = correlation_hhi(weights, corr)
        # wᵀ I w * 10000 = (0.5² + 0.5²) * 10000 = 5000
        assert result == pytest.approx(5000, abs=0.1)

    def test_perfect_correlation_equals_single_position(self):
        """Two 50/50 perfectly correlated → corr_hhi = 10000."""
        weights = {"A": 1, "B": 1}
        corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]],
                            index=["A", "B"], columns=["A", "B"])
        result = correlation_hhi(weights, corr)
        # wᵀ * J * w * 10000 = (0.5+0.5)² * 10000 = 10000
        assert result == pytest.approx(10000, abs=0.1)

    def test_none_corr_falls_back_to_plain_hhi(self):
        weights = {"A": 1, "B": 1, "C": 1}
        assert correlation_hhi(weights, None) == pytest.approx(hhi(weights), abs=0.1)

    def test_missing_symbols_in_corr_falls_back(self):
        """Corr matrix doesn't cover the symbols → plain HHI."""
        weights = {"X": 1, "Y": 1}
        corr = pd.DataFrame([[1.0]], index=["Z"], columns=["Z"])
        assert correlation_hhi(weights, corr) == pytest.approx(hhi(weights), abs=0.1)


# ===================================================================
# Summary
# ===================================================================


class TestSummary:
    def test_full_summary(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Fin"}
        corr = pd.DataFrame(
            [[1.0, 0.7, 0.2], [0.7, 1.0, 0.2], [0.2, 0.2, 1.0]],
            index=["AAPL", "MSFT", "JPM"],
            columns=["AAPL", "MSFT", "JPM"],
        )
        s = concentration_summary(weights, sectors, corr)
        assert s["n_holdings"] == 3
        assert s["hhi"] > 0
        assert s["hhi_label"] in {"分散", "中等集中", "高度集中"}
        assert s["effective_n"] < 3
        assert 0 < s["top_3_weight"] <= 1.0
        assert "sector_exposure" in s
        # Tech 80%, Fin 20% → sector HHI = 0.8² + 0.2² = 0.68 → 6800
        assert s["sector_hhi"] == pytest.approx(6800, abs=1)
        assert "correlation_hhi" in s

    def test_summary_without_sectors_or_corr(self):
        s = concentration_summary({"A": 1, "B": 1})
        assert "sector_exposure" not in s
        assert "correlation_hhi" not in s
        assert s["hhi"] == pytest.approx(5000, abs=0.1)


# ===================================================================
# Realistic watchlist scenario (the motivating bug)
# ===================================================================


class TestRealisticTraderbridgeWatchlist:
    """11 names all in Technology — the symbol HHI looks fine,
    sector HHI screams concentration. This is the value-add."""

    def test_symbol_hhi_misleading_when_sectors_concentrated(self):
        # 11 equal-weight Tech names
        weights = {f"TECH{i}": 1 for i in range(11)}
        sectors = {k: "Technology" for k in weights}

        sym_h = hhi(weights)
        sec_h = sector_hhi(weights, sectors)

        # Symbol HHI ≈ 10000/11 ≈ 909 — looks "diversified"
        assert sym_h < 1000
        assert hhi_label(sym_h) == "分散"
        # Sector HHI = 10000 — actually highly concentrated
        assert sec_h == pytest.approx(10000, abs=0.1)
        assert hhi_label(sec_h) == "高度集中"
