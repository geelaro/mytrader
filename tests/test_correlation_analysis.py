"""Tests for analysis/correlation_analysis.py — clustering / effective bets."""

import numpy as np
import pandas as pd
import pytest

from analysis.correlation_analysis import (
    correlation_clusters,
    correlation_matrix,
    correlation_summary,
    effective_bets,
    max_pairwise_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures with controlled correlation structure
# ---------------------------------------------------------------------------


@pytest.fixture
def three_independent():
    """3 independent random walks — pairwise corr ≈ 0."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0.0005, 0.015, (n, 3))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def two_perfectly_correlated_one_independent():
    """A and B share identical returns; C is independent."""
    rng = np.random.default_rng(7)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    shared = rng.normal(0.0005, 0.015, n)
    indep = rng.normal(0.0005, 0.015, n)
    return pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(shared)),
        "B": 200 * np.exp(np.cumsum(shared)),       # same shape
        "C": 150 * np.exp(np.cumsum(indep)),
    }, index=dates)


@pytest.fixture
def two_clusters():
    """4 symbols in 2 clusters: AB highly correlated, CD highly correlated,
    cross-cluster correlations are low."""
    rng = np.random.default_rng(1)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    base1 = rng.normal(0.0005, 0.015, n)
    base2 = rng.normal(0.0005, 0.015, n)
    noise = lambda: rng.normal(0, 0.002, n)
    return pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(base1 + noise())),
        "B": 100 * np.exp(np.cumsum(base1 + noise())),
        "C": 100 * np.exp(np.cumsum(base2 + noise())),
        "D": 100 * np.exp(np.cumsum(base2 + noise())),
    }, index=dates)


# ===================================================================
# correlation_matrix
# ===================================================================


class TestCorrelationMatrix:
    def test_diagonal_is_one(self, three_independent):
        c = correlation_matrix(three_independent)
        for sym in c.columns:
            assert c.loc[sym, sym] == pytest.approx(1.0, abs=1e-9)

    def test_independent_near_zero(self, three_independent):
        c = correlation_matrix(three_independent)
        # 500-sample IID Gaussian noise → correlations should be small
        for s1 in c.columns:
            for s2 in c.columns:
                if s1 != s2:
                    assert abs(c.loc[s1, s2]) < 0.2

    def test_insufficient_obs_returns_empty(self):
        small = pd.DataFrame({"A": [100, 101, 102], "B": [200, 201, 202]})
        assert correlation_matrix(small, min_obs=30).empty


# ===================================================================
# max_pairwise_correlation
# ===================================================================


class TestMaxPair:
    def test_finds_correlated_pair(self, two_perfectly_correlated_one_independent):
        result = max_pairwise_correlation(two_perfectly_correlated_one_independent)
        assert result is not None
        assert set(result["symbols"]) == {"A", "B"}
        assert result["correlation"] > 0.99

    def test_low_for_independent(self, three_independent):
        result = max_pairwise_correlation(three_independent)
        assert result["correlation"] < 0.3

    def test_returns_none_for_single_symbol(self):
        df = pd.DataFrame({"A": range(100, 200)})
        assert max_pairwise_correlation(df) is None


# ===================================================================
# effective_bets
# ===================================================================


class TestEffectiveBets:
    def test_independent_approaches_n_symbols(self, three_independent):
        """3 independent symbols → effective_n ≈ 3."""
        eb = effective_bets(three_independent)
        assert eb["n_symbols"] == 3
        assert eb["effective_n"] >= 2.5  # close to 3, with finite-sample noise
        assert eb["concentration_ratio"] > 0.8

    def test_two_perfectly_correlated_collapses(
            self, two_perfectly_correlated_one_independent):
        """A and B identical, C independent → effective_n ≈ 2, not 3."""
        eb = effective_bets(two_perfectly_correlated_one_independent)
        assert eb["n_symbols"] == 3
        # A and B are one bet; C is the other → ~2 effective bets
        assert 1.5 < eb["effective_n"] < 2.5

    def test_weighted_focuses_concentration(
            self, two_perfectly_correlated_one_independent):
        """When all weight is in the correlated pair, effective_n should
        be close to 1 (essentially one bet)."""
        eb = effective_bets(
            two_perfectly_correlated_one_independent,
            weights={"A": 1, "B": 1, "C": 0},  # C excluded
        )
        # A and B are perfectly correlated → 1 effective bet
        assert eb["effective_n"] < 1.5

    def test_single_symbol(self):
        df = pd.DataFrame({"A": np.linspace(100, 200, 100)},
                          index=pd.bdate_range("2025-01-01", periods=100))
        eb = effective_bets(df)
        assert eb["effective_n"] == 1.0

    def test_insufficient_data(self):
        small = pd.DataFrame({"A": [100, 101], "B": [200, 201]})
        assert effective_bets(small, min_obs=30) == {}


# ===================================================================
# correlation_clusters
# ===================================================================


class TestCorrelationClusters:
    def test_two_clusters_recovered(self, two_clusters):
        """4 symbols in 2 known clusters should produce 2 cluster groups."""
        result = correlation_clusters(two_clusters, distance_threshold=0.3)
        assert result["n_clusters"] == 2
        # A & B in same cluster, C & D in same cluster
        clusters = result["clusters"]
        # Find which cluster each symbol is in
        sym_to_cluster = {sym: cid for cid, syms in clusters.items() for sym in syms}
        assert sym_to_cluster["A"] == sym_to_cluster["B"]
        assert sym_to_cluster["C"] == sym_to_cluster["D"]
        assert sym_to_cluster["A"] != sym_to_cluster["C"]

    def test_strict_threshold_separates_all(self, two_clusters):
        """Very strict threshold (0.01) splits everything individually."""
        result = correlation_clusters(two_clusters, distance_threshold=0.01)
        assert result["n_clusters"] == 4

    def test_loose_threshold_merges_all(self, two_clusters):
        """Very loose threshold (1.5 > 1) merges everything into one cluster."""
        result = correlation_clusters(two_clusters, distance_threshold=1.5)
        assert result["n_clusters"] == 1

    def test_independent_three_separate(self, three_independent):
        """3 truly independent → 3 separate clusters (at default threshold)."""
        result = correlation_clusters(three_independent, distance_threshold=0.3)
        assert result["n_clusters"] == 3

    def test_single_symbol_returns_empty(self):
        df = pd.DataFrame({"A": range(100, 200)})
        assert correlation_clusters(df) == {}


# ===================================================================
# correlation_summary
# ===================================================================


class TestCorrelationSummary:
    def test_all_components_present(self, three_independent):
        s = correlation_summary(three_independent)
        assert "max_pair" in s
        assert "effective_bets" in s
        assert "clusters" in s

    def test_works_with_weights(self, two_clusters):
        s = correlation_summary(
            two_clusters,
            weights={"A": 1, "B": 1, "C": 1, "D": 1},
        )
        # Just check it doesn't crash and produces meaningful output
        assert s["effective_bets"]["n_symbols"] == 4
        assert s["effective_bets"]["effective_n"] < 4  # there's correlation
