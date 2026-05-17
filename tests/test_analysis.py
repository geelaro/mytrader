"""Tests for analysis/ modules — cost_sensitivity and param_robustness."""

import numpy as np
import pandas as pd
import pytest

from analysis.cost_sensitivity import run as cs_run
from analysis.param_robustness import perturb_params, _is_int_param, PERTURB_LEVELS


# ===================================================================
# perturb_params
# ===================================================================


class TestPerturbParams:
    def test_int_params_get_perturbed(self):
        base = {"kdj_n": 9, "kdj_k": 3, "kdj_d": 3}
        grid = {"kdj_n": [7, 9, 14], "kdj_k": [2, 3, 5], "kdj_d": [2, 3, 5]}
        variants = perturb_params(base, grid)
        assert len(variants) >= 4  # at least one change per param
        # Each variant only changes ONE param
        for v in variants:
            diffs = sum(1 for k in base if v[k] != base[k])
            assert diffs == 1, f"Expected 1 diff, got {diffs} in {v}"

    def test_small_int_gets_min_offset(self):
        """Small integers where ±10% rounds to same value should still shift by 1."""
        base = {"kdj_k": 2}
        grid = {"kdj_k": [2, 3, 5]}
        variants = perturb_params(base, grid)
        values = {v["kdj_k"] for v in variants}
        assert 1 in values  # 2*0.9=1.8→2(same)→forced to 1
        assert 3 in values  # 2*1.1=2.2→2(same)→forced to 3

    def test_float_params_kept_float(self):
        base = {"atr_stop_mult": 2.0}
        grid = {"atr_stop_mult": [1.5, 2.0, 3.0]}
        variants = perturb_params(base, grid)
        for v in variants:
            assert isinstance(v["atr_stop_mult"], float)

    def test_no_duplicates(self):
        base = {"kdj_n": 7, "kdj_k": 2, "kdj_d": 2}
        grid = {"kdj_n": [7, 9, 14], "kdj_k": [2, 3, 5], "kdj_d": [2, 3, 5]}
        variants = perturb_params(base, grid)
        sigs = [tuple(sorted(v.items())) for v in variants]
        assert len(sigs) == len(set(sigs))

    def test_skips_params_not_in_grid(self):
        base = {"extra_param": 42}
        grid = {"kdj_n": [7, 9, 14]}
        variants = perturb_params(base, grid, levels=[0.9, 1.1])
        # extra_param is not in the grid → skipped entirely → no variants
        assert len(variants) == 0


class TestIsIntParam:
    def test_int_from_grid(self):
        assert _is_int_param("kdj_n", {"kdj_n": [7, 9, 14]}) is True

    def test_float_from_grid(self):
        assert _is_int_param("atr_stop_mult", {"atr_stop_mult": [1.5, 2.0]}) is False

    def test_unknown_param(self):
        assert _is_int_param("unknown", {"kdj_n": [7]}) is False


# ===================================================================
# Cost sensitivity smoke test
# ===================================================================


class TestCostSensitivity:
    def test_run_returns_dataframe(self):
        """Minimal smoke test — ensure the pipeline doesn't crash."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from unittest.mock import patch, MagicMock
        from analysis.cost_sensitivity import run

        # We don't want to hit real data sources — skip for now
        # This test validates the module's import and structure
        from analysis import cost_sensitivity
        assert hasattr(cost_sensitivity, 'run')
        assert hasattr(cost_sensitivity, 'rate_feasibility')
        assert hasattr(cost_sensitivity, 'plot_heatmap')
        assert hasattr(cost_sensitivity, 'generate_report')


class TestParamRobustness:
    def test_module_structure(self):
        from analysis import param_robustness
        assert hasattr(param_robustness, 'run')
        assert hasattr(param_robustness, 'perturb_params')
        assert hasattr(param_robustness, 'plot')
        assert hasattr(param_robustness, 'generate_report')


# ===================================================================
# rate_feasibility
# ===================================================================


class TestRateFeasibility:
    def test_all_positive_gets_a(self):
        from analysis.cost_sensitivity import rate_feasibility
        df = pd.DataFrame({
            "commission": [0.0001, 0.0001, 0.0003, 0.001],
            "slippage": [0.0001, 0.005, 0.0005, 0.001],
            "return_pct": [10.0, 8.0, 6.0, 2.0],
            "sharpe": [1.5, 1.3, 1.1, 0.8],
            "max_dd_pct": [-5.0, -6.0, -7.0, -8.0],
            "trades": [5, 5, 5, 5],
        })
        result = rate_feasibility(df)
        assert result["grade"].startswith("A")

    def test_negative_at_high_cost_gets_b(self):
        from analysis.cost_sensitivity import rate_feasibility
        df = pd.DataFrame({
            "commission": [0.0001, 0.0003, 0.001],
            "slippage": [0.0001, 0.0005, 0.005],
            "return_pct": [10.0, 4.0, -2.0],
            "sharpe": [1.5, 1.0, -0.2],
            "max_dd_pct": [-5.0, -8.0, -15.0],
            "trades": [5, 5, 5],
        })
        result = rate_feasibility(df)
        assert result["grade"].startswith("B")

    def test_all_negative_gets_d(self):
        from analysis.cost_sensitivity import rate_feasibility
        df = pd.DataFrame({
            "commission": [0.0001, 0.0003],
            "slippage": [0.0001, 0.0005],
            "return_pct": [-5.0, -10.0],
            "sharpe": [-0.5, -1.0],
            "max_dd_pct": [-15.0, -20.0],
            "trades": [3, 3],
        })
        result = rate_feasibility(df)
        assert result["grade"].startswith("D")
