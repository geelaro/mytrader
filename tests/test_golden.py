"""Golden-sample regression tests — must-pass CI gate.

Fixed-seed synthetic data + known-good metric values.  If code changes
shift any of these beyond tolerance, CI fails — prevents silent PnL drift.
"""

import numpy as np
import pytest
from tests.conftest import make_ohlcv
from engine.trader import BacktestEngine
from strategy import EnhancedMACDStrategy, WeeklyMACD_KDJ, TurtleTrading


# Fixed 300-bar dataset, seed=42 (see conftest.py)
@pytest.fixture(scope="module")
def golden_df():
    return make_ohlcv(n_bars=300, seed=42)


# ---------------------------------------------------------------------------
# Golden values — computed 2026-05-22 on seed=42, 300 bars, $10k capital
# (next-bar-open execution model — see engine/trader.py pending_order)
# ---------------------------------------------------------------------------


def test_golden_enhanced_macd_fixed_capital(golden_df):
    s = EnhancedMACDStrategy()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())

    assert r.total_trades == 4, f"trades: {r.total_trades}"
    assert r.total_return_pct == pytest.approx(-3.1103, abs=0.01), f"return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(-1.3825, abs=0.01), f"sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(-3.2145, abs=0.01), f"maxdd: {r.max_drawdown_pct:.4f}"


def test_golden_weekly_macd_kdj_fixed_capital(golden_df):
    s = WeeklyMACD_KDJ()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())

    assert r.total_trades == 2, f"trades: {r.total_trades}"
    assert r.total_return_pct == pytest.approx(-11.7037, abs=0.01), f"return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(-2.4181, abs=0.01), f"sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(-18.4542, abs=0.01), f"maxdd: {r.max_drawdown_pct:.4f}"


def test_golden_turtle_trading_fixed_capital(golden_df):
    s = TurtleTrading()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())

    assert r.total_trades == 1, f"trades: {r.total_trades}"
    assert r.total_return_pct == pytest.approx(-1.0788, abs=0.01), f"return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(-0.7133, abs=0.01), f"sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(-2.0364, abs=0.01), f"maxdd: {r.max_drawdown_pct:.4f}"


# ---------------------------------------------------------------------------
# risk_budget mode — same golden data
# ---------------------------------------------------------------------------

_GOLDEN_RISK_BUDGET = {
    "enhanced_macd":   {"trades": 4, "return": -1.5344, "sharpe": -1.3950, "maxdd": -1.5860},
    "weekly_macd_kdj": {"trades": 2, "return": -1.0825, "sharpe": -2.4509, "maxdd": -1.8213},
    "turtle_trading":  {"trades": 1, "return": -0.8298, "sharpe": -0.7136, "maxdd": -1.5699},
}


@pytest.mark.parametrize("strat_name,cls", [
    ("enhanced_macd", EnhancedMACDStrategy),
    ("weekly_macd_kdj", WeeklyMACD_KDJ),
    ("turtle_trading", TurtleTrading),
])
def test_golden_risk_budget(golden_df, strat_name, cls):
    s = cls()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000, sizing_mode="risk_budget",
                            risk_per_trade=0.01, risk_atr_mult=2.0)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())

    expected = _GOLDEN_RISK_BUDGET[strat_name]
    assert r.total_trades == expected["trades"], f"{strat_name} trades"
    assert r.total_return_pct == pytest.approx(expected["return"], abs=0.01), \
        f"{strat_name} return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(expected["sharpe"], abs=0.01), \
        f"{strat_name} sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(expected["maxdd"], abs=0.01), \
        f"{strat_name} maxdd: {r.max_drawdown_pct:.4f}"


# ---------------------------------------------------------------------------
# Commission / slippage sensitivity — should NOT be golden-invariant
# ---------------------------------------------------------------------------

def test_commission_changes_metrics(golden_df):
    """Sanity: higher commission should reduce returns."""
    s = EnhancedMACDStrategy()
    df = s.calculate_indicators(golden_df)
    e1 = BacktestEngine(initial_capital=10000, commission_rate=0.0003)
    e2 = BacktestEngine(initial_capital=10000, commission_rate=0.003)
    e1.run(s, df)
    e2.run(s, df)
    r1 = e1.get_result(df["Close"].pct_change().dropna())
    r2 = e2.get_result(df["Close"].pct_change().dropna())
    assert r2.total_return_pct <= r1.total_return_pct + 0.001, \
        "higher commission should not increase return"
