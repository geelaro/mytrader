"""Golden-sample regression tests — must-pass CI gate.

Fixed-seed synthetic data + known-good metric values.  If code changes
shift any of these beyond tolerance, CI fails — prevents silent PnL drift.
"""

import numpy as np
import pytest
from tests.conftest import make_ohlcv
from engine.trader import BacktestEngine
from strategy import (
    EnhancedMACDStrategy, WeeklyMACD_KDJ, TurtleTrading,
    ATRBreakout, DonchianBreakout, DailyMACD_KDJ, WeeklyMACD,
    MACDKDJStrategy,
)


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
    assert r.total_return_pct == pytest.approx(-2.5970, abs=0.01), f"return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(-1.1506, abs=0.01), f"sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(-3.1977, abs=0.01), f"maxdd: {r.max_drawdown_pct:.4f}"


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

    assert r.total_trades == 5, f"trades: {r.total_trades}"
    assert r.total_return_pct == pytest.approx(-7.5886, abs=0.01), f"return: {r.total_return_pct:.4f}"
    assert r.sharpe_ratio == pytest.approx(-2.7013, abs=0.01), f"sharpe: {r.sharpe_ratio:.4f}"
    assert r.max_drawdown_pct == pytest.approx(-8.2342, abs=0.01), f"maxdd: {r.max_drawdown_pct:.4f}"


def test_golden_atr_breakout_fixed_capital(golden_df):
    s = ATRBreakout()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())
    assert r.total_trades == 4
    assert r.total_return_pct == pytest.approx(-4.3525, abs=0.01)
    assert r.sharpe_ratio == pytest.approx(-1.0176, abs=0.01)
    assert r.max_drawdown_pct == pytest.approx(-6.5464, abs=0.01)


def test_golden_donchian_breakout_fixed_capital(golden_df):
    s = DonchianBreakout()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())
    assert r.total_trades == 6
    assert r.total_return_pct == pytest.approx(-1.2705, abs=0.01)
    assert r.sharpe_ratio == pytest.approx(-0.2918, abs=0.01)
    assert r.max_drawdown_pct == pytest.approx(-3.8094, abs=0.01)


def test_golden_daily_macd_kdj_fixed_capital(golden_df):
    s = DailyMACD_KDJ()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())
    assert r.total_trades == 13
    assert r.total_return_pct == pytest.approx(-4.0445, abs=0.01)
    assert r.sharpe_ratio == pytest.approx(-0.7408, abs=0.01)
    assert r.max_drawdown_pct == pytest.approx(-7.7168, abs=0.01)


def test_golden_weekly_macd_fixed_capital(golden_df):
    s = WeeklyMACD()
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())
    assert r.total_trades == 2
    assert r.total_return_pct == pytest.approx(-15.9225, abs=0.01)
    assert r.sharpe_ratio == pytest.approx(-3.5648, abs=0.01)
    assert r.max_drawdown_pct == pytest.approx(-21.5758, abs=0.01)


def test_golden_macd_kdj_merged_fixed_capital(golden_df):
    s = MACDKDJStrategy(freq="D", use_atr_stop=True)
    df = s.calculate_indicators(golden_df)
    engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change().dropna())
    assert r.total_trades == 13
    assert r.total_return_pct == pytest.approx(-4.0445, abs=0.01)


# ---------------------------------------------------------------------------
# risk_budget mode — same golden data
# ---------------------------------------------------------------------------

_GOLDEN_RISK_BUDGET = {
    "enhanced_macd":   {"trades": 4, "return": -1.2794, "sharpe": -1.1579, "maxdd": -1.5769},
    "weekly_macd_kdj": {"trades": 2, "return": -1.0825, "sharpe": -2.4509, "maxdd": -1.8213},
    "turtle_trading":  {"trades": 5, "return": -5.7028, "sharpe": -2.7027, "maxdd": -6.1994},
    "atr_breakout":    {"trades": 4, "return": -3.2669, "sharpe": -1.0239, "maxdd": -4.9075},
    "donchian_breakout": {"trades": 6, "return": -0.9382, "sharpe": -0.2965, "maxdd": -2.8186},
    "daily_macd_kdj":  {"trades": 13, "return": -3.0836, "sharpe": -0.7595, "maxdd": -5.8822},
    "weekly_macd":     {"trades": 2, "return": -1.6163, "sharpe": -3.4692, "maxdd": -2.3275},
    "macd_kdj":        {"trades": 13, "return": -3.0836, "sharpe": -0.7595, "maxdd": -5.8822},
}

_MACD_KDJ_FACTORY = lambda: MACDKDJStrategy(freq="D", use_atr_stop=True)

@pytest.mark.parametrize("strat_name,cls_or_factory", [
    ("enhanced_macd", EnhancedMACDStrategy),
    ("weekly_macd_kdj", WeeklyMACD_KDJ),
    ("turtle_trading", TurtleTrading),
    ("atr_breakout", ATRBreakout),
    ("donchian_breakout", DonchianBreakout),
    ("daily_macd_kdj", DailyMACD_KDJ),
    ("weekly_macd", WeeklyMACD),
    ("macd_kdj", _MACD_KDJ_FACTORY),
])
def test_golden_risk_budget(golden_df, strat_name, cls_or_factory):
    s = cls_or_factory() if callable(cls_or_factory) else cls_or_factory()
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
