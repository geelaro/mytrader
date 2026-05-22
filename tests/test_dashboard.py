"""Smoke tests — verify dashboard modules can be imported and expose expected render functions."""

import pytest


def test_import_dashboard_main():
    from dashboard.main import main, render_todays_signals, render_market_state
    assert callable(main)
    assert callable(render_todays_signals)
    assert callable(render_market_state)


def test_import_dashboard_signals():
    from dashboard.signals import render_market_state, render_todays_signals
    assert callable(render_market_state)
    assert callable(render_todays_signals)


def test_import_dashboard_single_backtest():
    from dashboard.single_backtest import render_single_backtest
    assert callable(render_single_backtest)


def test_import_dashboard_portfolio_backtest():
    from dashboard.portfolio_backtest import render_portfolio_backtest, render_monte_carlo
    assert callable(render_portfolio_backtest)
    assert callable(render_monte_carlo)


def test_import_dashboard_ops():
    from dashboard.ops import render_ops
    assert callable(render_ops)
