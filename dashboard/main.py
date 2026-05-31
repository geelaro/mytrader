"""mytrader Dashboard — main orchestrator.

Sidebar + tab routing + calls into sub-modules.
"""

from datetime import date

import streamlit as st

import utils
from utils import get_logger, load_toml
from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP

from dashboard.signals import render_market_state, render_todays_signals
from dashboard.single_backtest import render_single_backtest
from dashboard.portfolio_backtest import render_portfolio_backtest, render_monte_carlo
from dashboard.factor_attribution import render_factor_attribution
from dashboard.ops import render_ops
from dashboard.config_editor import render_config_editor

logger = get_logger("dashboard.main")


def main():
    st.set_page_config(page_title="Mytrader", layout="wide")
    st.title("Mytrader Dashboard")

    # -------------------------------------------------------------------
    # Sidebar — controls
    # -------------------------------------------------------------------

    st.sidebar.header("控制面板")

    target_date = st.sidebar.date_input("日期", date.today())

    config = load_toml("watchlist.toml")
    symbols = [item["symbol"] for item in config.get("watchlist", [])]

    selected_symbol = st.sidebar.selectbox("标的", symbols)

    strategy_options = list(STRATEGY_MAP.keys())
    selected_strategy = st.sidebar.selectbox("策略", strategy_options, index=2)

    backtest_years = st.sidebar.slider("回测年数", 1, 10, 4)
    allocation_mode = st.sidebar.selectbox("组合分配模式", ["equal", "dynamic_equal"], index=1)
    pf_strategy = st.sidebar.selectbox("组合策略", strategy_options, index=2,
                                       help="所有标的使用统一策略")

    @st.cache_resource
    def get_provider():
        return DataProvider()

    @st.cache_resource  # WARNING: creates a singleton connection shared across all Streamlit sessions.
    # Multi-user deployments should use per-session connections or a connection pool.
    def get_cache():
        return CacheManager()

    provider = get_provider()
    cache = get_cache()

    # -------------------------------------------------------------------
    # Market state
    # -------------------------------------------------------------------

    render_market_state(config, target_date, provider)

    # -------------------------------------------------------------------
    # Today's signals
    # -------------------------------------------------------------------

    render_todays_signals(config, target_date, provider, cache)

    # -------------------------------------------------------------------
    # Tabs: single backtest | portfolio backtest
    # -------------------------------------------------------------------

    tab_single, tab_portfolio, tab_factors, tab_config = st.tabs(
        ["单标的回测", "组合回测", "因子归因", "配置管理"]
    )

    with tab_single:
        render_single_backtest(
            selected_symbol, selected_strategy, backtest_years,
            target_date, provider, cache,
        )

    with tab_portfolio:
        render_portfolio_backtest(
            config, target_date, backtest_years,
            allocation_mode, pf_strategy,
            strategy_options, symbols,
        )

    with tab_factors:
        render_factor_attribution(
            config, target_date, backtest_years,
            allocation_mode, pf_strategy, symbols,
        )

    with tab_config:
        render_config_editor(config)

    # -------------------------------------------------------------------
    # Monte Carlo expander
    # -------------------------------------------------------------------

    render_monte_carlo(strategy_options, symbols, target_date)

    # -------------------------------------------------------------------
    # Ops / sector / trade history
    # -------------------------------------------------------------------

    render_ops(config, cache)


if __name__ == "__main__":
    main()
