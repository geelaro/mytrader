"""traderbridge Dashboard — main orchestrator.

Sidebar + tab routing + calls into sub-modules.
"""

from datetime import date

import streamlit as st

import utils
from utils import get_logger, load_toml
from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP

from dashboard.signals import render_market_state, render_todays_signals, render_risk_light, render_signal_detail, render_position_watch, render_alert_history
from dashboard.single_backtest import render_single_backtest
from dashboard.portfolio_backtest import render_portfolio_backtest, render_monte_carlo
from dashboard.factor_attribution import render_factor_attribution
from dashboard.brinson_attribution import render_brinson_attribution
from dashboard.pnl_breakdown import render_pnl_breakdown
from dashboard.signal_effectiveness import render_signal_effectiveness
from dashboard.kill_switch import render_kill_switch
from dashboard.risk_report import render_risk_report
from dashboard.ops import render_ops
from dashboard.config_editor import render_config_editor
from dashboard.risk_analytics import render_risk_analytics

logger = get_logger("dashboard.main")


def main():
    st.set_page_config(page_title="TraderBridge", layout="wide")
    st.title("TraderBridge — 风险管理 / 决策辅助")

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
    # Market risk light — SPY MA200 + ADX + VIX → 🟢/🟡/🔴 with regime tag
    # (subsumes the old render_market_state panel; the underlying
    # MarketStateClassifier still drives SignalGate / StrategyEnsemble.)
    # -------------------------------------------------------------------

    render_risk_light(config, target_date, provider)

    # -------------------------------------------------------------------
    # Today's signals — summary metrics + per-symbol detail cards
    # (render_todays_signals is preserved as a public callable for backward
    #  compatibility but no longer drawn here — render_signal_detail now
    #  handles both the summary row and the per-symbol expanders.)
    # -------------------------------------------------------------------

    render_signal_detail(config, target_date, provider, cache)
    render_position_watch(config, target_date, provider)

    # -------------------------------------------------------------------
    # Tabs: single backtest | portfolio backtest
    # -------------------------------------------------------------------

    (tab_single, tab_portfolio, tab_factors, tab_brinson, tab_pnl,
     tab_signal_eff, tab_risk, tab_alerts, tab_report, tab_kill,
     tab_config) = st.tabs(
        ["单标的回测", "组合回测", "因子归因", "业绩归因 Brinson",
         "盈亏分析",
         "信号有效性", "风险量化", "风险告警历史", "📑 风险报告",
         "🚨 Kill Switch", "配置管理"]
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

    with tab_brinson:
        render_brinson_attribution(config, target_date, provider)

    with tab_pnl:
        render_pnl_breakdown(config, target_date, provider, cache)

    with tab_signal_eff:
        render_signal_effectiveness(
            config, target_date, backtest_years, provider,
            selected_symbol, selected_strategy,
            strategy_options, symbols,
        )

    with tab_risk:
        render_risk_analytics(config, target_date, provider)

    with tab_alerts:
        render_alert_history(cache)

    with tab_report:
        render_risk_report(config, target_date, provider, cache)

    with tab_kill:
        _render_kill_switch_tab(config, cache)

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


# ---------------------------------------------------------------------------
# Kill Switch wiring — needs a broker + risk_ctrl + notifier
# ---------------------------------------------------------------------------


@st.cache_resource
def _kill_switch_deps(_config_marker: str):
    """Build a MockBroker / RiskController / Notifier for the Kill Switch tab.

    Cached as a singleton so the broker state survives across reruns
    (positions, paused flag, etc.).  The ``_config_marker`` argument is
    just a cache key; passing the watchlist mtime would invalidate on
    config changes if needed.
    """
    from broker import MockBroker
    from live.risk_controller import RiskController
    from utils.risk import RiskLimits
    from utils.notify import Notifier
    from data.cache import CacheManager

    broker = MockBroker(initial_cash=10000)
    cache = CacheManager()
    risk = RiskLimits()
    notifier = Notifier(dry_run=True)
    risk_ctrl = RiskController(risk=risk, cache=cache, broker=broker,
                               notifier=notifier)
    return broker, risk_ctrl, notifier


def _render_kill_switch_tab(config: dict, cache):
    """Wrap render_kill_switch with broker/risk_ctrl/notifier injection."""
    broker, risk_ctrl, notifier = _kill_switch_deps("v1")
    render_kill_switch(broker, risk_ctrl, notifier, cache)


if __name__ == "__main__":
    main()
