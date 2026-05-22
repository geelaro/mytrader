"""Backtest engine — single-symbol, portfolio, and parameter optimization."""

from engine.trader import BacktestEngine, BacktestResult, Trade
from engine.execution import (
    ExecutionConfig,
    ExecutionModel,
    ExecutionPlan,
    ExecutionResult,
    ExecutionStyle,
    ExecutionTiming,
)
from engine.portfolio import (
    PortfolioBacktest,
    PortfolioResult,
    PortfolioTrade,
    Leg,
    DEFAULT_PORTFOLIO,
)
from engine.optimize import grid_search, walk_forward, PARAM_GRIDS
