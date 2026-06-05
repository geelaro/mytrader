"""Analysis package — risk + performance + reports.

Organised into logical sub-domains for navigation.  Physical layout is
still flat (one ``.py`` per concern) to keep imports stable; this
``__init__.py`` provides categorised re-exports as the canonical map.

Sub-domains
-----------
Risk measurement (Ex-ante)
    var / evt / stress / concentration / correlation_analysis /
    risk_decomposition / what_if / monte_carlo

Risk measurement (Ex-post / performance attribution)
    risk_metrics / drawdown / factor_attribution / factor_returns /
    brinson / rolling_alpha / forward_return / pnl_breakdown

Risk monitoring + reports
    risk_monitor / risk_report

Strategy research
    cost_sensitivity / param_robustness

When adding a new analysis module:
1. Drop it into ``analysis/<name>.py`` (flat layout).
2. Add a re-export below under the matching sub-domain section.
3. Add an entry to ``__all__``.
"""

# ---------------------------------------------------------------------------
# Risk measurement — Ex-ante (forward-looking loss / scenario)
# ---------------------------------------------------------------------------

from .var import (
    historical_var, parametric_var, conditional_var,
    portfolio_returns, var_summary,
)
from .garch import (
    ewma_volatility, fit_gjr_garch, forecast_volatility,
    forward_var, forward_var_summary,
)
from .var_coverage import coverage_backtest
from .evt import fit_gpd, evt_var, evt_es, evt_summary
from .stress import SCENARIOS, replay_scenario, replay_custom, run_scenarios
from .concentration import (
    hhi, hhi_label, effective_n, top_n_weight,
    sector_exposure, sector_hhi, correlation_hhi,
    concentration_summary,
)
from .correlation_analysis import (
    correlation_matrix, max_pairwise_correlation,
    effective_bets, correlation_clusters, correlation_summary,
)
from .risk_decomposition import (
    parametric_portfolio_var, marginal_var, component_var,
    risk_contribution_pct, risk_parity_weights,
    inverse_volatility_weights, risk_decomposition_summary,
)
from .what_if import apply_rebalance, compare_portfolios
from .monte_carlo import run as monte_carlo_run

# ---------------------------------------------------------------------------
# Risk measurement — Ex-post + performance attribution
# ---------------------------------------------------------------------------

from .risk_metrics import (
    sortino_ratio, calmar_ratio, mar_ratio, omega_ratio,
    information_ratio, pain_index, pain_ratio, risk_adjusted_summary,
)
from .drawdown import (
    underwater_curve, drawdown_episodes,
    time_to_recover_stats, drawdown_summary,
)
from .drawdown_attribution import (
    attribute_drawdown, attribute_active_drawdown,
    trade_overlap_attribution, historical_drawdown_attribution,
)
from .factor_attribution import FactorAttribution
from .factor_returns import FactorReturns
from .brinson import (
    SECTOR_ETF, brinson_attribution,
    portfolio_sector_breakdown, compute_period_returns,
)
from .rolling_alpha import run as rolling_alpha_run
from .forward_return import compute_forward_returns
from .pnl_breakdown import (
    resolve_period, realized_pnl_summary,
    unrealized_pnl_summary, pnl_summary,
)
from .decision_attribution import (
    join_decision_pnl, hit_rate_by_group, decision_attribution_summary,
)
from .hypothetical_positions import compute_hypothetical_positions
from .proximity import proximity_warnings, proximity_summary

# ---------------------------------------------------------------------------
# Monitoring + reports
# ---------------------------------------------------------------------------

from .risk_monitor import RiskLevel, RiskState, compute_risk_state
from .risk_report import RiskReport, Section

# ---------------------------------------------------------------------------
# Strategy research
# ---------------------------------------------------------------------------

from .cost_sensitivity import run as cost_sensitivity_run
from .param_robustness import run as param_robustness_run


__all__ = [
    # Ex-ante risk
    "historical_var", "parametric_var", "conditional_var",
    "portfolio_returns", "var_summary",
    "ewma_volatility", "fit_gjr_garch", "forecast_volatility",
    "forward_var", "forward_var_summary",
    "coverage_backtest",
    "fit_gpd", "evt_var", "evt_es", "evt_summary",
    "SCENARIOS", "replay_scenario", "replay_custom", "run_scenarios",
    "hhi", "hhi_label", "effective_n", "top_n_weight",
    "sector_exposure", "sector_hhi", "correlation_hhi",
    "concentration_summary",
    "correlation_matrix", "max_pairwise_correlation",
    "effective_bets", "correlation_clusters", "correlation_summary",
    "parametric_portfolio_var", "marginal_var", "component_var",
    "risk_contribution_pct", "risk_parity_weights",
    "inverse_volatility_weights", "risk_decomposition_summary",
    "apply_rebalance", "compare_portfolios",
    "monte_carlo_run",
    # Ex-post / performance attribution
    "sortino_ratio", "calmar_ratio", "mar_ratio", "omega_ratio",
    "information_ratio", "pain_index", "pain_ratio", "risk_adjusted_summary",
    "underwater_curve", "drawdown_episodes",
    "time_to_recover_stats", "drawdown_summary",
    "attribute_drawdown", "attribute_active_drawdown",
    "trade_overlap_attribution", "historical_drawdown_attribution",
    "FactorAttribution", "FactorReturns",
    "SECTOR_ETF", "brinson_attribution",
    "portfolio_sector_breakdown", "compute_period_returns",
    "rolling_alpha_run",
    "compute_forward_returns",
    "resolve_period", "realized_pnl_summary",
    "unrealized_pnl_summary", "pnl_summary",
    "join_decision_pnl", "hit_rate_by_group",
    "decision_attribution_summary",
    "compute_hypothetical_positions",
    "proximity_warnings", "proximity_summary",
    # Monitoring + reports
    "RiskLevel", "RiskState", "compute_risk_state",
    "RiskReport", "Section",
    # Strategy research
    "cost_sensitivity_run", "param_robustness_run",
]
