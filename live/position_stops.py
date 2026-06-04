"""Backward-compat shim — re-exports from analysis.hypothetical_positions.

This module's logic was moved to ``analysis/hypothetical_positions.py``
because it's pure compute (config + provider → positions list, no broker
calls, no DB writes).  See AGENTS.md "Module boundaries" for the
analysis-layer rule.

Existing callers may keep importing from ``live.position_stops``;
new code should prefer ``from analysis.hypothetical_positions import ...``
or ``from analysis import compute_hypothetical_positions``.
"""

from analysis.hypothetical_positions import (
    _find_open_simulated_trade,
    compute_hypothetical_positions,
)

__all__ = ["compute_hypothetical_positions", "_find_open_simulated_trade"]
