"""Decision logger — snapshot risk context whenever the user/system acts.

Why
---
The dashboard shows what the *current* risk looks like.  Looking back, you
want to ask: "in the last 90 days, when I made decisions under RED light
versus GREEN, what were the outcomes?"  That requires capturing the risk
context **at the moment of every decision**, not derived after the fact.

What counts as a "decision"?
----------------------------
- ``trade_buy`` / ``trade_sell`` — broker.submit_order succeeded
- ``kill_switch`` — emergency liquidation triggered
- ``rebalance`` — manual rebalance applied from dashboard What-If
- ``manual_override`` — user changed config to enable/disable a strategy
- ``signal_ignored`` — strategy emitted a signal that wasn't acted on
  (e.g., paused, exposure cap, manual hold) — optional

Persistence
-----------
:meth:`data.cache.StateStore.record_decision` writes one row per call to
the ``decision_history`` table; risk-snapshot fields land in dedicated
columns so SQL filters work without JSON parsing.

Usage
-----
    logger = DecisionLogger(cache, context_resolver=resolver_fn)
    logger.log("trade_buy", symbol="NVDA",
               reason="weekly_macd_kdj long signal",
               payload={"order_id": "X", "qty": 10, "price": 145.2})

The ``context_resolver`` is a zero-arg callable returning a dict of the
current risk picture (risk_light, vix, portfolio_value, hhi, var_95,
drawdown_pct, ...).  Keeping resolution lazy means a 5-second-late
``submit_order`` still captures the live snapshot, not a stale one.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# Convention — keep the strings stable, callers and SQL filters depend on them.
DECISION_TRADE_BUY      = "trade_buy"
DECISION_TRADE_SELL     = "trade_sell"
DECISION_KILL_SWITCH    = "kill_switch"
DECISION_REBALANCE      = "rebalance"
DECISION_MANUAL_OVERRIDE = "manual_override"
DECISION_SIGNAL_IGNORED = "signal_ignored"


class DecisionLogger:
    """Snapshot + persist one row of decision context per call.

    Construction
    ------------
    cache : ``StateStore`` (or ``CacheManager``) — anything exposing
        ``record_decision(decision_type, symbol, reason, context, payload)``.
    context_resolver : Optional[Callable[[], dict]]
        Zero-arg function returning the current risk-context dict.
        Called *lazily* on each :meth:`log` so each row captures the
        risk picture at decision time, not at logger construction.
        If omitted, callers must pass ``context=`` to every log call.
    """

    def __init__(
        self,
        cache,
        context_resolver: Optional[Callable[[], dict]] = None,
    ):
        self.cache = cache
        self.context_resolver = context_resolver

    def log(
        self,
        decision_type: str,
        symbol: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[dict] = None,
        payload: Optional[dict] = None,
    ) -> int:
        """Record one decision.  Best-effort — never raises into caller.

        If ``context`` is omitted and a ``context_resolver`` was provided,
        the resolver is invoked.  Any exception in the resolver is
        swallowed and the decision is logged without risk context (so the
        audit trail still survives).
        """
        if context is None and self.context_resolver is not None:
            try:
                context = self.context_resolver()
            except Exception:
                logger.exception(
                    "DecisionLogger: context_resolver failed for %s on %s",
                    decision_type, symbol or "*",
                )
                context = None
        try:
            return self.cache.record_decision(
                decision_type=decision_type,
                symbol=symbol,
                reason=reason,
                context=context,
                payload=payload,
            )
        except Exception:
            logger.exception(
                "DecisionLogger: failed to persist %s on %s",
                decision_type, symbol or "*",
            )
            return 0
