"""Prometheus / OpenMetrics text endpoint for daemon observability.

Why
---
``/health`` answers "is daemon alive?".  ``/metrics`` answers everything
else: how many orders, how often the circuit breaker fires, how many
Yahoo 429s, how big the realtime VIX cache hit rate is, etc.  Grafana +
Prometheus consume this and can alert on derivatives.

This is a **dependency-free** implementation that emits OpenMetrics text
format directly — no need to add ``prometheus-client``.  Counters are
plain int dicts protected by a module-level lock.

Counters / gauges currently exposed
-----------------------------------
- ``orders_submitted_total{side, status}`` — every order flowing through
  OrderManager
- ``risk_circuit_break_total{reason}`` — RiskController pause events
- ``risk_alert_fired_total{type}`` — RiskAlerter triggers (risk_light /
  vix_spike / position_stop)
- ``kill_switch_triggered_total`` — Kill Switch fires
- ``data_source_failure_total{source, status}`` — provider fallbacks
- ``realtime_vix_cache_hits_total`` / ``realtime_vix_cache_misses_total``
- ``daemon_tick_total`` — completed daemon cycles
- ``daemon_paused`` gauge (1 / 0)

Add new metrics
---------------
Just call :func:`incr("name", labels=None)` or :func:`set_gauge("name",
value, labels=None)` from anywhere — the metric appears in /metrics
automatically.  No registration step.

Threading
---------
Single-process daemon, multiple threads (notify worker + main tick).
Lock-protected dict updates are safe.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# {metric_name: {label_tuple: int}}.  label_tuple is sorted ((k, v), ...) so
# the same labels in different insertion orders hash equal.
_COUNTERS: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
_GAUGES: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))


def _label_key(labels: Optional[dict]) -> tuple:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def incr(name: str, labels: Optional[dict] = None, value: int = 1) -> None:
    """Increment a counter.  Thread-safe."""
    key = _label_key(labels)
    with _LOCK:
        _COUNTERS[name][key] += value


def set_gauge(name: str, value: float, labels: Optional[dict] = None) -> None:
    """Set a gauge.  Thread-safe."""
    key = _label_key(labels)
    with _LOCK:
        _GAUGES[name][key] = float(value)


def reset_for_tests() -> None:
    """Clear all metrics — called by test fixtures only."""
    with _LOCK:
        _COUNTERS.clear()
        _GAUGES.clear()


def render() -> str:
    """Render the current state as Prometheus text format."""
    lines: list[str] = []
    with _LOCK:
        for name, by_label in sorted(_COUNTERS.items()):
            lines.append(f"# TYPE {name} counter")
            for label_key, value in sorted(by_label.items()):
                lines.append(f"{name}{_render_labels(label_key)} {value}")
        for name, by_label in sorted(_GAUGES.items()):
            lines.append(f"# TYPE {name} gauge")
            for label_key, value in sorted(by_label.items()):
                lines.append(f"{name}{_render_labels(label_key)} {value}")
    return "\n".join(lines) + "\n"


def _render_labels(label_key: tuple) -> str:
    if not label_key:
        return ""
    parts = ",".join(f'{k}="{_escape(v)}"' for k, v in label_key)
    return "{" + parts + "}"


def _escape(v: object) -> str:
    s = str(v)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
