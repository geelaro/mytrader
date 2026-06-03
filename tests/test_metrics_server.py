"""Tests for utils/metrics_server.py — Prometheus text format renderer."""

import pytest

from utils import metrics_server


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics_server.reset_for_tests()
    yield
    metrics_server.reset_for_tests()


class TestCounters:
    def test_simple_counter(self):
        metrics_server.incr("orders_total")
        metrics_server.incr("orders_total")
        text = metrics_server.render()
        assert "# TYPE orders_total counter" in text
        assert "orders_total 2" in text

    def test_labeled_counter(self):
        metrics_server.incr("orders_total", {"side": "BUY", "status": "FILLED"})
        metrics_server.incr("orders_total", {"side": "BUY", "status": "FILLED"})
        metrics_server.incr("orders_total", {"side": "SELL", "status": "REJECTED"})
        text = metrics_server.render()
        assert 'orders_total{side="BUY",status="FILLED"} 2' in text
        assert 'orders_total{side="SELL",status="REJECTED"} 1' in text

    def test_label_order_irrelevant(self):
        """Same labels in different insertion order are the same series."""
        metrics_server.incr("test", {"a": "1", "b": "2"})
        metrics_server.incr("test", {"b": "2", "a": "1"})
        text = metrics_server.render()
        # Should collapse to a single series with count 2
        assert text.count('test{a="1",b="2"} 2') == 1

    def test_value_arg(self):
        metrics_server.incr("test", value=5)
        metrics_server.incr("test", value=3)
        text = metrics_server.render()
        assert "test 8" in text


class TestGauges:
    def test_gauge_set_overrides(self):
        metrics_server.set_gauge("paused", 1)
        metrics_server.set_gauge("paused", 0)
        text = metrics_server.render()
        assert "# TYPE paused gauge" in text
        assert "paused 0.0" in text

    def test_labeled_gauge(self):
        metrics_server.set_gauge("queue_depth", 5, {"worker": "notify"})
        text = metrics_server.render()
        assert 'queue_depth{worker="notify"} 5.0' in text


class TestEscaping:
    def test_label_value_quote_escaped(self):
        metrics_server.incr("t", {"k": 'has"quote'})
        text = metrics_server.render()
        assert 'has\\"quote' in text

    def test_label_value_backslash_escaped(self):
        metrics_server.incr("t", {"k": r"a\b"})
        text = metrics_server.render()
        assert 'a\\\\b' in text


class TestRenderFormat:
    def test_no_metrics_returns_just_newline(self):
        assert metrics_server.render() == "\n"

    def test_mixed_counter_and_gauge(self):
        metrics_server.incr("c1")
        metrics_server.set_gauge("g1", 42)
        text = metrics_server.render()
        assert "# TYPE c1 counter" in text
        assert "# TYPE g1 gauge" in text
        assert "c1 1" in text
        assert "g1 42.0" in text
