"""Tests for utils/logging.py JsonFormatter."""

import json
import logging
from io import StringIO
from utils.logging import JsonFormatter


class TestJsonFormatter:
    def test_basic_format(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "path", 10, "hello world", (), None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "hello world"
        assert "ts" in data

    def test_extra_fields_merged(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            "live", logging.WARNING, "path", 20, "risk event", (), None,
        )
        record.symbol = "AAPL"
        record.event = "circuit_break"
        record.detail = "drawdown 30%"
        output = fmt.format(record)
        data = json.loads(output)
        assert data["symbol"] == "AAPL"
        assert data["event"] == "circuit_break"
        assert data["detail"] == "drawdown 30%"

    def test_exception_included(self):
        fmt = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                "live", logging.ERROR, "path", 30, "failed", (),
                sys.exc_info(),
            )
            output = fmt.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "test error" in data["exception"]

    def test_empty_extra_ignored(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            "live", logging.INFO, "path", 10, "msg", (), None,
        )
        record.symbol = ""
        record.detail = ""
        output = fmt.format(record)
        data = json.loads(output)
        assert "symbol" not in data
        assert "detail" not in data
