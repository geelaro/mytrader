"""Tests for config.py — RuntimeConfig, _Section, _deep_merge."""

import os
from unittest.mock import patch

import pytest
from config import RuntimeConfig, _deep_merge, PROJECT_ROOT


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 99})
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_merge(self):
        base = {"log": {"level": "INFO", "file_dir": "logs"}}
        override = {"log": {"level": "DEBUG"}}
        result = _deep_merge(base, override)
        assert result["log"]["level"] == "DEBUG"
        assert result["log"]["file_dir"] == "logs"  # preserved

    def test_new_key_added(self):
        base = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result["b"] == 2

    def test_does_not_mutate_base(self):
        base = {"log": {"level": "INFO"}}
        _deep_merge(base, {"log": {"level": "DEBUG"}})
        assert base["log"]["level"] == "INFO"


class TestSection:
    def test_dot_access(self):
        from config import _Section
        s = _Section({"key": 42, "name": "test"})
        assert s.key == 42
        assert s.name == "test"

    def test_attribute_error_on_missing(self):
        from config import _Section
        s = _Section({"a": 1})
        with pytest.raises(AttributeError):
            _ = s.no_such_key

    def test_get_with_default(self):
        from config import _Section
        s = _Section({"a": 1})
        assert s.get("a") == 1
        assert s.get("missing", "fallback") == "fallback"
        assert s.get("missing") is None


class TestRuntimeConfig:
    def test_defaults_loaded(self):
        cfg = RuntimeConfig()
        assert cfg.risk.max_position_pct == 0.30
        assert cfg.trading.market_open == "21:30"
        assert cfg.notification.queue_maxsize == 1000
        assert cfg.data.cache_db == "trading_data.db"

    def test_feishu_defaults_empty(self):
        cfg = RuntimeConfig()
        assert cfg.feishu.webhook == ""

    def test_env_vars_overlay_feishu(self, monkeypatch):
        monkeypatch.setenv("FEISHU_WEBHOOK", "https://hook.test/x")
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
        monkeypatch.setenv("FEISHU_APP_SECRET", "sec_test")
        monkeypatch.setenv("FEISHU_CHAT_ID", "oc_test")

        cfg = RuntimeConfig()
        assert cfg.feishu.webhook == "https://hook.test/x"
        assert cfg.feishu.app_id == "cli_test"
        assert cfg.feishu.app_secret == "sec_test"
        assert cfg.feishu.chat_id == "oc_test"

    def test_env_vars_overlay_futu(self, monkeypatch):
        monkeypatch.setenv("FUTU_HOST", "10.0.0.1")
        monkeypatch.setenv("FUTU_PORT", "22222")
        monkeypatch.setenv("FUTU_INITIAL_CASH", "50000")

        cfg = RuntimeConfig()
        assert cfg.broker_futu.host == "10.0.0.1"
        assert cfg.broker_futu.port == 22222
        assert cfg.broker_futu.initial_cash == 50000

    def test_env_vars_partial_no_crash(self, monkeypatch):
        monkeypatch.setenv("FUTU_HOST", "10.0.0.1")
        # no FUTU_PORT set — should keep default
        cfg = RuntimeConfig()
        assert cfg.broker_futu.host == "10.0.0.1"
        assert cfg.broker_futu.port == 11111  # default

    def test_raw_returns_deepcopy(self):
        cfg = RuntimeConfig()
        r1 = cfg.raw
        r2 = cfg.raw
        assert r1 == r2
        assert r1 is not r2

    def test_watchlist_data_lazy_load(self, monkeypatch):
        # Ensure we're in the project root so load_toml finds watchlist.toml
        monkeypatch.chdir(PROJECT_ROOT)
        cfg = RuntimeConfig()
        data = cfg.watchlist_data
        assert "watchlist" in data
        assert isinstance(data["watchlist"], list)
        assert len(data["watchlist"]) > 0

    def test_watchlist_data_cached(self, monkeypatch):
        monkeypatch.chdir(PROJECT_ROOT)
        cfg = RuntimeConfig()
        d1 = cfg.watchlist_data
        d2 = cfg.watchlist_data
        assert d1 is d2  # same object, cached
