"""Unified runtime configuration — single import point.

Usage:
    from config import config

    print(config.risk.max_position_pct)     # dot-access
    print(config.feishu.app_id)             # from env vars
    watchlist = config.watchlist_data        # lazy-loads watchlist.toml

Load order: DEFAULT_CONFIG → config.yaml (optional) → env vars (secrets).
watchlist.toml is kept as the canonical source for symbols/strategy/risk;
config.yaml can override runtime params like log level, trading hours etc.
"""

import copy
import os
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Defaults — one place for all runtime parameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "trading": {
        "market_open": "21:30",
        "market_close": "04:00",
        "daemon_interval_minutes": 5,
        "lookback_years": 3,
    },
    "risk": {
        "max_position_pct": 0.30,
        "max_total_exposure_pct": 0.80,
        "max_daily_loss_pct": 0.05,
        "min_order_value": 500.0,
        "max_slippage_pct": 0.02,
        "max_consecutive_losses": 3,
        "max_daily_trades": 5,
        "base_risk_pct": 0.02,
        "vol_sensitivity": 5.0,
        "min_vol_scalar": 0.3,
    },
    "log": {
        "level": "INFO",
        "file_dir": "logs",
        "max_bytes": 10485760,
        "backup_count": 3,
    },
    "notification": {
        "queue_maxsize": 1000,
    },
    "broker_futu": {
        "host": "127.0.0.1",
        "port": 11111,
        "initial_cash": 10000,
    },
    "data": {
        "cache_db": "trading_data.db",
        "request_timeout": 10,
    },
    "feishu": {
        "webhook": "",
        "app_id": "",
        "app_secret": "",
        "chat_id": "",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _Section:
    """Dot-access wrapper around a config section dict."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"no config key '{name}'") from None

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self):
        return f"<Section {self._data}>"


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class RuntimeConfig:
    """Unified runtime configuration singleton.

    Properties
    ----------
    trading, risk, log, notification, broker_futu, data, feishu : _Section
        Dot-access config sections.
    watchlist_data : dict
        Lazy-loaded contents of watchlist.toml.
    raw : dict
        Deep-copy of the full internal data dict (for debugging).
    """

    def __init__(self):
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        self._load_yaml()
        self._load_env_secrets()
        self._watchlist: Optional[dict] = None

        # Build section accessors
        self.trading = _Section(self._data["trading"])
        self.risk = _Section(self._data["risk"])
        self.log = _Section(self._data["log"])
        self.notification = _Section(self._data["notification"])
        self.broker_futu = _Section(self._data["broker_futu"])
        self.data = _Section(self._data["data"])
        self.feishu = _Section(self._data["feishu"])

    # ---- YAML loading ----

    def _load_yaml(self):
        yaml_path = PROJECT_ROOT / "config.yaml"
        if not yaml_path.exists():
            return
        try:
            import yaml as _yaml
        except ImportError:
            return
        try:
            with open(yaml_path, "rt", encoding="utf-8") as f:
                user_data = _yaml.safe_load(f) or {}
        except Exception:
            return
        if not isinstance(user_data, dict):
            return
        for section, values in user_data.items():
            if section in self._data and isinstance(values, dict):
                self._data[section] = _deep_merge(self._data[section], values)

    # ---- Environment variable overlay ----

    def _load_env_secrets(self):
        f = self._data["feishu"]
        f["webhook"] = os.getenv("FEISHU_WEBHOOK", f["webhook"])
        f["app_id"] = os.getenv("FEISHU_APP_ID", f["app_id"])
        f["app_secret"] = os.getenv("FEISHU_APP_SECRET", f["app_secret"])
        f["chat_id"] = os.getenv("FEISHU_CHAT_ID", f["chat_id"])

        b = self._data["broker_futu"]
        b["host"] = os.getenv("FUTU_HOST", b["host"])
        port = os.getenv("FUTU_PORT")
        if port:
            b["port"] = int(port)
        cash = os.getenv("FUTU_INITIAL_CASH")
        if cash:
            b["initial_cash"] = int(cash)

    # ---- watchlist.toml lazy-load ----

    @property
    def watchlist_data(self) -> dict:
        if self._watchlist is None:
            from utils.env import load_toml

            self._watchlist = load_toml(str(PROJECT_ROOT / "watchlist.toml"))
        return self._watchlist

    # ----

    @property
    def raw(self) -> dict:
        return copy.deepcopy(self._data)


# ---- Module-level singleton ----

config = RuntimeConfig()
