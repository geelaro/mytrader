"""Unified logging — console, rotating file, structured format.

Usage:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("message")
    logger.warning("something", extra={"symbol": "AAPL"})
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Escape sequences removed to keep it simple — file handlers are pure text
CONSOLE_FORMAT = (
    "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s"
)

_log_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
    config_path: str = "watchlist.toml",
):
    """One-time setup — called at process start.

    If *config_path* exists, reads [log] section from TOML config.
    """
    global _log_initialized
    if _log_initialized:
        return
    _log_initialized = True

    # Try reading from config file
    try:
        from utils.env import load_toml
        cfg = load_toml(config_path)
        log_cfg = cfg.get("log", {})
        level = log_cfg.get("level", level)
        log_dir = log_cfg.get("file_dir", log_dir)
        max_bytes = log_cfg.get("max_bytes", max_bytes)
        backup_count = log_cfg.get("backup_count", backup_count)
    except Exception:
        pass  # config file doesn't exist or has no [log] section

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    # File
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / "mytrader.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(file_handler)

    # Suppress noisy libraries
    for lib in ["matplotlib", "urllib3", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    setup_logging()  # safe to call multiple times
    return logging.getLogger(name)
