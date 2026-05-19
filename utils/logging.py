"""Unified logging — console, rotating file, structured format.

Usage:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("message")
    logger.warning("something", extra={"symbol": "AAPL"})

Named loggers ("live", "daily") → logs/{name}.log + console, isolated.
All other loggers → logs/mytrader.log + console (shared).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

CONSOLE_FORMAT = (
    "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s"
)

_log_initialized = False

# Stored by setup_logging() for use by _setup_named_logger()
_log_dir = "logs"
_max_bytes = 10 * 1024 * 1024
_backup_count = 3

# Track which named loggers have their own handlers
_named_initialized: set = set()

# Logger names that get their own log file (not routed to shared mytrader.log)
_NAMED_LOGGERS = {"live", "daily"}


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
    config_path: str = "watchlist.toml",
):
    """One-time setup — called at process start.

    Sets up root logger with console + shared file (for modules not
    covered by a named logger).  Named loggers ("live", "daily") are
    set up lazily by get_logger().

    If *config_path* exists, reads [log] section from TOML config.
    """
    global _log_initialized, _log_dir, _max_bytes, _backup_count
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
        pass

    _log_dir = log_dir
    _max_bytes = max_bytes
    _backup_count = backup_count

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console — everything
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    # Shared file — captures all loggers that don't have their own file
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    shared_file = RotatingFileHandler(
        log_path / "mytrader.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    shared_file.setLevel(logging.DEBUG)
    shared_file.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(shared_file)

    # Suppress noisy libraries
    for lib in ["matplotlib", "urllib3", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def _setup_named_logger(logger: logging.Logger, name: str):
    """Give *logger* its own console + file handler; stop propagation
    so its messages don't also land in the shared mytrader.log."""
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    log_path = Path(_log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / f"{name}.log",
        maxBytes=_max_bytes,
        backupCount=_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    logger = logging.getLogger(name)
    if name in _NAMED_LOGGERS and name not in _named_initialized:
        _setup_named_logger(logger, name)
        _named_initialized.add(name)
    return logger
