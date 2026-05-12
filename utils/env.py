"""Environment bootstrap — call once at module import time.

Handles platform quirks, config loading, and matplotlib setup.
"""

import os
import sys
from pathlib import Path


def _fix_windows_encoding():
    """Fix GBK encoding issues on Windows terminals."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        os.environ["PYTHONIOENCODING"] = "utf-8"


def _setup_matplotlib():
    """Set non-interactive backend before any pyplot import."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass


def _load_dotenv():
    """Load .env from project root."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass


def load_toml(path: str) -> dict:
    """Load a TOML config file. Works on Python 3.10+."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


# Run on import
_fix_windows_encoding()
_setup_matplotlib()
_load_dotenv()
