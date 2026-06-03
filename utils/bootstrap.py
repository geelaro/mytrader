"""Explicit runtime bootstrap — call once from each entry point.

Replaces the import-time side effects that used to live at the bottom of
``utils/env.py`` and the corresponding ``import utils  # noqa`` ceremony
in library modules.

Old fragility
-------------
Library modules (engine/trader.py, analysis/*.py) had to do this dance::

    import utils  # noqa: F401 - triggers env setup before matplotlib
    import matplotlib.pyplot as plt

If anyone reordered imports, or imported engine before utils, matplotlib
would use the wrong backend and headless deployments would crash.

New flow
--------
Each ENTRY POINT (live_trader.py, daily.py, dashboard.py, scripts/*.py,
tests/conftest.py) calls :func:`setup_runtime` exactly once at top, BEFORE
any business import.  Library modules no longer need the ``import utils``
ceremony — they can just ``import matplotlib`` directly.

The function is idempotent; calling it twice is a no-op.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SETUP_DONE = False


def setup_runtime() -> None:
    """Idempotent runtime initialisation.

    Effects (in order):
      1. Force matplotlib's non-interactive backend via MPLBACKEND env var
         so any subsequent ``import matplotlib`` uses it (works in headless
         servers / Docker / streamlit).
      2. Reconfigure stdout to UTF-8 on Windows (avoid GBK encoding errors
         on emojis / Chinese characters).
      3. Load .env from the project root.

    Safe to call from anywhere; subsequent calls return immediately.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    # 1. Matplotlib backend — must happen before matplotlib is imported.
    # Using the env var means we don't need to import matplotlib here.
    os.environ.setdefault("MPLBACKEND", "Agg")

    # 2. Windows UTF-8 stdout.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        os.environ["PYTHONIOENCODING"] = "utf-8"

    # 3. Load .env from project root (gitignored).
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    _SETUP_DONE = True


def is_setup_done() -> bool:
    """Inspection helper for tests."""
    return _SETUP_DONE
