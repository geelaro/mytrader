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


def save_toml(path: str, data: dict):
    """Write a dict as TOML (hand-rolled, no external dependency)."""
    lines = []
    # Sections that use [[array]] syntax
    array_sections = {"watchlist"}

    for section, content in data.items():
        if isinstance(content, list):
            for item in content:
                lines.append(f"\n[[{section}]]")
                for k, v in item.items():
                    lines.append(f"{k} = {_toml_value(v)}")
        elif isinstance(content, dict):
            first = True
            for k, v in content.items():
                if isinstance(v, dict):
                    lines.append(f"\n[{section}.{k}]")
                    for sk, sv in v.items():
                        lines.append(f"{sk} = {_toml_value(sv)}")
                else:
                    if first:
                        lines.append(f"\n[{section}]")
                        first = False
                    lines.append(f"{k} = {_toml_value(v)}")
        else:
            lines.append(f"{section} = {_toml_value(content)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).lstrip() + "\n")


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return f'"{v}"'


# Run on import
_fix_windows_encoding()
_setup_matplotlib()
_load_dotenv()
