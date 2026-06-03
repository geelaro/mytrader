"""TOML helpers — load/save watchlist.toml and friends.

Runtime bootstrap (matplotlib backend / Windows encoding / dotenv) has
moved to :mod:`utils.bootstrap`.  Entry-point scripts now call
``setup_runtime()`` explicitly instead of relying on import-time side
effects.  See ``utils/bootstrap.py`` for the rationale.
"""


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


