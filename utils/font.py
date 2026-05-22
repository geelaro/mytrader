"""Matplotlib Chinese font bootstrap — single source for all chart modules."""

import matplotlib


def setup_chinese_font():
    """Try common CJK fonts; first found wins."""
    for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [font]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
