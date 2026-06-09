"""Historical scenario stress testing.

Replay famous crash / drawdown windows against the current portfolio:
apply current weights to historical returns in the window, aggregate
losses, report per-scenario portfolio return and max drawdown.

This complements Monte Carlo (random shocks) by answering the more
specific question "if 2008 happened *to my current portfolio*, what
would have happened?".  MC gives a distribution; stress gives a story.

Scenarios are calibrated to US equity history (SPY).  For non-US
exposures the dates still apply but the interpretation may shift.

Usage
-----
    from analysis.stress import run_scenarios, SCENARIOS

    # prices: DataFrame indexed by date, columns = symbols
    # weights: {symbol: weight}
    result = run_scenarios(prices, weights)
    for sid, r in result.items():
        print(f"{SCENARIOS[sid]['name']:30s}  {r['return_pct']:+.2f}%  "
              f"MaxDD {r['max_dd_pct']:.2f}%")
"""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Scenario library — calibrated against SPY
# ---------------------------------------------------------------------------

SCENARIOS: dict = {
    "2008_lehman": {
        "name": "2008 雷曼倒闭",
        "start": "2008-09-15",
        "end": "2009-03-09",
        "description": "次贷危机峰值 → 标普 6 个月 -47%",
        "spy_pct": -47.0,
    },
    "2018_q4_selloff": {
        "name": "2018 Q4 抛售",
        "start": "2018-10-01",
        "end": "2018-12-24",
        "description": "加息恐慌 + 中美贸易战, 标普 -19.8%",
        "spy_pct": -19.8,
    },
    "2020_covid": {
        "name": "2020 COVID 闪崩",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "WHO 全球警报 → 触底, 标普 -34% (33 天)",
        "spy_pct": -34.0,
    },
    "2022_rate_hike": {
        "name": "2022 加息周期",
        "start": "2022-01-03",
        "end": "2022-10-12",
        "description": "美联储 8 次加息, 标普 -25%, 纳指 -38%",
        "spy_pct": -25.0,
    },
    "2015_china_devaluation": {
        "name": "2015 8月人民币贬值",
        "start": "2015-08-17",
        "end": "2015-08-25",
        "description": "8.11 汇改 → 全球闪崩, 标普 -11% 一周",
        "spy_pct": -11.0,
    },
}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_scenario(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    scenario_key: str,
) -> dict:
    """Apply ``weights`` to symbol prices over the scenario window.

    Returns a dict::

        {
            "scenario": scenario_key,
            "name": str,                # human-readable
            "start": str, "end": str,
            "return_pct":  float,       # total portfolio return in window (%)
            "max_dd_pct":  float,       # worst drawdown in window (%)
            "n_days":      int,         # bars used
            "by_symbol":   {sym: return_pct},  # per-symbol total return
            "missing_symbols": [str],   # symbols lacking data in window
        }

    Symbols missing prices in the window are excluded from the calc and
    listed in ``missing_symbols``.  If *all* requested symbols are missing,
    returns NaNs and an empty per-symbol dict.
    """
    if scenario_key not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_key}")
    cfg = SCENARIOS[scenario_key]
    return _replay(prices, weights, cfg["start"], cfg["end"],
                   scenario_key=scenario_key, name=cfg["name"])


def replay_custom(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    start: str,
    end: str,
    name: str = "自定义场景",
) -> dict:
    """Same as :func:`replay_scenario` but with caller-provided window."""
    return _replay(prices, weights, start, end,
                   scenario_key="custom", name=name)


def run_scenarios(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    scenarios: Optional[list] = None,
) -> dict:
    """Run all configured scenarios, return dict keyed by scenario_id."""
    keys = scenarios if scenarios is not None else list(SCENARIOS.keys())
    return {k: replay_scenario(prices, weights, k) for k in keys}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _replay(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    start: str,
    end: str,
    scenario_key: str,
    name: str,
) -> dict:
    base = {
        "scenario": scenario_key,
        "name": name,
        "start": start,
        "end": end,
        "return_pct": float("nan"),
        "max_dd_pct": float("nan"),
        "n_days": 0,
        "by_symbol": {},
        "missing_symbols": [],
    }
    if prices is None or prices.empty:
        base["missing_symbols"] = list(weights.keys())
        return base

    window = prices.loc[
        (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    ]
    if window.empty or len(window) < 2:
        base["missing_symbols"] = list(weights.keys())
        return base

    # Symbols with non-NaN data spanning the window
    used: dict = {}
    by_symbol: dict = {}
    missing: list = []
    for sym, w in weights.items():
        if w <= 0:
            continue
        if sym not in window.columns:
            missing.append(sym)
            continue
        s = window[sym].dropna()
        # Need both endpoints — incomplete coverage isn't worth fudging
        if s.empty or len(s) < 2:
            missing.append(sym)
            continue
        used[sym] = w
        by_symbol[sym] = (s.iloc[-1] / s.iloc[0] - 1) * 100  # total return %

    base["missing_symbols"] = missing
    if not used:
        return base

    total = sum(used.values())
    normed = {s: w / total for s, w in used.items()}

    # Daily portfolio return path with reindex/ffill for cross-symbol alignment
    px = window[list(used.keys())].ffill().dropna(how="all")
    rets = px.pct_change(fill_method=None).dropna(how="all").fillna(0)
    w_series = pd.Series(normed)
    pf_returns = (rets * w_series).sum(axis=1)
    if pf_returns.empty:
        return base

    # Prepend the 1.0 starting point so cummax includes the initial
    # capital — otherwise a monotonic drop underreports MaxDD by ~one
    # day's worth of return (the first pct_change row is dropped above).
    equity = pd.concat([pd.Series([1.0]), (1 + pf_returns).cumprod()],
                       ignore_index=True)
    total_return_pct = (equity.iloc[-1] - 1) * 100
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_dd_pct = float(drawdown.min())

    base["return_pct"] = float(total_return_pct)
    base["max_dd_pct"] = max_dd_pct
    base["n_days"] = int(len(pf_returns))
    base["by_symbol"] = {s: float(v) for s, v in by_symbol.items()}
    return base
