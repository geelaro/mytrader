"""Historical analog finder — match today's drop to past events with similar setup.

Use case
--------
When you ask "QQQ is down 3% today on hot NFP, what happened last time?",
this module:

1. Scans the price history for all days where the daily drop hit a
   threshold (e.g., ≤ −3%).
2. Tags each event with its likely macro driver (NFP / CPI / FOMC /
   OTHER) via :mod:`analysis.macro_calendar`.
3. Computes forward 3/5/10/20/60-day returns from each event's close.
4. Ranks past events by similarity to today's setup (same macro tag
   first, then closest drop magnitude).

API
---
- :func:`find_drop_events`     — enumerate all drops with forward returns
- :func:`analog_summary_by_tag` — aggregate stats per macro tag
- :func:`find_closest_analogs`  — top-N past events most like today

All forward returns are decimal *percentages* (e.g., ``+1.5`` = +1.5%).
Events with insufficient forward data leave NaN in those columns —
callers should ``.dropna()`` before aggregating.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis.macro_calendar import macro_tag


DEFAULT_HORIZONS = (3, 5, 10, 20, 60)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def find_drop_events(
    prices: pd.Series,
    drop_threshold_pct: float = -3.0,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Find all days where close-to-close return ≤ ``drop_threshold_pct``.

    Parameters
    ----------
    prices : pd.Series indexed by date, values = daily close.
    drop_threshold_pct : float
        Threshold (negative; e.g., ``-3.0`` for ≥3% drops).
    horizons : iterable of int
        Forward-return horizons (in trading days).

    Returns
    -------
    DataFrame indexed by event date with columns::

        prev_close, close, drop_pct, macro_tag,
        fwd_3d, fwd_5d, fwd_10d, fwd_20d, fwd_60d (per horizon)

    Empty DataFrame if no events found or input invalid.
    """
    horizons = tuple(horizons)
    if prices is None or not isinstance(prices, pd.Series) or prices.empty:
        return _empty_df(horizons)

    p = pd.to_numeric(prices, errors="coerce").dropna()
    if len(p) < 2:
        return _empty_df(horizons)

    df = pd.DataFrame({"close": p})
    df["prev_close"] = df["close"].shift(1)
    df["drop_pct"] = (df["close"] / df["prev_close"] - 1) * 100

    for n in horizons:
        df[f"fwd_{n}d"] = (df["close"].shift(-n) / df["close"] - 1) * 100

    events = df[df["drop_pct"] <= drop_threshold_pct].copy()
    if events.empty:
        return _empty_df(horizons)

    events["macro_tag"] = pd.Series(events.index).apply(
        lambda d: macro_tag(d) or "OTHER"
    ).values
    cols = ["prev_close", "close", "drop_pct", "macro_tag"] + [
        f"fwd_{n}d" for n in horizons
    ]
    return events[cols]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def analog_summary_by_tag(
    events: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Aggregate per-tag stats: N, mean, median, win-rate per horizon.

    Parameters
    ----------
    events : DataFrame from :func:`find_drop_events`.
    horizons : forward horizons present in ``events``.

    Returns
    -------
    DataFrame with one row per tag, columns::

        tag, n,
        {n}d_mean, {n}d_median, {n}d_winrate   for each horizon

    Sorted by N descending.  Empty DataFrame if input is empty.
    """
    horizons = tuple(horizons)
    if events is None or events.empty:
        return pd.DataFrame()

    rows = []
    for tag in events["macro_tag"].dropna().unique():
        sub = events[events["macro_tag"] == tag]
        row: dict = {"tag": tag, "n": len(sub)}
        for n in horizons:
            col = f"fwd_{n}d"
            s = sub[col].dropna()
            row[f"{n}d_mean"] = float(s.mean()) if len(s) else float("nan")
            row[f"{n}d_median"] = float(s.median()) if len(s) else float("nan")
            row[f"{n}d_winrate"] = (
                float((s > 0).mean() * 100) if len(s) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Similarity ranking
# ---------------------------------------------------------------------------


def find_closest_analogs(
    events: pd.DataFrame,
    today_drop_pct: float,
    today_macro_tag: Optional[str] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank past events by similarity to today.

    Similarity rule (deterministic, no ML):
    1. Events matching ``today_macro_tag`` rank above non-matching.
    2. Within each group, closer ``drop_pct`` to ``today_drop_pct`` ranks
       higher (absolute difference).

    This favours "same driver" interpretations.  Pass ``today_macro_tag=None``
    or ``"OTHER"`` to disable tag preference and rank purely by drop magnitude.

    Parameters
    ----------
    events : DataFrame from :func:`find_drop_events`.
    today_drop_pct : float
        Today's drop percentage (negative).
    today_macro_tag : str, optional
        Today's tag ("NFP" / "CPI" / "FOMC" / "OTHER" / None).
    top_n : int

    Returns
    -------
    DataFrame — same columns as ``events``, sorted, head ``top_n``.
    """
    if events is None or events.empty:
        return events if events is not None else pd.DataFrame()

    df = events.copy()
    if today_macro_tag and today_macro_tag != "OTHER":
        df["_tag_match"] = (df["macro_tag"] == today_macro_tag).astype(int)
    else:
        df["_tag_match"] = 0
    df["_drop_diff"] = (df["drop_pct"] - today_drop_pct).abs()
    df = df.sort_values(
        ["_tag_match", "_drop_diff"], ascending=[False, True],
    )
    return df.drop(columns=["_tag_match", "_drop_diff"]).head(top_n)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _empty_df(horizons: Iterable[int]) -> pd.DataFrame:
    cols = ["prev_close", "close", "drop_pct", "macro_tag"] + [
        f"fwd_{n}d" for n in horizons
    ]
    return pd.DataFrame(columns=cols)
