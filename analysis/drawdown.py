"""Drawdown analytics — beyond a single MaxDD number.

A single MaxDD figure tells you the worst point.  It doesn't tell you:
- how long the portfolio spent underwater,
- how many separate drawdown episodes occurred,
- how long the typical recovery takes,
- whether the portfolio is *still* in a drawdown.

These all matter for whether a strategy is psychologically sustainable.
A −15% MaxDD that recovers in 2 weeks is very different from a −15%
MaxDD that takes 18 months to recover.

Modules
-------
- :func:`underwater_curve` — time series of "% below all-time peak".
- :func:`drawdown_episodes` — DataFrame of historical drawdown episodes
  with peak/trough/recovery dates, depth, duration.
- :func:`time_to_recover_stats` — distributional stats of episode recovery
  times (median, p75, p95).
- :func:`drawdown_summary` — one-stop combined diagnostic for dashboards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def underwater_curve(returns: pd.Series) -> pd.Series:
    """Equity / cummax − 1, in percent (≤ 0).

    Each point: how far below the all-time-high running peak the equity
    is at that moment.  0 means at a new peak; −20.0 means 20% below the
    most recent peak.
    """
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    return (equity / peak - 1) * 100


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Enumerate distinct drawdown episodes.

    An episode starts when equity dips below its running peak and ends
    when equity reaches a new all-time-high.  Episodes still underwater
    at the end of the series have ``recovery_date = NaT`` and
    ``duration_to_recovery = NaN``.

    Returns
    -------
    DataFrame with columns:
        - peak_date:               start of the episode (when peak was set)
        - trough_date:             date of the worst point
        - recovery_date:           first date back to peak, or NaT
        - depth_pct:               negative, e.g. −25.3
        - duration_to_trough_days: days from peak to trough
        - duration_to_recovery_days: peak → recovery (or NaN if still in DD)

    Sorted by ``depth_pct`` ascending (worst first).
    """
    r = _clean(returns)
    if r.empty:
        return _empty_episodes_df()
    equity = (1 + r).cumprod()
    peak = equity.cummax()

    # Episode segmentation: an episode is a contiguous run of bars where
    # equity < peak.  The bar BEFORE the run is the peak_date.
    in_dd = equity < peak
    if not in_dd.any():
        return _empty_episodes_df()

    # Group consecutive in_dd=True bars
    # group_id increments each time the in_dd value flips True->False or False->True
    group_id = (in_dd != in_dd.shift()).cumsum()
    episodes: list = []
    for gid, group in in_dd.groupby(group_id):
        if not group.iloc[0]:  # False group (at peak), skip
            continue
        dd_idx = group.index
        # Peak date: the bar immediately before the run.  If the series
        # starts in DD (impossible since first bar is always peak), use first.
        first_dd_pos = equity.index.get_loc(dd_idx[0])
        peak_date = equity.index[first_dd_pos - 1] if first_dd_pos > 0 else dd_idx[0]

        # Trough: min equity within the episode
        ep_equity = equity.loc[dd_idx]
        trough_date = ep_equity.idxmin()
        depth_pct = (ep_equity.min() / equity.loc[peak_date] - 1) * 100

        # Recovery: bar immediately after the run, if any
        last_dd_pos = equity.index.get_loc(dd_idx[-1])
        if last_dd_pos + 1 < len(equity):
            recovery_date = equity.index[last_dd_pos + 1]
            duration_recover = _days(peak_date, recovery_date)
        else:
            recovery_date = pd.NaT
            duration_recover = float("nan")

        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "depth_pct": float(depth_pct),
            "duration_to_trough_days": _days(peak_date, trough_date),
            "duration_to_recovery_days": duration_recover,
        })

    df = pd.DataFrame(episodes)
    if df.empty:
        return _empty_episodes_df()
    return df.sort_values("depth_pct").reset_index(drop=True)


def time_to_recover_stats(returns: pd.Series) -> dict:
    """Distributional stats of recovery duration (days) across episodes.

    Excludes episodes still underwater (no recovery_date).  Returns dict::

        {
            "n_episodes":          int,   # completed only
            "median_days":         float, # NaN if no completed episodes
            "p75_days":            float,
            "p95_days":            float,
            "max_days":            float,
            "still_in_drawdown":   bool,
            "current_dd_days":     int,   # if still underwater, days since peak
        }
    """
    episodes = drawdown_episodes(returns)
    if episodes.empty:
        return _empty_recover_stats()

    completed = episodes.dropna(subset=["duration_to_recovery_days"])
    in_progress = episodes[episodes["recovery_date"].isna()]

    out: dict = {
        "n_episodes": int(len(completed)),
        "median_days": float("nan"),
        "p75_days": float("nan"),
        "p95_days": float("nan"),
        "max_days": float("nan"),
        "still_in_drawdown": bool(not in_progress.empty),
        "current_dd_days": 0,
    }
    if not completed.empty:
        durations = completed["duration_to_recovery_days"]
        out["median_days"] = float(durations.median())
        out["p75_days"] = float(durations.quantile(0.75))
        out["p95_days"] = float(durations.quantile(0.95))
        out["max_days"] = float(durations.max())
    if not in_progress.empty:
        # Most recent ongoing episode
        ongoing = in_progress.sort_values("peak_date").iloc[-1]
        last_idx = _clean(returns).index[-1]
        out["current_dd_days"] = _days(ongoing["peak_date"], last_idx)
    return out


def drawdown_summary(returns: pd.Series, top_n: int = 5) -> dict:
    """One-stop summary: max + average DD, top-N episodes, recovery stats."""
    r = _clean(returns)
    if r.empty:
        return {
            "max_drawdown_pct": 0.0,
            "avg_drawdown_pct": 0.0,
            "pct_time_underwater": 0.0,
            "top_episodes": [],
            "recovery_stats": _empty_recover_stats(),
        }

    underwater = underwater_curve(r)
    pct_underwater = float((underwater < 0).mean() * 100)
    avg_dd = float(-underwater[underwater < 0].mean()) if (underwater < 0).any() else 0.0

    episodes = drawdown_episodes(r)
    top = episodes.head(top_n).to_dict("records") if not episodes.empty else []

    return {
        "max_drawdown_pct": float(-underwater.min()),
        "avg_drawdown_pct": avg_dd,
        "pct_time_underwater": pct_underwater,
        "top_episodes": top,
        "recovery_stats": time_to_recover_stats(r),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series):
        return pd.Series(dtype=float)
    r = pd.to_numeric(returns, errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _days(start, end) -> int:
    """Days between two index values.  Handles both DatetimeIndex and ints."""
    if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        return int((end - start).days)
    try:
        return int(end - start)
    except Exception:
        return 0


def _empty_episodes_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "peak_date", "trough_date", "recovery_date", "depth_pct",
        "duration_to_trough_days", "duration_to_recovery_days",
    ])


def _empty_recover_stats() -> dict:
    return {
        "n_episodes": 0,
        "median_days": float("nan"),
        "p75_days": float("nan"),
        "p95_days": float("nan"),
        "max_days": float("nan"),
        "still_in_drawdown": False,
        "current_dd_days": 0,
    }
