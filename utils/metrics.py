"""Portfolio metrics utilities — used by dashboard and testable in isolation."""

import pandas as pd


def drawdown_stats(curve: pd.Series) -> tuple[float, float, int]:
    """Return (current_drawdown_pct, max_drawdown_pct, longest_drawdown_days)."""
    rolling_max = curve.expanding().max()
    dd = (curve - rolling_max) / rolling_max * 100
    current_dd = float(dd.iloc[-1])
    is_under = dd < 0
    longest = 0
    streak = 0
    for flag in is_under:
        if flag:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return current_dd, float(dd.min()), longest


def exposure_from_trades(result, curve: pd.Series) -> tuple[pd.Series, float, dict]:
    """Reconstruct daily net exposure as % of equity (mark-to-market).

    Return (net_pct_series, last_pct, top3_weights_dict).
    """
    if not result.closed_trades or len(curve) < 2:
        return pd.Series(dtype=float), 0.0, {}

    exposure = pd.Series(0.0, index=curve.index)
    symbols = list({t.symbol for t in result.closed_trades})
    by_symbol = {sym: pd.Series(0.0, index=curve.index) for sym in symbols}

    for t in result.closed_trades:
        entry_cost = t.entry_price * t.qty * 1.0004
        exit_value = entry_cost + (t.pnl or 0)

        mask = (exposure.index >= t.entry_time) & (exposure.index <= t.exit_time)
        dates_in = exposure.index[mask]

        if len(dates_in) < 2:
            exposure.loc[mask] += entry_cost
            if t.symbol in by_symbol:
                by_symbol[t.symbol].loc[mask] += entry_cost
            continue

        total_days = (dates_in[-1] - dates_in[0]).days or 1
        for d in dates_in:
            frac = (d - dates_in[0]).days / total_days
            mtm_value = entry_cost + (exit_value - entry_cost) * frac
            exposure.loc[d] += mtm_value
            if t.symbol in by_symbol:
                by_symbol[t.symbol].loc[d] += mtm_value

    net_pct = (exposure / curve) * 100
    last_exp = float(net_pct.iloc[-1]) if len(net_pct) > 0 else 0.0
    top = {}
    for sym, ser in sorted(by_symbol.items(), key=lambda x: -x[1].iloc[-1])[:3]:
        top[sym] = round(float(ser.iloc[-1] / curve.iloc[-1] * 100), 1)
    return net_pct, last_exp, top
