"""Data quality checks — flag issues before they reach strategies / backtests."""

from typing import Tuple

import numpy as np
import pandas as pd


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_missing`` column: True where any OHLCV column is NaN."""
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    cols = [c for c in ohlcv if c in df.columns]
    df["_missing"] = df[cols].isna().any(axis=1)
    return df


def flag_price_jumps(df: pd.DataFrame, threshold_pct: float = 20.0) -> pd.DataFrame:
    """Add ``_price_jump`` column: True where daily return exceeds ±threshold%.

    Helps detect bad ticks, split-correction errors, or data glitches.
    """
    ret = df["Close"].pct_change().abs() * 100
    df["_price_jump"] = ret > threshold_pct
    return df


def flag_non_trading(df: pd.DataFrame, min_volume_ratio: float = 0.05) -> pd.DataFrame:
    """Add ``_non_trading`` column: True for bars with anomalously low volume.

    A bar whose volume is below *min_volume_ratio* × 20-day median is flagged.
    """
    median_vol = df["Volume"].rolling(20).median()
    df["_non_trading"] = (df["Volume"] < median_vol * min_volume_ratio) & (median_vol > 0)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Return a summary dict of all quality flags.  Lightweight, loggable."""
    return {
        "bars": len(df),
        "missing_pct": round(df["_missing"].mean() * 100, 2) if "_missing" in df.columns else None,
        "price_jumps": int(df["_price_jump"].sum()) if "_price_jump" in df.columns else None,
        "non_trading_pct": round(df["_non_trading"].mean() * 100, 2) if "_non_trading" in df.columns else None,
    }


def clean(
    df: pd.DataFrame,
    drop_missing: bool = True,
    drop_jumps: bool = False,
    drop_non_trading: bool = False,
) -> pd.DataFrame:
    """Run all quality checks and optionally drop flagged rows."""
    df = flag_missing(df)
    df = flag_price_jumps(df)
    df = flag_non_trading(df)
    if drop_missing:
        df = df[~df["_missing"]]
    if drop_jumps:
        df = df[~df["_price_jump"]]
    if drop_non_trading:
        df = df[~df["_non_trading"]]
    return df


def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Fast pre-trade sanity check on the most recent bar.

    Returns (ok, reason).  Checks: non-empty, OHLC cols present,
    High ≥ Low, Close within [Low, High].
    """
    if df is None or df.empty:
        return False, "empty_df"
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        return False, f"missing_cols:{missing}"
    last = df.iloc[-1]
    if pd.isna(last["Close"]):
        return False, "close_nan"
    if last["High"] < last["Low"]:
        return False, f"high({last['High']}) < low({last['Low']})"
    if last["Close"] < last["Low"] or last["Close"] > last["High"]:
        return False, f"close({last['Close']}) oob [{last['Low']}, {last['High']}]"
    return True, "ok"
