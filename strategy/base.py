"""Strategy interface — every strategy must implement this.

Design
------
- `calculate_indicators(df)` → DataFrame with indicators + 'Signal' column
- `check_exit(...)` → (exit_flag, reason) — exit logic lives in the strategy
- `position_size(...)` → int — sizing logic lives in the strategy
- `min_bars` property — minimum bars before first signal

The backtest engine never inspects strategy internals — it acts solely on
public methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters — frozen dataclass, validated on construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter bag.  Subclass per strategy with typed defaults."""

    def validate(self):
        """Override to add parameter sanity checks.  Called by __post_init__."""
        pass

    def __post_init__(self):
        self.validate()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Every strategy MUST subclass this and implement the abstract methods.

    Subclass contract
    -----------------
    1. `calculate_indicators(df)` — return `df` with indicator columns AND
       a `'Signal'` column (1 = long entry, -1 = exit, 0 = hold).
    2. `min_bars` — how many bars must elapse before the first signal fires.

    Optional overrides (sensible defaults provided):
    3. `check_exit(...)` — exit conditions (default: Signal == -1)
    4. `position_size(...)` — position sizing (default: 95 % of capital)
    5. `entry_signal(...)` — entry trigger (default: Signal == 1)
    """

    params: StrategyParams  # set by __init_subclass__ or constructor

    def __init__(self, params: Optional[StrategyParams] = None):
        if params is not None:
            self.params = params

    # -- mandatory -----------------------------------------------------------

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with indicator columns and a `'Signal'` column added.

        Signal semantics
        ----------------
          1  = open / hold long
         -1  = close / exit
          0  = hold current state

        The returned DataFrame may have a different index (e.g. weekly) from
        the input — the engine handles alignment.

        Must always include an `'ATR'` column (for position sizing).
        """
        ...

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum rows needed before `calculate_indicators` produces valid data."""
        ...

    # -- optional (default implementations) ----------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        """Default: invest 95 % of available capital in one trade."""
        if price <= 0:
            return 0
        return int(capital * 0.95 / price)

    def entry_signal(self, df: pd.DataFrame, i: int) -> bool:
        """Return True if bar *i* triggers a long entry."""
        return int(df["Signal"].iloc[i]) == 1

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Return (exit_now, reason).

        Override to add stop-loss / take-profit / trailing-stop logic.
        Default: exit when Signal == -1.
        """
        if int(df["Signal"].iloc[i]) == -1:
            return True, "卖出信号"
        return False, ""


# ---------------------------------------------------------------------------
# Built-in helpers that subclasses can reuse
# ---------------------------------------------------------------------------


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Welles Wilder ATR via EMA smoothing."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_macd(
    df: pd.DataFrame, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    """Add MACD / MACD_signal / MACD_hist columns to *df*."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def compute_kdj(
    df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3
) -> pd.DataFrame:
    """Add K / D / J columns to *df*."""
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)

    k_vals, d_vals = [50.0], [50.0]
    alpha_k, alpha_d = 1 / k_period, 1 / d_period
    for r in rsv.iloc[1:]:
        k_vals.append(alpha_k * r + (1 - alpha_k) * k_vals[-1])
        d_vals.append(alpha_d * k_vals[-1] + (1 - alpha_d) * d_vals[-1])

    df["K"] = k_vals
    df["D"] = d_vals
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Friday close)."""
    return (
        df.resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
