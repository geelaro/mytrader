"""Portfolio backtest — multi-symbol, shared capital pool.

Usage:
    pipenv run python portfolio.py
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Tuple

import utils  # noqa: F401 — triggers env setup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from data import DataProvider
from strategy import BaseStrategy, STRATEGY_MAP as _STRATEGY_MAP
from trader import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """One leg of the portfolio — a symbol + strategy pair."""

    symbol: str
    strategy_name: str
    params: dict = field(default_factory=dict)

    def create_strategy(self) -> BaseStrategy:
        cls = _STRATEGY_MAP.get(self.strategy_name)
        if cls is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        return cls(**self.params)


# ---------------------------------------------------------------------------
# Portfolio backtest engine
# ---------------------------------------------------------------------------


class PortfolioBacktest:
    """Multi-symbol backtest with shared cash pool.

    Each leg gets an equal fraction of capital by default.
    All legs share a single cash pool — when one exits, cash
    is returned to the pool for other legs to use.
    """

    def __init__(
        self,
        legs: List[Leg],
        initial_capital: float = 100000,
        allocation: str = "equal",
        max_positions: int = 10,
        commission_rate: float = 0.0003,
        slippage_pct: float = 0.0001,
    ):
        self.legs = legs
        self.initial_capital = initial_capital
        self.allocation = allocation
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct

    def run(
        self, start: str = "2018-01-01", end: Optional[str] = None
    ) -> "PortfolioResult":
        if end is None:
            end = date.today().isoformat()

        provider = DataProvider()

        # Fetch all data & calculate signals upfront
        leg_data = []  # list of (leg, strategy, df_sig)
        for leg in self.legs:
            df = provider.get_daily(leg.symbol, start=start, end=end)
            if df is None or df.empty:
                print(f"  ! {leg.symbol} 无数据，跳过")
                continue
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            strategy = leg.create_strategy()
            df_sig = strategy.calculate_indicators(df)
            leg_data.append((leg, strategy, df_sig))
            print(f"  {leg.symbol:<8s}  {leg.strategy_name:<20s}  {len(df_sig)} 根K线")

        if not leg_data:
            raise RuntimeError("所有标的均无数据")

        # Build common timeline
        all_dates = set()
        for _, _, df_sig in leg_data:
            all_dates.update(df_sig.index)
        timeline = sorted(all_dates)

        # Per-leg state
        leg_state = {}
        for i, (leg, strategy, df_sig) in enumerate(leg_data):
            leg_state[i] = {
                "leg": leg,
                "strategy": strategy,
                "df": df_sig,
                "position": 0,
                "entry_price": 0.0,
                "highest": 0.0,
                "capital_allocated": 0.0,
            }

        cash = self.initial_capital
        total_trades = []
        equity_history = []

        for date_idx in timeline:
            total_equity = cash

            for i, st in leg_state.items():
                df_sig = st["df"]
                if date_idx not in df_sig.index:
                    total_equity += st["position"] * st.get("last_price", 0)
                    continue

                idx_pos = df_sig.index.get_loc(date_idx)
                if idx_pos < st["strategy"].min_bars:
                    total_equity += st["position"] * st.get("last_price", 0)
                    continue

                row = df_sig.iloc[idx_pos]
                price = float(row["Close"])
                atr = float(row["ATR"]) if "ATR" in row.index else 0
                st["last_price"] = price

                # Exit check
                if st["position"] > 0:
                    if price > st["highest"]:
                        st["highest"] = price

                    exit_now, reason = st["strategy"].check_exit(
                        df_sig, idx_pos,
                        entry_price=st["entry_price"],
                        highest_since_entry=st["highest"],
                    )
                    if exit_now:
                        actual_price = price * (1 - self.slippage_pct)
                        proceeds = actual_price * st["position"] * (1 - self.commission_rate)
                        cash += proceeds
                        st["position"] = 0
                        st["capital_allocated"] = 0
                        st["entry_price"] = 0
                        st["highest"] = 0

                # Entry check
                elif st["position"] == 0 and st["strategy"].entry_signal(df_sig, idx_pos):
                    alloc = self._allocate(st["leg"], cash, len(leg_data))
                    if alloc > 0:
                        qty = st["strategy"].position_size(alloc, price, atr)
                        actual_price = price * (1 + self.slippage_pct)
                        cost = actual_price * qty * (1 + self.commission_rate)
                        if cost <= cash and qty > 0:
                            cash -= cost
                            st["position"] = qty
                            st["entry_price"] = actual_price
                            st["highest"] = price
                            st["capital_allocated"] = cost

                total_equity += st["position"] * price

            equity_history.append((date_idx, total_equity))

        # Close all open positions
        final_equity = cash
        for i, st in leg_state.items():
            if st["position"] > 0 and "last_price" in st:
                price = st["last_price"] * (1 - self.slippage_pct)
                final_equity += st["position"] * price

        return PortfolioResult(
            equity_history=equity_history,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            legs=self.legs,
        )

    def _allocate(self, leg: Leg, available_cash: float, num_legs: int) -> float:
        """Determine how much cash to allocate to a new position."""
        if self.allocation == "equal":
            # Equal split of total capital
            return self.initial_capital / max(num_legs, 1)
        elif self.allocation == "fraction":
            # Fraction of available cash
            return available_cash * 0.25
        # Default: use all available cash (single-position portfolio)
        return available_cash


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    equity_history: List[Tuple[pd.Timestamp, float]]
    initial_capital: float
    final_equity: float
    legs: List[Leg]

    @property
    def equity_curve(self) -> pd.Series:
        df = pd.DataFrame(self.equity_history, columns=["date", "equity"])
        df = df.drop_duplicates("date").set_index("date").sort_index()
        return df["equity"]

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def cagr_pct(self) -> float:
        curve = self.equity_curve
        days = (curve.index[-1] - curve.index[0]).days
        if days <= 0:
            return 0
        return ((self.final_equity / self.initial_capital) ** (365.25 / days) - 1) * 100

    @property
    def sharpe_ratio(self) -> float:
        rets = self.equity_curve.pct_change().dropna()
        if rets.std() == 0:
            return 0
        return np.sqrt(252) * rets.mean() / rets.std()

    @property
    def max_drawdown_pct(self) -> float:
        curve = self.equity_curve
        rolling_max = curve.expanding().max()
        dd = (curve - rolling_max) / rolling_max * 100
        return float(dd.min())

    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"  组合回测结果")
        print(f"{'=' * 60}")
        print(f"  标的数量:      {len(self.legs)}")
        print(f"  初始资金:      ${self.initial_capital:,.0f}")
        print(f"  最终权益:      ${self.final_equity:,.0f}")
        print(f"  总收益率:      {self.total_return_pct:+.2f}%")
        print(f"  年化收益率:    {self.cagr_pct:+.2f}%")
        print(f"  夏普比率:      {self.sharpe_ratio:.2f}")
        print(f"  最大回撤:      {self.max_drawdown_pct:.2f}%")
        print()

    def plot(self, save_path: str = "charts/portfolio_result.png"):
        try:
            for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
                try:
                    plt.rcParams["font.sans-serif"] = [font]
                    break
                except Exception:
                    continue
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})

        curve = self.equity_curve
        ax1 = axes[0]
        ax1.plot(curve.index, curve, color="#2ca02c", linewidth=1.2, label="组合权益")
        ax1.axhline(y=self.initial_capital, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax1.set_ylabel("Equity ($)")
        ax1.set_title("组合权益曲线", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        ax2 = axes[1]
        rolling_max = curve.expanding().max()
        drawdown = (curve - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.4)
        ax2.plot(drawdown.index, drawdown, color="#d62728", linewidth=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.set_title("组合回撤", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


DEFAULT_PORTFOLIO = [
    Leg("AAPL", "trend_follower", {"short_ma": 10, "long_ma": 40, "adx_threshold": 15, "trail_atr_mult": 3.0}),
    Leg("NVDA", "trend_follower", {"short_ma": 10, "long_ma": 40, "adx_threshold": 15, "trail_atr_mult": 3.0}),
    Leg("TSLA", "trend_follower", {"short_ma": 10, "long_ma": 40, "adx_threshold": 15, "trail_atr_mult": 3.0}),
    Leg("QQQ",  "trend_follower", {"short_ma": 10, "long_ma": 40, "adx_threshold": 15, "trail_atr_mult": 3.0}),
    Leg("SPY",  "weekly_macd",   {}),
    Leg("510300", "weekly_macd", {}),
]


def main():
    print("组合回测")
    print(f"{'=' * 60}")

    legs = DEFAULT_PORTFOLIO
    bt = PortfolioBacktest(
        legs=legs,
        initial_capital=100000,
        allocation="equal",
    )
    result = bt.run(start="2018-01-01")
    result.summary()
    result.plot()


if __name__ == "__main__":
    main()
