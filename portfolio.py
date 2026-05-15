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


@dataclass
class PortfolioTrade:
    """A single round-trip trade recorded during portfolio backtest."""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    qty: int = 0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    reason: str = ""
    hold_days: Optional[int] = None


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
        trades: List[PortfolioTrade] = []
        open_trade_idx: dict = {}  # leg_index -> trades index
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
                        cash = self._close_trade(
                            st, date_idx, price, reason,
                            open_trade_idx, trades, i, cash,
                        )

                # Entry check
                elif st["position"] == 0 and st["strategy"].entry_signal(df_sig, idx_pos):
                    active_positions = sum(1 for s in leg_state.values() if s.get("position", 0) > 0)
                    if active_positions >= self.max_positions:
                        continue
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

                            trade = PortfolioTrade(
                                symbol=st["leg"].symbol,
                                entry_time=date_idx,
                                qty=qty,
                                entry_price=actual_price,
                            )
                            trades.append(trade)
                            open_trade_idx[i] = len(trades) - 1

                total_equity += st["position"] * price

            equity_history.append((date_idx, total_equity))

        # Close all open positions at end of period (same path as signal exit)
        last_date = timeline[-1] if timeline else pd.Timestamp(end)
        for i, st in leg_state.items():
            if st["position"] > 0 and "last_price" in st:
                cash = self._close_trade(
                    st, last_date, st["last_price"], "end_of_period",
                    open_trade_idx, trades, i, cash,
                )
        final_equity = cash

        return PortfolioResult(
            equity_history=equity_history,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            legs=self.legs,
            trades=trades,
        )

    def _close_trade(
        self, st: dict, date_idx: pd.Timestamp, price: float, reason: str,
        open_trade_idx: dict, trades: List[PortfolioTrade],
        leg_index: int, cash: float,
    ) -> float:
        """Close a position — unified path for signal exits and end-of-period close.

        Applies slippage and commission consistently.  Fills the matching
        PortfolioTrade with exit info computed on a net basis:
        entry_cost includes buy-side commission, exit_proceeds deducts
        sell-side commission, so trade PnL is strictly aligned with
        the cash curve.
        """
        qty = st["position"]
        exit_price = price * (1 - self.slippage_pct)
        exit_proceeds = exit_price * qty * (1 - self.commission_rate)
        entry_cost = st["capital_allocated"]
        cash += exit_proceeds

        if leg_index in open_trade_idx:
            t = trades[open_trade_idx[leg_index]]
            t.exit_time = date_idx
            t.exit_price = exit_price
            t.pnl = exit_proceeds - entry_cost
            t.pnl_pct = (t.pnl / entry_cost * 100) if entry_cost > 0 else 0
            t.reason = reason
            t.hold_days = (date_idx - t.entry_time).days
            del open_trade_idx[leg_index]

        st["position"] = 0
        st["capital_allocated"] = 0
        st["entry_price"] = 0
        st["highest"] = 0
        return cash

    def _allocate(self, leg: Leg, available_cash: float, num_legs: int) -> float:
        """Determine how much cash to allocate to a new position."""
        if self.allocation == "equal":
            return min(available_cash, self.initial_capital / max(num_legs, 1))
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
    trades: List[PortfolioTrade] = field(default_factory=list)

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

    # --- Trade statistics ---

    @property
    def closed_trades(self) -> List[PortfolioTrade]:
        return [t for t in self.trades if t.exit_price is not None]

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate_pct(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0
        wins = sum(1 for t in closed if t.pnl is not None and t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0

    @property
    def avg_hold_days(self) -> float:
        closed = [t for t in self.closed_trades if t.hold_days is not None]
        return sum(t.hold_days for t in closed) / len(closed) if closed else 0

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

        # Trade statistics
        print(f"  --- 交易统计 ---")
        print(f"  总交易笔数:    {self.total_trades}")
        print(f"  胜率:          {self.win_rate_pct:.1f}%")
        print(f"  盈亏比:        {self.profit_factor:.2f}")
        print(f"  平均盈利:      ${self.avg_win:,.0f}")
        print(f"  平均亏损:      ${self.avg_loss:,.0f}")
        print(f"  平均持仓天数:  {self.avg_hold_days:.1f}")
        print()

        # Per-symbol breakdown
        if self.closed_trades:
            by_symbol: dict = {}
            for t in self.closed_trades:
                by_symbol.setdefault(t.symbol, []).append(t)
            print(f"  {'标的':<8s} {'笔数':>4s} {'胜率':>6s} {'总PnL':>10s} {'平均PnL':>10s}")
            print(f"  {'─' * 44}")
            for sym, sym_trades in sorted(by_symbol.items()):
                n = len(sym_trades)
                wr = sum(1 for t in sym_trades if t.pnl is not None and t.pnl > 0) / n * 100
                total_pnl = sum(t.pnl or 0 for t in sym_trades)
                avg_pnl = total_pnl / n
                print(f"  {sym:<8s} {n:>4d} {wr:>5.0f}% ${total_pnl:>9,.0f} ${avg_pnl:>9,.0f}")
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
    Leg("AAPL", "weekly_macd_kdj"),
    Leg("NVDA", "weekly_macd_kdj"),
    Leg("TSLA", "weekly_macd_kdj"),
    Leg("QQQ",  "weekly_macd_kdj"),
    Leg("SPY",  "turtle_trading",  {}),
    Leg("510300", "turtle_trading", {}),
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
