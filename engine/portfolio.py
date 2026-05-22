"""Portfolio backtest — multi-symbol, shared capital pool.

Usage:
    pipenv run python engine/portfolio.py
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Tuple

import utils  # noqa: F401 — triggers env setup
from utils.font import setup_chinese_font
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from data import DataProvider
from broker import OrderSide, OrderStatus
from strategy import BaseStrategy, STRATEGY_MAP as _STRATEGY_MAP
from engine.trader import BacktestEngine, BacktestResult
from engine.execution import ExecutionConfig, ExecutionModel, ExecutionTiming


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
        # --- portfolio risk ---
        max_symbol_weight: float = 0.0,
        max_daily_new_positions: int = 0,
        max_gross_exposure: float = 0.0,
        # --- execution constraints ---
        lot_size: int = 0,
        max_participation_rate: float = 0.0,
        # --- sizing ---
        sizing_mode: str = "fixed_capital",
        risk_per_trade: float = 0.005,
        risk_atr_mult: float = 2.0,
        execution_model: ExecutionModel | None = None,
        execution_timing: str = "next_open",
        # --- enhanced risk ---
        max_sector_weight: float = 0.0,
        sector_map: Optional[dict] = None,
        cooldown_after_stop_days: int = 0,
    ):
        self.legs = legs
        self.initial_capital = initial_capital
        self.allocation = allocation
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        if execution_model is None:
            execution_model = ExecutionModel(ExecutionConfig(
                timing=ExecutionTiming(execution_timing),
                slippage_pct=slippage_pct,
                commission_rate=commission_rate,
                max_participation_rate=max_participation_rate,
            ))
        self.execution_model = execution_model
        self.max_symbol_weight = max_symbol_weight
        self.max_daily_new_positions = max_daily_new_positions
        self._rejections: list[dict] = []
        self.max_gross_exposure = max_gross_exposure
        self.lot_size = lot_size
        self.max_participation_rate = max_participation_rate
        self.sizing_mode = sizing_mode
        self.risk_per_trade = risk_per_trade
        self.risk_atr_mult = risk_atr_mult
        self.max_sector_weight = max_sector_weight
        self.sector_map = sector_map or {}
        self.cooldown_after_stop_days = cooldown_after_stop_days
        self._stop_dates: dict[str, Optional[pd.Timestamp]] = {}

    def _reject(self, date_idx, symbol: str, reason: str, detail: str = "") -> None:
        """Record a rejection and return — used in risk gate chains."""
        self._rejections.append({
            "date": date_idx, "symbol": symbol,
            "reason": reason, "detail": detail,
        })

    def _calc_risk_budget_qty(self, capital: float, price: float, atr: float) -> int:
        """Return position size such that stop_distance loss = risk_per_trade % of capital."""
        if price <= 0:
            return 0
        if atr is None or np.isnan(atr) or atr <= 0:
            atr = price * 0.02
        risk_dollar = capital * self.risk_per_trade
        stop_distance = atr * self.risk_atr_mult
        if stop_distance <= 0:
            return 0
        shares = int(risk_dollar / stop_distance)
        max_shares = int(capital / (price * (1 + self.slippage_pct + self.commission_rate)))
        return max(0, min(shares, max_shares))

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
                "pending_order": None,
            }

        cash = self.initial_capital
        trades: List[PortfolioTrade] = []
        open_trade_idx: dict = {}  # leg_index -> trades index
        equity_history = []
        daily_new_count = 0
        prev_date = None

        for date_idx in timeline:
            # Reset daily counter on new date
            if prev_date is not None and date_idx.date() != prev_date.date():
                daily_new_count = 0
            prev_date = date_idx

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
                open_price = float(row["Open"]) if "Open" in row.index else price
                atr = float(row["ATR"]) if "ATR" in row.index else 0
                st["last_price"] = price

                pending = st.get("pending_order")
                if pending is not None:
                    fill = self.execution_model.execute_bar(
                        pending,
                        row,
                        date_idx,
                        idx_pos,
                        available_qty=st["position"] if pending.side == OrderSide.SELL else None,
                    )
                    if pending.side == OrderSide.SELL and st["position"] > 0:
                        if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                            cash = self._apply_close_fill(
                                st, fill, pending.reason,
                                open_trade_idx, trades, i, cash,
                            )
                            pending.quantity -= fill.filled_qty
                        if (fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
                                or st["position"] == 0):
                            st["pending_order"] = None
                        continue
                    if pending.side == OrderSide.BUY:
                        if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                            if self._apply_open_fill(st, fill, trades, open_trade_idx, i, cash):
                                cash += fill.net_cash_delta
                                pending.quantity -= fill.filled_qty
                        if (fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
                                or pending.quantity <= 0):
                            st["pending_order"] = None
                        total_equity += st["position"] * price
                        continue
                    elif fill.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                        st["pending_order"] = None
                        total_equity += st["position"] * price
                        continue

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
                        st["pending_order"] = self.execution_model.make_plan(
                            symbol=st["leg"].symbol,
                            side=OrderSide.SELL,
                            quantity=st["position"],
                            created_index=idx_pos,
                            reason=reason,
                        )

                # Entry check
                elif (st["position"] == 0
                      and st.get("pending_order") is None
                      and idx_pos < len(df_sig) - 1
                      and st["strategy"].entry_signal(df_sig, idx_pos)):
                    active_positions = sum(1 for s in leg_state.values() if s.get("position", 0) > 0)
                    if active_positions >= self.max_positions:
                        self._reject(date_idx, st["leg"].symbol, "仓位上限",
                                     f"活跃仓位{active_positions}≥{self.max_positions}")
                        continue

                    # Compute current equity for dynamic allocation & risk checks
                    current_equity = cash + sum(
                        s["position"] * s.get("last_price", 0) for s in leg_state.values()
                    )

                    # --- risk: cooldown after stop ---
                    sym = st["leg"].symbol
                    if self.cooldown_after_stop_days > 0 and sym in self._stop_dates:
                        last_stop = self._stop_dates[sym]
                        if last_stop is not None:
                            days_since = (date_idx - last_stop).days
                            if days_since < self.cooldown_after_stop_days:
                                self._reject(date_idx, sym, "冷却期",
                                             f"距止损{days_since}天 (<{self.cooldown_after_stop_days})")
                                continue

                    # --- risk: sector weight ---
                    if self.max_sector_weight > 0:
                        sector = self.sector_map.get(sym, "Unknown")
                        sector_exposure = sum(
                            s["position"] * s.get("last_price", 0)
                            for j, s in leg_state.items()
                            if self.sector_map.get(s["leg"].symbol, "Unknown") == sector
                        )
                        alloc = self._allocate(st["leg"], cash, len(leg_data), current_equity)
                        entry_cost = price * (1 + self.slippage_pct) * (1 + self.commission_rate)
                        new_exposure = (sector_exposure + (alloc if alloc > 0 else 0)) / max(current_equity, 1)
                        if new_exposure > self.max_sector_weight:
                            self._reject(date_idx, sym, "行业权重",
                                         f"{sector}敞口{new_exposure*100:.0f}% > {self.max_sector_weight*100:.0f}%")
                            continue

                    # Portfolio risk: max_daily_new_positions
                    if self.max_daily_new_positions > 0 and daily_new_count >= self.max_daily_new_positions:
                        self._reject(date_idx, sym, "日开仓上限",
                                     f"当日已开{daily_new_count}笔 (≥{self.max_daily_new_positions})")
                        continue

                    alloc = self._allocate(st["leg"], cash, len(leg_data), current_equity)
                    if alloc > 0:
                        if self.sizing_mode == "risk_budget":
                            qty = self._calc_risk_budget_qty(alloc, price, atr)
                        else:
                            qty = st["strategy"].position_size(alloc, price, atr)

                        # Execution constraints
                        qty = self._apply_execution_constraints(qty, row, df_sig, idx_pos)
                        if qty <= 0:
                            continue

                        actual_price = price * (1 + self.slippage_pct)
                        cost = actual_price * qty * (1 + self.commission_rate)

                        # Portfolio risk: max_symbol_weight
                        if self.max_symbol_weight > 0 and current_equity > 0:
                            if cost / current_equity > self.max_symbol_weight:
                                self._reject(date_idx, sym, "标的上限",
                                             f"{sym}占比{cost/current_equity*100:.0f}% > {self.max_symbol_weight*100:.0f}%")
                                continue

                        # Portfolio risk: max_gross_exposure
                        if self.max_gross_exposure > 0 and current_equity > 0:
                            existing_exposure = sum(
                                s["position"] * s.get("last_price", 0) for s in leg_state.values()
                            )
                            if (existing_exposure + cost) / current_equity > self.max_gross_exposure:
                                self._reject(date_idx, sym, "总敞口上限",
                                             f"总敞口{(existing_exposure+cost)/current_equity*100:.0f}% > {self.max_gross_exposure*100:.0f}%")
                                continue

                        if cost <= cash and qty > 0:
                            st["pending_order"] = self.execution_model.make_plan(
                                symbol=sym,
                                side=OrderSide.BUY,
                                quantity=qty,
                                created_index=idx_pos,
                            )
                            daily_new_count += 1

                total_equity += st["position"] * price

            total_equity = cash + sum(
                s["position"] * s.get("last_price", 0) for s in leg_state.values()
            )
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
        if equity_history:
            # Keep result.final_equity and the last equity-curve point on the
            # same liquidation basis used by performance metrics.
            equity_history[-1] = (last_date, final_equity)

        return PortfolioResult(
            equity_history=equity_history,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            legs=self.legs,
            trades=trades,
            rejections=self._rejections,
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
        # Track stop-loss exit for cooldown
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = date_idx
        return cash

    def _apply_open_fill(
        self,
        st: dict,
        fill,
        trades: List[PortfolioTrade],
        open_trade_idx: dict,
        leg_index: int,
        cash: float,
    ) -> bool:
        """Apply a backtest fill to open or add to a portfolio leg."""
        cost = fill.gross_value + fill.commission
        if cost > cash or fill.filled_qty <= 0:
            return False

        old_qty = st["position"]
        st["position"] += fill.filled_qty
        st["highest"] = max(st.get("highest", 0), fill.fill_price)
        st["capital_allocated"] += cost
        if old_qty > 0:
            st["entry_price"] = (
                st["entry_price"] * old_qty + fill.fill_price * fill.filled_qty
            ) / st["position"]
            if leg_index in open_trade_idx:
                t = trades[open_trade_idx[leg_index]]
                t.qty = st["position"]
                t.entry_price = st["entry_price"]
        else:
            st["entry_price"] = fill.fill_price
            trade = PortfolioTrade(
                symbol=st["leg"].symbol,
                entry_time=fill.date,
                qty=fill.filled_qty,
                entry_price=fill.fill_price,
            )
            trades.append(trade)
            open_trade_idx[leg_index] = len(trades) - 1
        return True

    def _apply_close_fill(
        self,
        st: dict,
        fill,
        reason: str,
        open_trade_idx: dict,
        trades: List[PortfolioTrade],
        leg_index: int,
        cash: float,
    ) -> float:
        """Apply a backtest fill to close or reduce a portfolio leg."""
        qty = min(fill.filled_qty, st["position"])
        if qty <= 0:
            return cash
        avg_entry_cost = st["capital_allocated"] / st["position"] if st["position"] > 0 else 0
        entry_cost = avg_entry_cost * qty
        exit_proceeds = fill.fill_price * qty - fill.fill_price * qty * self.commission_rate
        cash += exit_proceeds
        st["position"] -= qty
        st["capital_allocated"] -= entry_cost

        if leg_index in open_trade_idx:
            t = trades[open_trade_idx[leg_index]]
            if st["position"] == 0:
                t.exit_time = fill.date
                t.exit_price = fill.fill_price
                t.pnl = exit_proceeds - entry_cost
                t.pnl_pct = (t.pnl / entry_cost * 100) if entry_cost > 0 else 0
                t.reason = reason
                t.hold_days = (fill.date - t.entry_time).days
                del open_trade_idx[leg_index]
            else:
                t.qty = st["position"]

        if st["position"] == 0:
            st["capital_allocated"] = 0
            st["entry_price"] = 0
            st["highest"] = 0
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = fill.date
        return cash

    def _apply_execution_constraints(
        self, qty: int, row: pd.Series, df_sig: pd.DataFrame, idx_pos: int,
    ) -> int:
        """Enforce lot size and participation rate on *qty*.  Return adjusted qty."""
        if self.lot_size > 0:
            qty = (qty // self.lot_size) * self.lot_size
        if self.max_participation_rate > 0 and qty > 0:
            try:
                vol = float(df_sig.iloc[idx_pos]["Volume"])
                if vol > 0:
                    max_qty = int(vol * self.max_participation_rate)
                    qty = min(qty, max_qty)
            except (KeyError, IndexError, TypeError):
                pass
        return qty

    def _allocate(self, leg: Leg, available_cash: float, num_legs: int,
                  current_equity: float = 0) -> float:
        """Determine how much cash to allocate to a new position."""
        if self.allocation == "equal":
            return min(available_cash, self.initial_capital / max(num_legs, 1))
        elif self.allocation == "dynamic_equal":
            ref = current_equity if current_equity > 0 else self.initial_capital
            return min(available_cash, ref / max(num_legs, 1))
        elif self.allocation == "fraction":
            return available_cash * 0.25
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
    rejections: list = field(default_factory=list)

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

        # Rejection log
        if self.rejections:
            print(f"  --- 风控拦截 ({len(self.rejections)} 次) ---")
            # Group by reason
            by_reason: dict = {}
            for r in self.rejections:
                by_reason.setdefault(r["reason"], []).append(r)
            for reason, items in sorted(by_reason.items()):
                syms = sorted(set(r.get("symbol", "") for r in items))
                print(f"  {reason}: {len(items)}次  {', '.join(syms)}")
            # Show last 5
            print(f"\n  最近拦截:")
            for r in self.rejections[-5:]:
                d = str(r["date"])[:10]
                sym = r.get("symbol", "")
                print(f"    {d} {sym:<6s} {r['reason']}: {r['detail']}")
            print()

    def plot(self, save_path: str = "charts/portfolio_result.png"):
        try:
            setup_chinese_font()
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
