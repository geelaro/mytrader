import pandas as pd
import numpy as np
import time
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

from data import DataProvider

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_days: int = 0

    def __post_init__(self):
        if self.entry_date and self.exit_date:
            self.holding_days = (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    buy_hold_return_pct: float
    initial_capital: float
    final_equity: float


# ---------------------------------------------------------------------------
# Database manager (preserved from original)
# ---------------------------------------------------------------------------

class DatabaseManager:
    def __init__(self, db_path="trading_data.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS market_data (
            timestamp TEXT, symbol TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, PRIMARY KEY (timestamp, symbol))''')
        c.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
            timestamp TEXT PRIMARY KEY, symbol TEXT, signal INTEGER,
            price REAL, position INTEGER, macd REAL, macd_signal REAL, macd_hist REAL)''')
        conn.commit()
        conn.close()

    def save_market_data(self, df, symbol):
        conn = sqlite3.connect(str(self.db_path))
        try:
            data = df.reset_index()
            data['Date'] = data['Date'].astype(str)
            data['symbol'] = symbol
            data.to_sql('market_data', conn, if_exists='append', index=False)
        finally:
            conn.close()

    def get_market_data(self, symbol, days=365):
        conn = sqlite3.connect(str(self.db_path))
        try:
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM market_data "
                "WHERE symbol=? AND timestamp >= date('now', ?) ORDER BY timestamp ASC",
                conn, params=(symbol, f'-{days} days'))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        finally:
            conn.close()

    def save_trade_log(self, timestamp, symbol, signal, price, position,
                       macd, macd_signal, macd_hist):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.cursor().execute(
                "INSERT INTO trade_logs VALUES (?,?,?,?,?,?,?,?)",
                (timestamp, symbol, signal, price, position, macd, macd_signal, macd_hist))
            conn.commit()
        finally:
            conn.close()

    def get_trade_history(self, symbol, days=30):
        conn = sqlite3.connect(str(self.db_path))
        try:
            return pd.read_sql_query(
                "SELECT * FROM trade_logs WHERE symbol=? AND "
                "timestamp >= datetime('now', ?) ORDER BY timestamp DESC",
                conn, params=(symbol, f'-{days} days'))
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulated trading environment with slippage and commission."""

    def __init__(self, initial_capital=10000, commission_rate=0.0003, slippage_pct=0.0001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.current_entry: Optional[Dict] = None
        self._current_price = 0.0

    @property
    def equity(self):
        return self.cash + self.position * self._current_price

    def buy(self, date, price, quantity):
        if quantity <= 0 or price <= 0:
            return False
        actual_price = price * (1 + self.slippage_pct)
        total_cost = actual_price * quantity * (1 + self.commission_rate)
        if total_cost > self.cash:
            quantity = int(self.cash / (actual_price * (1 + self.commission_rate)))
            if quantity <= 0:
                return False
            total_cost = actual_price * quantity * (1 + self.commission_rate)
        self.cash -= total_cost
        self.position += quantity
        self.current_entry = {'date': date, 'price': actual_price, 'quantity': quantity}
        return True

    def sell(self, date, price, quantity=None, reason='signal'):
        if quantity is None:
            quantity = self.position
        quantity = min(quantity, self.position)
        if quantity <= 0 or self.position <= 0:
            return None
        actual_price = price * (1 - self.slippage_pct)
        proceeds = actual_price * quantity * (1 - self.commission_rate)
        self.cash += proceeds
        self.position -= quantity

        trade = None
        if self.current_entry and self.current_entry['quantity'] > 0:
            e = self.current_entry
            trade = Trade(
                entry_date=e['date'], exit_date=date,
                entry_price=e['price'], exit_price=actual_price,
                quantity=quantity,
                pnl=(actual_price - e['price']) * quantity,
                pnl_pct=(actual_price / e['price'] - 1) * 100,
                exit_reason=reason)
            self.trades.append(trade)
            if self.position == 0:
                self.current_entry = None
            else:
                self.current_entry = {**e, 'quantity': e['quantity'] - quantity}
        return trade

    def update(self, date, price):
        self._current_price = price
        self.equity_history.append((date, self.equity))

    def get_result(self, benchmark_returns=None):
        if not self.equity_history:
            raise ValueError("No equity history recorded")
        eq_df = pd.DataFrame(self.equity_history, columns=['date', 'equity']).set_index('date')
        curve = eq_df['equity']
        rets = curve.pct_change().dropna()

        final = curve.iloc[-1]
        total_ret = (final / self.initial_capital - 1) * 100
        days = (curve.index[-1] - curve.index[0]).days
        cagr = ((final / self.initial_capital) ** (365.25 / days) - 1) * 100 if days > 0 else 0
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0

        rolling_max = curve.expanding().max()
        dd = (curve - rolling_max) / rolling_max * 100
        max_dd = dd.min()

        trades = self.trades
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = pf = 0

        bh_ret = ((1 + benchmark_returns).prod() - 1) * 100 if benchmark_returns is not None else 0

        return BacktestResult(
            trades=trades, equity_curve=curve,
            total_return_pct=total_ret, cagr_pct=cagr, sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd, win_rate_pct=win_rate, profit_factor=pf,
            avg_win_pct=avg_win, avg_loss_pct=avg_loss,
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl <= 0]),
            buy_hold_return_pct=bh_ret,
            initial_capital=self.initial_capital, final_equity=final)


from strategy import (
    BaseStrategy,
    EnhancedMACDStrategy,
    TrendFollower,
    WeeklyMACD,
    WeeklyMACD_KDJ,
)


# ---------------------------------------------------------------------------
# Backtest runner & visualization
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(symbol, start, end, seed=42):
    """Generate realistic synthetic OHLCV data when all data sources are unavailable."""
    np.random.seed(seed)
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    dates = pd.bdate_range(start=start_date, end=end_date)

    base_price = 130.0 if symbol == "AAPL" else 100.0
    n = len(dates)
    daily_ret = np.random.normal(0.0003, 0.018, n)
    prices = base_price * np.cumprod(1 + daily_ret)

    data = []
    for d, close in zip(dates, prices):
        daily_range = close * np.random.uniform(0.01, 0.04)
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        open_p = low + np.random.uniform(0, high - low)
        volume = int(np.random.uniform(30_000_000, 120_000_000))
        data.append({
            'Date': d, 'Open': round(open_p, 2), 'High': round(high, 2),
            'Low': round(low, 2), 'Close': round(close, 2), 'Volume': volume
        })

    df = pd.DataFrame(data).set_index('Date')
    print(f"  注意: 使用模拟数据 ({symbol}, {n} 根K线)，所有数据源均不可用")
    return df


def _fetch_tencent(symbol, start, end):
    """Fetch US stock OHLCV data from Tencent (ifzq.gtimg.cn).

    Data format per row: [date, open, close, high, low, volume]
    """
    import requests
    import json

    code_map = {
        "AAPL": "usAAPL.OQ",
        "MSFT": "usMSFT.OQ",
        "GOOGL": "usGOOGL.OQ",
        "AMZN": "usAMZN.OQ",
        "TSLA": "usTSLA.OQ",
        "NVDA": "usNVDA.OQ",
        "META": "usMETA.OQ",
        "QQQ": "usQQQ.OQ",
        "SPY": "usSPY.AM",
    }
    tencent_code = code_map.get(symbol.upper(), f"us{symbol.upper()}.OQ")

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    # Use empty end date — API rejects future/non-trading dates
    # Max count is ~2000, errors on larger values
    params = {"param": f"{tencent_code},day,{start},,2000,qfq"}

    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    r = s.get(url, params=params, timeout=30)
    data = r.json()

    if data.get("code") != 0 or not isinstance(data.get("data"), dict):
        raise RuntimeError(f"Tencent API returned error: {data.get('msg', 'unknown')}")

    stock_data = data["data"].get(tencent_code)
    if not isinstance(stock_data, dict):
        raise RuntimeError(f"Tencent API unexpected data format")

    raw_rows = stock_data.get("day", [])
    if not raw_rows:
        raise RuntimeError("Tencent API returned empty data")

    rows = []
    for row in raw_rows:
        if len(row) < 6:
            continue
        date_str, open_, close_, high_, low_, vol_ = row[:6]
        rows.append({
            "Date": pd.Timestamp(date_str),
            "Open": float(open_),
            "High": float(high_),
            "Low": float(low_),
            "Close": float(close_),
            "Volume": float(vol_),
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()

    # Manual split adjustment — Tencent API does not adjust US stock splits
    # Format: (split_date, ratio) — prices before date are divided by ratio
    splits = {
        "AAPL":  [("2020-08-31", 4)],
        "NVDA":  [("2021-07-20", 4), ("2024-06-10", 10)],
        "TSLA":  [("2020-08-31", 5), ("2022-08-25", 3)],
        "AMZN":  [("2022-06-06", 20)],
        "GOOGL": [("2022-07-18", 20)],
    }
    if symbol.upper() in splits:
        for split_date, ratio in splits[symbol.upper()]:
            split_dt = pd.Timestamp(split_date)
            pre_split = df.index < split_dt
            for col in ['Open', 'High', 'Low', 'Close']:
                df.loc[pre_split, col] = df.loc[pre_split, col] / ratio
            df.loc[pre_split, 'Volume'] = df.loc[pre_split, 'Volume'] * ratio

    # Filter to requested date range
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df


def _fetch_sina(symbol, start, end):
    """Fetch Chinese A-share/ETF OHLCV from Sina Finance.

    Symbol format: 'sh510300' (Shanghai) or 'sz159919' (Shenzhen).
    """
    import requests
    import json

    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": symbol, "scale": "240", "ma": "no", "datalen": "2000"}

    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    r = s.get(url, params=params, timeout=30)
    data = json.loads(r.text)

    if not data or not isinstance(data, list):
        raise RuntimeError(f"Sina API returned unexpected data")

    rows = []
    for row in data:
        rows.append({
            "Date": pd.Timestamp(row["day"]),
            "Open": float(row["open"]),
            "High": float(row["high"]),
            "Low": float(row["low"]),
            "Close": float(row["close"]),
            "Volume": float(row["volume"]),
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df


def run_backtest(symbol="AAPL", start="2020-01-01", end=None,
                 initial_capital=10000, strategy_cls=None, **strategy_params):
    """Run a full backtest and return results + dataframe.

    Args:
        strategy_cls: Strategy class (default: EnhancedMACDStrategy).
                      Use TrendFollower for Chandelier exit strategy.
    """
    if strategy_cls is None:
        strategy_cls = EnhancedMACDStrategy
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"获取 {symbol} 数据 ({start} ~ {end}) ...")
    provider = DataProvider()
    df = provider.get_daily(symbol, start=start, end=end)
    if df is None or df.empty:
        print(f"  !! 未能获取 {symbol} 数据, 使用模拟数据")
        df = _synthetic_ohlcv(symbol, start, end)
    else:
        print(f"  OK: {len(df)} 根K线")

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    strategy = strategy_cls(**strategy_params)
    df = strategy.calculate_indicators(df)
    engine = BacktestEngine(initial_capital=initial_capital)

    highest_since_entry = 0

    for i in range(strategy.min_bars, len(df)):
        date = df.index[i]
        price = float(df['Close'].iloc[i])
        atr = float(df['ATR'].iloc[i])

        if engine.position > 0 and engine.current_entry:
            if price > highest_since_entry:
                highest_since_entry = price

            exit_now, reason = strategy.check_exit(
                df, i,
                entry_price=engine.current_entry['price'],
                highest_since_entry=highest_since_entry,
                position=engine.current_entry,
            )
            if exit_now:
                engine.sell(date, price, reason=reason)

        elif strategy.entry_signal(df, i) and engine.position == 0:
            qty = strategy.position_size(engine.cash, price, atr)
            if qty > 0:
                engine.buy(date, price, qty)
                highest_since_entry = price

        engine.update(date, price)

    # Close any open position at end
    if engine.position > 0:
        last_price = float(df['Close'].iloc[-1])
        engine.sell(df.index[-1], last_price, reason='回测结束')
        engine.update(df.index[-1], last_price)

    benchmark_rets = df['Close'].pct_change().dropna()
    # Align benchmark returns to backtest period
    if len(engine.equity_history) > 0:
        eq_dates = pd.DataFrame(engine.equity_history, columns=['date', 'equity']).set_index('date')
        benchmark_rets = benchmark_rets[benchmark_rets.index >= eq_dates.index[0]]

    result = engine.get_result(benchmark_rets)
    return result, df


def print_result(result):
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  回测结果")
    print("=" * 60)
    print(f"  初始资金:      ${result.initial_capital:,.2f}")
    print(f"  最终权益:      ${result.final_equity:,.2f}")
    print(f"  总收益率:      {result.total_return_pct:+.2f}%")
    print(f"  年化收益率:    {result.cagr_pct:+.2f}%")
    print(f"  买入持有:      {result.buy_hold_return_pct:+.2f}%")
    print(f"  夏普比率:      {result.sharpe_ratio:.2f}")
    print(f"  最大回撤:      {result.max_drawdown_pct:.2f}%")
    print(f"  交易次数:      {result.total_trades}")
    print(f"  胜率:          {result.win_rate_pct:.1f}% ({result.winning_trades}胜/{result.losing_trades}负)")
    print(f"  盈亏比:        {result.profit_factor:.2f}")
    print(f"  平均盈利:      {result.avg_win_pct:+.2f}%")
    print(f"  平均亏损:      {result.avg_loss_pct:+.2f}%")

    alpha = result.total_return_pct - result.buy_hold_return_pct
    print(f"\n  超额收益:      {alpha:+.2f}%")
    print("=" * 60)

    if result.trades:
        print("\n  最近 10 笔交易:")
        print(f"  {'入场':<12} {'出场':<12} {'持仓天':<8} {'盈亏%':<10} {'原因':<10}")
        print("  " + "-" * 52)
        for t in result.trades[-10:]:
            print(f"  {t.entry_date.strftime('%Y-%m-%d'):<12} "
                  f"{t.exit_date.strftime('%Y-%m-%d'):<12} "
                  f"{t.holding_days:<8} "
                  f"{t.pnl_pct:+.2f}%{'':<4} "
                  f"{t.exit_reason:<10}")


def plot_result(result, df, symbol="AAPL", save_path=None):
    """Plot backtest results: price + trades, equity curve, drawdown."""
    try:
        for font in ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                break
            except Exception:
                continue
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # --- Panel 1: Price + trades ---
    ax1 = axes[0]
    price_data = df['Close']
    # Align to backtest period
    if not result.equity_curve.empty:
        price_data = price_data[price_data.index >= result.equity_curve.index[0]]
    ax1.plot(price_data.index, price_data, color='#1f77b4', linewidth=0.8, label=f'{symbol} Close')

    # Mark buy/sell points
    for t in result.trades:
        if t.pnl > 0:
            color = 'green'
            marker = '^'
        else:
            color = 'red'
            marker = 'v'
        ax1.scatter(t.entry_date, t.entry_price, c=color, marker=marker, s=60, zorder=5, alpha=0.8)
        ax1.scatter(t.exit_date, t.exit_price, c=color, marker='o', s=40, zorder=5, alpha=0.6)

    # Add dummy points for legend
    ax1.scatter([], [], c='green', marker='^', s=60, label='入场 (盈)')
    ax1.scatter([], [], c='red', marker='v', s=60, label='入场 (亏)')
    ax1.scatter([], [], c='green', marker='o', s=40, label='出场 (盈)')
    ax1.scatter([], [], c='red', marker='o', s=40, label='出场 (亏)')

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} 策略回测 — 交易信号', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Equity curve ---
    ax2 = axes[1]
    if not result.equity_curve.empty:
        eq = result.equity_curve
        ax2.plot(eq.index, eq, color='#2ca02c', linewidth=1.2, label='策略权益')

        # Buy & hold benchmark
        if result.buy_hold_return_pct != 0:
            bh_start = result.initial_capital
            bh_curve = bh_start * (price_data / price_data.iloc[0])
            ax2.plot(bh_curve.index, bh_curve, color='#d62728', linewidth=0.8,
                     linestyle='--', alpha=0.7, label='买入持有')

        ax2.axhline(y=result.initial_capital, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax2.set_ylabel('Equity ($)')
        ax2.set_title('权益曲线', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # --- Panel 3: Drawdown ---
    ax3 = axes[2]
    if not result.equity_curve.empty:
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.4, label='回撤')
        ax3.plot(drawdown.index, drawdown, color='#d62728', linewidth=0.6)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('回撤', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    run_strategy()
