"""Monte Carlo simulation — shuffle trade sequences to test drawdown sensitivity.

Usage:
    pipenv run python analysis/monte_carlo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.portfolio import PortfolioBacktest, Leg


def run():
    symbols = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY"]
    start, end = "2018-01-01", "2026-05-19"

    legs = [Leg(sym, "weekly_macd_kdj") for sym in symbols]
    pf = PortfolioBacktest(legs, initial_capital=100000)
    result = pf.run(start=start, end=end)

    # Use trade PnL as % of initial capital (realistic portfolio drawdown)
    initial_capital = result.initial_capital
    pnl_pcts = np.array([
        t.pnl / initial_capital * 100 for t in result.trades if t.pnl is not None
    ])
    if len(pnl_pcts) == 0:
        print("无交易数据")
        return

    print(f"交易总数: {len(pnl_pcts)}")
    win_rate = (pnl_pcts > 0).sum() / len(pnl_pcts) * 100
    print(f"胜率: {win_rate:.1f}%")
    print(f"平均单笔盈亏: {pnl_pcts.mean():+.2f}% (占初始资金)")
    print(f"单笔标准差: {pnl_pcts.std():.2f}%")

    # Monte Carlo: shuffle trade order 2000 times, track drawdowns
    n_sims = 2000
    max_dds = []
    end_equities = []

    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled = rng.permutation(pnl_pcts)
        equity = 100.0
        peak = 100.0
        max_dd = 0.0
        for pnl in shuffled:
            equity *= (1 + pnl / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        max_dds.append(max_dd)
        end_equities.append(equity)

    max_dds = np.array(max_dds)
    end_equities = np.array(end_equities)
    final_return = (end_equities.mean() - 100)

    print(f"\nMonte Carlo ({n_sims} 次打乱):")
    print(f"  预期最终收益: {final_return:+.1f}%")
    print(f"  最大回撤分布:")
    for pct in [5, 25, 50, 75, 95]:
        d = np.percentile(max_dds, pct)
        print(f"    P{pct:02d}: {d:.1f}%")

    # Probability of hitting various drawdown levels
    print(f"\n  回撤概率:")
    for threshold in [20, 30, 40, 50, 60]:
        prob = (max_dds >= threshold).sum() / n_sims * 100
        print(f"    回撤 ≥ {threshold}%:  {prob:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_dds, bins=40, color="#d62728", edgecolor="white", alpha=0.8)
    ax.axvline(max_dds.mean(), color="blue", linestyle="--", linewidth=1.5,
               label=f"均值 {max_dds.mean():.1f}%")
    ax.axvline(np.percentile(max_dds, 95), color="orange", linestyle="--", linewidth=1.5,
               label=f"P95 {np.percentile(max_dds, 95):.1f}%")
    ax.set_xlabel("最大回撤 %")
    ax.set_ylabel("频次")
    ax.set_title(f"Monte Carlo 最大回撤分布 ({n_sims}次打乱交易序列)", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = Path("reports/monte_carlo.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"\n图表已保存: {path}")


if __name__ == "__main__":
    run()
