# mytrader

个人量化交易系统 — 数据管线 → 策略库 → 回测 → 参数优化 → 每日扫描 → 实盘桥梁。

[![CI](https://github.com/geelaro/mytrader/actions/workflows/ci.yml/badge.svg)](https://github.com/geelaro/mytrader/actions/workflows/ci.yml)

## 功能

| 模块 | 说明 |
|------|------|
| 数据管线 | 统一 DataProvider，腾讯/新浪/AKShare/YFinance 四源，SQLite 本地缓存 + 增量更新 |
| 策略库 | 10 个策略（趋势/均值回归/动量突破/波动率），BaseStrategy 统一接口 |
| 策略选择层 | `active`（实盘执行）+ `monitor`（观察对比），每日扫描全部 |
| 回测引擎 | 含滑点佣金，退出逻辑收归策略，支持单标 + 组合回测 |
| 仓位管理 | `fixed_capital`（策略自定） / `risk_budget`（ATR 风险预算）双模式 |
| 参数优化 | 网格搜索 + Walk-forward 样本外验证（资金连续传递）+ 热力图 |
| 分析工具 | 成本敏感性网格扫描 + 参数鲁棒性邻域扰动 + 实盘可行性评级 |
| 每日回溯 | 批量扫描 watchlist，信号表格 + 飞书卡片推送 |
| 实盘桥梁 | Broker 抽象接口 + MockBroker（dry-run）+ FutuBroker（富途 OpenD） |
| 风控 | 连续亏损熔断、波动率自适应仓位、单日上限、总敞口、行业权重、止损冷却期、滑点检查、日内亏损上限 |
| 组合回测 | 多标的共享资金池，支持 equal/dynamic_equal 分配，组合级风控（集中度/敞口/行业/日开仓上限） |
| Dashboard | Streamlit Web UI — Tab 分页（单标/组合），回测图表+买卖点，交易明细筛选与收益归因 |
| CI | GitHub Actions — push/PR 自动跑 pytest（399 测试） |

## 策略

| 策略 | 类型 | 描述 |
|------|------|------|
| `enhanced_macd` | 趋势 | 双MA + MACD + RSI + ATR 止损止盈 |
| `trend_follower` | 趋势 | MA + ADX + Chandelier 移动止损 |
| `weekly_macd` | 周线趋势 | MACD 金叉死叉 |
| `weekly_macd_kdj` | 周线趋势 | KDJ 金叉买入 + MACD 死叉卖出 |
| `daily_macd_kdj` | 日线KDJ | KDJ 金叉 + MACD 死叉 |
| `bollinger_mean_reversion` | 均值回归 | 布林下轨 + RSI 超卖回升 |
| `donchian_breakout` | 动量突破 | 唐奇安通道突破 |
| `atr_breakout` | 波动率突破 | MA + N×ATR 突破 |
| `bollinger_squeeze` | 波动率收缩 | BB 带宽低分位 + 突破上轨 |
| `turtle_trading` | 趋势 | 双SMA + 唐奇安通道 + ATR 止损 |

## 项目结构

```
mytrader/
  data/              # 数据管线 (protocol/cache/provider/sources)
  strategy/          # 策略库 (base + 10 个策略)
  broker/            # 券商接口 (base + mock + futu)
  engine/            # 回测引擎
  ├─ trader.py       #   单标回测 (BacktestEngine/Trade/BacktestResult)
  ├─ portfolio.py    #   组合回测 (PortfolioBacktest/PortfolioTrade)
  └─ optimize.py     #   参数优化 (grid_search/walk_forward)
  analysis/          # 分析工具
  ├─ cost_sensitivity.py    #   成本敏感性网格扫描
  └─ param_robustness.py    #   参数鲁棒性邻域扰动
  utils/             # 工具 (日志/飞书通知/环境/配置/行业映射)
  tests/             # 399 个测试
  reports/           # 自动生成的 CSV + PNG 报告
  daily.py           # 每日回溯扫描 (入口脚本)
  live_trader.py     # 实盘信号执行 + 风控 (入口脚本)
  dashboard.py       # Streamlit Web 仪表盘 (入口脚本)
  config.py          # 统一运行时配置
  watchlist.toml     # 标的 + 策略 + 风控 + 日志配置
```

## 快速开始

```bash
# Python 3.10+
pip install pipenv
pipenv install --dev
```

## 使用

```bash
# 每日扫描
pipenv run python daily.py                        # 今天信号
pipenv run python daily.py --notify               # 推送到飞书
pipenv run python daily.py --history --days 7     # 近 7 天历史

# 参数优化
pipenv run python engine/optimize.py -s trend_follower -symbol AAPL --top 10
pipenv run python engine/optimize.py -s weekly_macd --walk-forward

# 组合回测
pipenv run python engine/portfolio.py

# Dashboard
pipenv run streamlit run dashboard.py

# 实盘（模拟模式）
pipenv run python live_trader.py --dry-run
pipenv run python live_trader.py --notify

# 测试
pipenv run python -m pytest tests/ -v
```

## 仓位管理

两种仓位模式，通过 `sizing_mode` 切换：

### fixed_capital（默认）

策略自行决定仓位大小（通常为可用资金的 95%）：

```python
from engine.trader import run_backtest
from strategy import WeeklyMACD_KDJ

result, _ = run_backtest("AAPL", "2020-01-01", strategy_cls=WeeklyMACD_KDJ)
```

### risk_budget（风险预算）

引擎统一按 `风险金额 / 止损距离` 计算仓位：

```
qty = capital × risk_per_trade / (ATR × risk_atr_mult)
```

```python
result, _ = run_backtest(
    "TSLA", "2020-01-01", strategy_cls=WeeklyMACD_KDJ,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,   # 单笔风险 2%
    risk_atr_mult=2.0,     # 止损 = 2×ATR
)
```

| 模式 | 收益特征 | 回撤特征 | 适用场景 |
|------|:---:|:---:|------|
| fixed_capital | 高（满仓复利） | 高（TSLA -56%） | 回测研究，了解策略上限 |
| risk_budget 2% | 中 | 可控（<10%） | 实盘稳健执行 |

## 风控增强

组合回测支持完整的风控层级：

```python
from engine.portfolio import PortfolioBacktest, Leg
from utils.sectors import DEFAULT_SECTORS

bt = PortfolioBacktest(
    legs=[Leg("AAPL", "weekly_macd_kdj"), Leg("NVDA", "weekly_macd_kdj"),
          Leg("TSLA", "weekly_macd_kdj"), Leg("SPY", "turtle_trading")],
    initial_capital=100000,
    allocation="dynamic_equal",
    # --- 仓位 ---
    sizing_mode="risk_budget",
    risk_per_trade=0.02,
    # --- 组合风控 ---
    max_symbol_weight=0.25,           # 单标的上限 25%
    max_sector_weight=0.30,           # 单行业上限 30%
    max_gross_exposure=0.80,          # 总敞口 ≤ 80%
    max_daily_new_positions=3,        # 单日最大新开仓
    cooldown_after_stop_days=10,      # 止损后 10 天内禁止重入
    # --- 成交约束 ---
    lot_size=0,                       # 整手取整，0=不限制
    max_participation_rate=0.01,      # 单笔 ≤ 1% 成交量
    sector_map=DEFAULT_SECTORS,       # 行业分类映射
)
result = bt.run(start="2020-01-01")
result.summary()
```

被风控拦截的信号会记录到 `rejections` 列表并在 `summary()` 中展示：

```
--- 风控拦截 (12 次) ---
  行业权重: 9次  AAPL, GOOGL, NVDA
  冷却期: 1次  AAPL
  标的上限: 2次  AAPL

  最近拦截:
    2024-04-12 AAPL   行业权重: Technology敞口32% > 30%
    2025-09-19 AAPL   冷却期: 距止损9天 (<10)
```

## 分析工具

### 成本敏感性

扫描佣金 × 滑点网格，评估策略在不同交易成本下的表现：

```bash
pipenv run python analysis/cost_sensitivity.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/cost_sensitivity.py --sizing-mode risk_budget --risk-per-trade 0.01
```

输出：`reports/cost_sensitivity_<strategy>_<symbol>.csv` + `.png`（双面板热力图：收益率 + 夏普），附带 A+ ~ D- 实盘可行性评级。

### 参数鲁棒性

IS 寻优 → 邻域 ±10%/±20% 扰动 → OOS 分布统计 → ROBUST/STABLE/SENSITIVE/OVERFIT 评级：

```bash
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/param_robustness.py -s enhanced_macd --sizing-mode risk_budget
```

输出：`reports/param_robustness_<strategy>_<symbol>.csv` + `.png`（箱线图：各参数 OOS 收益分布 + 敏感度排序）。

## 飞书通知

支持两种模式：

- **Webhook** — `export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"`
- **App** — 在 `.env` 中配置 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_CHAT_ID`

配置后 `--notify` 即可推送信号卡片和成交通知。

## 配置系统

`config.py` 提供统一的运行时配置，支持层级覆盖：

```
默认值 (config.py) → config.yaml (可选) → 环境变量 (.env)
```

```python
from config import config

print(config.risk.max_position_pct)  # 0.3
print(config.feishu.app_id)          # 从环境变量读取
```

创建 `config.yaml` 可覆盖任意默认值：

```yaml
log:
  level: DEBUG
trading:
  daemon_interval_minutes: 10
```

## 添加新标的

编辑 `watchlist.toml`：

```toml
[[watchlist]]
symbol = "META"
name = "Meta"
active = "trend_follower"        # 实盘执行的策略
monitor = ["weekly_macd_kdj"]    # 观察对比的策略列表
```

## 添加新策略

```python
from strategy.base import BaseStrategy, StrategyParams

class MyParams(StrategyParams):
    ma_period: int = 20

class MyStrategy(BaseStrategy):
    params: MyParams

    def calculate_indicators(self, df):
        df["Signal"] = 0
        df.loc[..., "Signal"] = 1
        return df

    @property
    def min_bars(self) -> int:
        return 20
```

在 `strategy/__init__.py` 的 `STRATEGY_MAP` 注册，并在 `engine/optimize.py` 的 `PARAM_GRIDS` 添加搜索空间。

## 接入真实券商

```python
from broker.base import Broker

class MyBroker(Broker):
    def get_account(self): ...
    def get_positions(self): ...
    def submit_order(self, order): ...
    def cancel_order(self, order_id): ...

trader = LiveTrader(broker=MyBroker())
trader.run()
```

内置 FutuBroker（富途 OpenD）可直接使用：

```python
from broker.futu import FutuBroker
trader = LiveTrader(broker=FutuBroker(host="127.0.0.1", port=11111))
```
