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
| 参数优化 | 网格搜索 + Walk-forward 样本外验证（资金连续传递）+ 热力图 |
| 每日回溯 | 批量扫描 watchlist，信号表格 + 飞书卡片推送 |
| 实盘桥梁 | Broker 抽象接口 + MockBroker（dry-run）+ FutuBroker（富途 OpenD） |
| 风控 | 连续亏损熔断、波动率自适应仓位、单日上限、总敞口、滑点检查、日内亏损上限 |
| Dashboard | Streamlit Web UI — 今日信号、回测图表（含买卖点）、策略对比、组合回测、交易明细 |
| CI | GitHub Actions — push/PR 自动跑 pytest（353 测试） |

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
  engine/            # 回测引擎 (单标/组合/参数优化)
  utils/             # 工具 (日志/飞书通知/环境/配置)
  tests/             # 353 个测试
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

## 飞书通知

支持两种模式：

- **Webhook** — `export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"`
- **App** — 在 `.env` 中配置 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_CHAT_ID`

配置后 `--notify` 即可推送信号卡片和成交通知。

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

在 `strategy/__init__.py` 的 `STRATEGY_MAP` 注册，并在 `optimize.py` 的 `PARAM_GRIDS` 添加搜索空间。

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
