# mytrader

个人量化交易系统 — 数据 → 回测 → 优化 → 扫描 → 实盘。

## 功能

| 模块 | 说明 |
|------|------|
| 数据管线 | 统一 DataProvider，腾讯/新浪/AKShare/YFinance 四源，SQLite 本地缓存 + 增量更新 |
| 策略框架 | BaseStrategy 统一接口，4 个内置策略（增强MACD / 趋势跟踪 / 周线MACD / 周线KDJ+MACD） |
| 回测引擎 | 含滑点佣金，退出逻辑收归策略，支持单标 + 组合回测 |
| 参数优化 | 网格搜索 + Walk-forward 样本外验证 + 热力图 |
| 每日回溯 | 批量扫描 watchlist，信号表格输出，推送到飞书 |
| 实盘桥梁 | Broker 抽象接口 + MockBroker + LiveTrader 风控执行 |

## 项目结构

```
mytrader/
  data/           # 数据管线 (protocol/cache/provider/sources)
  strategy/       # 策略库 (base + 4 个策略)
  broker/         # 券商接口 (base + mock)
  utils/          # 工具 (日志/飞书通知/环境)
  tests/          # 100 个测试
  trader.py       # 单标回测引擎
  daily.py        # 每日回溯扫描
  optimize.py     # 参数优化
  portfolio.py    # 组合回测
  live_trader.py  # 实盘信号执行
  app.py          # 批量回测脚本
  watchlist.toml  # 标的 + 策略 + 风控 + 日志配置
```

## 快速开始

```bash
# Python 3.10+
pip install pipenv
pipenv install
```

## 使用

```bash
# 每日扫描
pipenv run python daily.py                        # 今天信号
pipenv run python daily.py --notify               # 推送到飞书
pipenv run python daily.py --history --days 7     # 近 7 天历史

# 回测
pipenv run python app.py                          # 批量回测全部标的

# 参数优化
pipenv run python optimize.py --strategy trend_follower --symbol AAPL --top 10
pipenv run python optimize.py --strategy weekly_macd --walk-forward

# 组合回测
pipenv run python portfolio.py

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

配置好之后 `--notify` 即可推送信号卡片和成交通知。

## 添加新标的

编辑 `watchlist.toml`：

```toml
[[watchlist]]
symbol = "META"
name = "Meta"
strategies = ["trend_follower", "weekly_macd"]
```

## 添加新策略

```python
from strategy.base import BaseStrategy, StrategyParams

class MyStrategy(BaseStrategy):
    params: MyParams
    
    def calculate_indicators(self, df):
        df["Signal"] = 0
        df.loc[..., "Signal"] = 1  # entry
        return df
    
    @property
    def min_bars(self) -> int:
        return 20
```

然后在 `strategy/__init__.py` 的 `STRATEGY_MAP` 注册即可。

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
