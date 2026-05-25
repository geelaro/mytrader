# mytrader

个人量化交易系统 — 数据管线 → 策略库 → 回测 → 参数优化 → 每日扫描 → 实盘桥梁。

[![CI](https://github.com/geelaro/mytrader/actions/workflows/ci.yml/badge.svg)](https://github.com/geelaro/mytrader/actions/workflows/ci.yml)

## 功能

| 模块 | 说明 |
|------|------|
| 数据管线 | 统一 DataProvider，腾讯/新浪/AKShare/YFinance 四源，SQLite 本地缓存 + 增量更新 |
| 策略库 | 9 个活跃策略（趋势/动量/波动率/宏观过滤/多空），BaseStrategy 统一接口 |
| 多空支持 | 有符号持仓引擎，做空/平空完整链路，mock/live broker 全面适配 |
| MTF 框架 | 多时间框架接口 `calculate_indicators(df, df_weekly)`，周线指标映射日线执行 |
| 策略组合 | StrategyEnsemble 加权投票，MarketRegime 自适应权重分配 |
| 策略门控 | SignalGate：市场状态感知 + 风控暂停 + 敞口检查，`active`/`monitor` 双轨 |
| 回测引擎 | 含滑点佣金，有符号持仓，单标 + 组合回测 |
| 仓位管理 | `fixed_capital` / `risk_budget` 双模式，回测实盘共用 `utils/sizing.py` |
| 参数优化 | 网格搜索 + Walk-forward 样本外验证 + 滚动优化自动更新 watchlist.toml |
| 分析工具 | 成本敏感性 + 参数鲁棒性 + Monte Carlo 模拟 + 滚动窗口 α 衰减 + 压力测试 |
| 每日回溯 | 批量扫描 watchlist，信号表格 + 飞书卡片推送 + 每日报告 |
| 实盘桥梁 | Broker 抽象接口 + MockBroker（dry-run）+ FutuBroker（富途 OpenD） |
| 风控 | 连续亏损熔断、波动率自适应仓位、单日上限、总敞口、行业权重、止损冷却期、滑点检查 |
| 风控持久化 | risk_state + entry_prices 表，守护进程重启后恢复熔断/日内计数/入场价 |
| 订单管理 | 部分成交轮询、限价单超时撤单、滑点统计、批量等风险分配 |
| 组合回测 | 多标的共享资金池，组合级风控（集中度/敞口/行业/日开仓上限） |
| 运维可观测 | ops_log 统一运维表，schema_version 版本化迁移，结构化 JSON 日志，/health HTTP 端点 |
| 数据质量 | quality.py — 缺失值/异常跳变/停牌标记 + validate_ohlcv 前置检查 |
| Dashboard | Streamlit Web UI — 市场状态 → 今日信号 → 策略分布 → 行业饼图 → Monte Carlo 风控 |
| CI | GitHub Actions — push/PR 自动跑 pytest（637 测试 + 黄金样本回归）|
| Docker | docker-compose.yml 一键部署 mytrader + futu-opend + HEALTHCHECK |

## 策略

| 策略 | 类型 | 方向 | 推荐度 | 描述 |
|------|------|:---:|:---:|------|
| `spy_ma_breakout` | 宏观 + 突破 | 纯多 | ★★★ | SPY MA200 宏观过滤 + MA200 趋势 + 20日新高突破 + ATR 动态止盈 + MA 止损（长线） |
| `weekly_macd_kdj` | 周线趋势 | 纯多 | ★★★ | KDJ 金叉买入 + MACD 死叉卖出（主力）|
| `trend_follower` | 短线趋势 | 纯多 | ★★★ | SMA5/20 + ADX + Chandelier 尾随止损（短线优化版）|
| `turtle_trading` | 趋势 | 多空 | ★★☆ | 大哥2.2 递归 SMA + 唐奇安通道 + ATR 固定止损 + 趋势过滤 |
| `donchian_breakout` | 动量突破 | 多空 | ★★ | 唐奇安通道突破 + 移动止损 |
| `daily_macd_kdj` | 日线 KDJ | 纯多 | ★★ | 日线 KDJ 金叉 + MACD 死叉 + ATR 止损 |
| `atr_breakout` | 波动率突破 | 纯多 | ★★ | MA + N×ATR 突破 + 移动止损 |
| `weekly_macd` | 周线趋势 | 纯多 | ★ | MACD 金叉死叉 |
| `macd_kdj` | 日/周 KDJ | 纯多 | ★ | 统一 MACDKDJStrategy，freq="W"/"D" + use_atr_stop |

> 推荐度基于 2026-05 全策略鲁棒性扫描 + 多标的 6 年回测
>
> ~~`enhanced_macd` `bollinger_mean_reversion` `bollinger_squeeze`~~ — 已从活跃策略移除（过拟合/零交易），源文件保留于 `strategy/` 供测试兼容

### 策略参数速查

| 策略 | 关键参数 | 入场条件 | 出场条件 |
|------|---------|---------|---------|
| `spy_ma_breakout` | ma=200, high=20, tp_ATR=4.0 | SPY>MA200 + Close>MA200 + 20日新高 | 止盈: entry+ATR×4 / 止损: Close<MA |
| `trend_follower` | short=5, long=20, ADX_th=15, trail=2.5 | SMA5>SMA20 + ADX>15 + +DI>-DI | 尾随止损: Close<=最高价-ATR×2.5 |
| `turtle_trading` | short=20, long=50, channel=20, trail=3.0 | 递归SMA交叉 + 通道突破(前根) + 趋势过滤 | 入场ATR固定: Close<=最高High-ATR_entry×3.0 |
| `weekly_macd_kdj` | n=9, k=3, d=3 | KDJ金叉(周线) | MACD死叉(周线) |
| `daily_macd_kdj` | n=9, k=3, d=3, trail=3.0 | KDJ金叉(日线) | MACD死叉 + ATR尾随止损 |

## 项目结构

```
mytrader/
  data/              # 数据管线 (protocol/cache/provider/sources/quality/splits.json)
  strategy/          # 策略库 (base + 9 个活跃策略 + 3 个已弃用 + ensemble)
  broker/            # 券商接口 (base + mock + futu)
  engine/            # 回测引擎
  ├─ trader.py       #   单标回测 (BacktestEngine/Trade/BacktestResult) — 有符号持仓
  ├─ portfolio.py    #   组合回测 (PortfolioBacktest/PortfolioTrade)
  ├─ execution.py    #   执行模型 (回测/实盘共用的订单执行语义)
  └─ optimize.py     #   参数优化 (grid_search/walk_forward)
  analysis/          # 分析工具
  ├─ cost_sensitivity.py    #   成本敏感性网格扫描
  ├─ param_robustness.py    #   参数鲁棒性邻域扰动
  ├─ monte_carlo.py         #   Monte Carlo 模拟（交易序列随机化）
  ├─ rolling_alpha.py       #   滚动窗口 α 衰减检测
  └─ stress_test.py         #   极端行情压力测试 (2008/2020/2022)
  live/              # 实盘交易组件
  ├─ risk_controller.py     #   风控检查、仓位计算、熔断持久化
  └─ order_manager.py       #   信号→订单生成（4路多空矩阵）、批量等风险分配
  utils/             # 工具
  ├─ signal_gate.py         #   策略门控层（市场状态+风控+敞口+孤儿守卫）
  ├─ market_state.py        #   四象限市场状态分类器（SPY MA20/50/200 + ADX25）
  ├─ notify.py              #   飞书通知 (Webhook/App + daily_card PnL归因)
  ├─ signal_scanner.py      #   共享信号扫描引擎（MTF 跨频率）
  ├─ logging.py             #   结构化 JSON 日志 (JsonFormatter)
  ├─ sizing.py              #   统一仓位计算（回测实盘共用）
  ├─ sectors.py             #   行业分类映射
  ├─ risk.py                #   RiskLimits 数据类
  └─ metrics.py             #   回撤统计、敞口重构
  tests/             # 637 个测试 (含黄金样本回归 test_golden.py)
  reports/           # 自动生成的 CSV + PNG 报告
  live_trader.py     # 实盘信号执行 + 风控 + HTTP /health (入口脚本)
  daily.py           # 每日回溯扫描 + --optimize 滚动优化 (入口脚本)
  dashboard.py       # Streamlit Web 仪表盘 (入口脚本)
  config.py          # 统一运行时配置
  watchlist.toml     # 标的 + 策略 + 风控 + 市场状态 + 孤儿持仓配置
  docker-compose.yml # Docker Compose 一键部署 (mytrader + futu-opend)
```

## 快速开始

```bash
# Python 3.10+
pip install pipenv
pipenv install --dev
pipenv run pytest tests/ -v   # 637 tests, verify env
```

## 核心流程（30 分钟上手）

### 1. 数据获取 + 回测（5 min）

```bash
pipenv run python -c "
from engine.trader import run_backtest, print_result
from strategy import WeeklyMACD_KDJ
r, _ = run_backtest('AAPL', '2020-01-01', strategy_cls=WeeklyMACD_KDJ)
print_result(r)
"
```

### 2. Dashboard（5 min）

```bash
pipenv run streamlit run dashboard.py --server.port 8501 --server.headless true  # http://localhost:8501
```

- **单标 Tab**：选标的/策略 → 权益曲线 + 买卖点 + 策略对比
- **组合 Tab**：多标组合回测 → 风险看板 + PnL 归因 + 交易明细筛选导出

### 3. 组合回测（5 min）

```bash
pipenv run python engine/portfolio.py
```

### 4. 策略分析（10 min）

```bash
# 成本敏感性：策略在不同费率下的表现
pipenv run python analysis/cost_sensitivity.py -s weekly_macd_kdj --symbol AAPL

# 参数鲁棒性：参数是否稳定（核心诊断工具）
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL --sizing-mode risk_budget
```

### 5. 每日扫描（5 min）

```bash
pipenv run python daily.py              # 今日信号
pipenv run python daily.py --history    # 近 7 天历史
pipenv run python daily.py --optimize   # 扫描 + 滚动优化
```

## 推荐参数模板

### 主力：spy_ma_breakout（长线）

```python
from engine.trader import run_backtest
from strategy.spy_ma_breakout import SPYMABreakout

result, df = run_backtest(
    "QQQ", "2020-01-01",
    strategy_cls=SPYMABreakout,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,
    ma_period=200, high_period=20, take_profit_atr_mult=4.0,
)
```
| 标的 | MA | OOS 收益 | Sharpe | 评级 |
|------|-----|--------:|------:|:---:|
| NVDA | 200 | +66% | 1.37 | STABLE |
| QQQ | 200 | +37% | 0.74 | STABLE |
| SPY | 200 | +25% | 0.60 | STABLE |

### 短线：trend_follower

```python
from strategy import TrendFollower

result, df = run_backtest(
    "QQQ", "2020-01-01",
    strategy_cls=TrendFollower,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,
    short_ma=5, long_ma=20,
    adx_threshold=15.0, trail_atr_mult=2.5,
)
```
| 标的 | OOS 收益 | Sharpe | 交易 |
|------|--------:|------:|:---:|
| NVDA | +53% | 1.26 | 32 |
| TSLA | +33% | 1.01 | 30 |
| QQQ | +27% | 1.08 | 27 |

### 备选策略

| 策略 | 适用场景 | 推荐标的 |
|------|------|------|
| `turtle_trading` | 多空双向，趋势+震荡 | SPY / QQQ / PFE |
| `weekly_macd_kdj` | 周线低频，低回撤 | AAPL / NVDA |
| `daily_macd_kdj` | 日线高频，分散风险 | AAPL / TSLA |
| `donchian_breakout` | 多空双向，波动率敏感 | QQQ |

## 仓位管理

两种仓位模式，通过 `sizing_mode` 切换。**回测与实盘共用同一仓位计算函数**（`utils/sizing.py`），确保参数优化结果可直接迁移到实盘。

### fixed_capital（默认）

策略自行决定仓位大小（通常为可用资金的 95%）：

```python
result, _ = run_backtest("AAPL", "2020-01-01", strategy_cls=WeeklyMACD_KDJ)
```

### risk_budget（风险预算，推荐实盘使用）

引擎统一按 `风险金额 / 止损距离` 计算仓位。公式与 `RiskController` 共用 `calc_risk_budget_qty()`：

```
qty = capital × risk_pct / (ATR × stop_atr_mult) × vol_scalar
```

其中 `vol_scalar` 实盘启用（根据 `vol_sensitivity` 自适应缩放），回测默认关闭。

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
| fixed_capital | 高（满仓复利） | 高 | 回测研究，了解策略上限 |
| risk_budget 2% | 中 | 可控（<10%） | 实盘稳健执行 |

## 风控

组合回测支持完整的风控层级，所有被拦截信号记录 `rejections`：

```python
from engine.portfolio import PortfolioBacktest, Leg
from utils.sectors import DEFAULT_SECTORS

bt = PortfolioBacktest(
    legs=[Leg("AAPL", "weekly_macd_kdj"), Leg("NVDA", "weekly_macd_kdj"),
          Leg("TSLA", "weekly_macd_kdj"), Leg("SPY", "turtle_trading")],
    initial_capital=100000,
    allocation="dynamic_equal",
    # --- 仓位 ---
    sizing_mode="risk_budget", risk_per_trade=0.02,
    # --- 组合风控 ---
    max_symbol_weight=0.25,           # 单标的上限 25%
    max_sector_weight=0.30,           # 单行业上限 30%
    max_gross_exposure=0.80,          # 总敞口 ≤ 80%
    max_daily_new_positions=3,        # 单日最大新开仓
    cooldown_after_stop_days=10,      # 止损后 10 天内禁止重入
    sector_map=DEFAULT_SECTORS,
)
result = bt.run(start="2020-01-01")
result.summary()
```

## 实盘前检查清单

在启动 `live_trader.py` 实盘前，逐项确认：

- [ ] **637 测试全部通过** `pytest tests/ -q`
- [ ] **黄金样本无漂移** — CI 绿标
- [ ] `param_robustness` 评级 ROBUST 或 STABLE，非 OVERFIT
- [ ] `cost_sensitivity` 评级 A 或 B（10bp/3bp 佣金下仍盈利）
- [ ] `risk_budget` 模式 MaxDD < 15%（扛得住）
- [ ] `daily.py` 能正常输出今日信号，无数据缺失告警
- [ ] FutuOpenD 已启动（如用富途）— `ps aux | grep FutuOpenD`
- [ ] 飞书 Webhook 已配置（如用通知）— `echo $FEISHU_WEBHOOK`
- [ ] MockBroker dry-run 先跑一周，确认无异常
- [ ] 初始资金 ≤ 可承受全部亏损的金额

## 运维特性

### 日志

三类日志隔离 + 文件 JSON 格式输出：

| 日志文件 | 写入者 | 格式 |
|---------|--------|------|
| `logs/live.log` | live_trader.py | JSON 行（Loki/ELK 可解析） |
| `logs/daily.log` | daily.py | JSON 行 |
| `logs/mytrader.log` | 全部模块 | JSON 行 |

### Docker 部署

```bash
docker compose up -d
```

### 市场状态配置

`watchlist.toml`:

```toml
[market_state]
enabled = true
proxy_symbol = "SPY"
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

需要同步修改 3 处：
1. `strategy/__init__.py` — 添加到 `STRATEGY_MAP`
2. 策略类定义 `grid` 属性（优化器自动发现）
3. `watchlist.toml` — 添加 `[strategy.xxx]` 参数段

## 接入券商

```python
from broker.futu import FutuBroker
trader = LiveTrader(broker=FutuBroker(host="127.0.0.1", port=11111))
trader.run()
```

实现 `Broker` 抽象接口即可接入任意券商。
