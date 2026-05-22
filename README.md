# mytrader

个人量化交易系统 — 数据管线 → 策略库 → 回测 → 参数优化 → 每日扫描 → 实盘桥梁。

[![CI](https://github.com/geelaro/mytrader/actions/workflows/ci.yml/badge.svg)](https://github.com/geelaro/mytrader/actions/workflows/ci.yml)

## 功能

| 模块 | 说明 |
|------|------|
| 数据管线 | 统一 DataProvider，腾讯/新浪/AKShare 三源，SQLite 本地缓存 + 增量更新 |
| 策略库 | 10 个策略（趋势/均值回归/动量突破/波动率），BaseStrategy 统一接口 |
| 策略选择层 | SignalGate 门控层：`active`（实盘执行）+ `monitor`（观察对比），市场状态感知 + 风控暂停 + 敞口检查 |
| 回测引擎 | 含滑点佣金，退出逻辑收归策略，支持单标 + 组合回测 |
| 仓位管理 | `fixed_capital`（策略自定） / `risk_budget`（ATR 风险预算）双模式 |
| 参数优化 | 网格搜索 + Walk-forward 样本外验证（资金连续传递）+ 热力图 |
| 分析工具 | 成本敏感性 + 参数鲁棒性 + Monte Carlo 模拟 + 滚动窗口 α 衰减 + 压力测试 |
| 每日回溯 | 批量扫描 watchlist，信号表格 + 飞书卡片推送 + 每日报告 |
| 实盘桥梁 | Broker 抽象接口 + MockBroker（dry-run）+ FutuBroker（富途 OpenD），CLI 切换券商 |
| 风控 | 连续亏损熔断、波动率自适应仓位、单日上限、总敞口、行业权重、止损冷却期、滑点检查、日内亏损上限、超阈值暂停开仓 + 飞书告警 |
| 风控持久化 | risk_state + entry_prices 表，守护进程重启后恢复熔断/日内计数/入场价 |
| 订单管理 | 部分成交轮询（市价60s/限价5min）、限价单超时自动撤单、滑点统计 |
| 组合回测 | 多标的共享资金池，支持 equal/dynamic_equal 分配，组合级风控（集中度/敞口/行业/日开仓上限） |
| 运维可观测 | ops_log 统一运维表（source/level/event 维度），trade_pnl 买卖自动配对，24h 拒单率指标 |
| Dashboard | Streamlit Web UI — 市场状态 → 今日信号 → 策略分布 → 行业饼图 → Monte Carlo 风控 → 实盘记录 → 运行健康面板 |
| CI | GitHub Actions — push/PR 自动跑 pytest（485 测试 + 黄金样本回归） |

## 策略

| 策略 | 类型 | 推荐度 | 描述 |
|------|------|:---:|------|
| `weekly_macd_kdj` | 周线趋势 | ★★★ | KDJ 金叉买入 + MACD 死叉卖出（主力）|
| `daily_macd_kdj` | 日线KDJ | ★★★ | 日线 KDJ 金叉 + MACD 死叉 + ATR 止损 |
| `atr_breakout` | 波动率突破 | ★★ | MA + N×ATR 突破 + 移动止损 |
| `turtle_trading` | 趋势 | ★★ | 双SMA + 唐奇安通道 + ATR 止损 |
| `donchian_breakout` | 动量突破 | ★★ | 唐奇安通道突破 + 移动止损 |
| `trend_follower` | 趋势 | ★ | MA + ADX + Chandelier 移动止损 |
| `weekly_macd` | 周线趋势 | ★ | MACD 金叉死叉 |
| `enhanced_macd` | 趋势 | ✗ | 双MA + MACD + RSI + ATR 止损止盈（过拟合）|
| `bollinger_mean_reversion` | 均值回归 | ✗ | 布林下轨 + RSI 超卖回升（零交易）|
| `bollinger_squeeze` | 波动率收缩 | ✗ | BB 带宽低分位 + 突破上轨（零交易）|

> 推荐度基于 2026-05 全策略鲁棒性扫描（AAPL / risk_budget 2% / IS 2019-2024 / OOS 2024-2026）

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
  ├─ param_robustness.py    #   参数鲁棒性邻域扰动
  ├─ monte_carlo.py         #   Monte Carlo 模拟（交易序列随机化）
  ├─ rolling_alpha.py       #   滚动窗口 α 衰减检测
  └─ stress_test.py         #   极端行情压力测试 (2008/2020/2022)
  utils/             # 工具
  ├─ signal_gate.py         #   策略门控层（市场状态+风控+敞口+孤儿守卫）
  ├─ market_state.py        #   四象限市场状态分类器
  ├─ sectors.py             #   行业分类映射
  ├─ notify.py              #   飞书通知 (Webhook/App 双模式)
  └─ ...
  logs/              # 运行时日志 (live.log / daily.log / mytrader.log)
  tests/             # 485 个测试 (含黄金样本回归 test_golden.py)
  reports/           # 自动生成的 CSV + PNG 报告
  daily.py           # 每日回溯扫描 (入口脚本)
  live_trader.py     # 实盘信号执行 + 风控 (入口脚本)
  dashboard.py       # Streamlit Web 仪表盘 (入口脚本)
  config.py          # 统一运行时配置
  watchlist.toml     # 标的 + 策略 + 风控 + 日志 + 孤儿持仓配置
  Dockerfile         # Docker 一键部署
```

## 快速开始

```bash
# Python 3.10+
pip install pipenv
pipenv install --dev
pipenv run python -m pytest tests/ -v   # 485 tests, verify env
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
pipenv run streamlit run dashboard.py  --server.port 8501 --server.headless true # http://localhost:8501
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
```

## 推荐参数模板

基于 2026-05 全策略鲁棒性扫描，以下为已验证的推荐配置：

### 主力：weekly_macd_kdj + risk_budget 2%

```python
from engine.trader import run_backtest
from strategy import WeeklyMACD_KDJ

result, df = run_backtest(
    "AAPL", "2020-01-01",
    strategy_cls=WeeklyMACD_KDJ,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,       # 单笔风险 2%
    risk_atr_mult=2.0,         # 止损 = 2×ATR
    kdj_n=7, kdj_k=2, kdj_d=2,  # AAPL 最优参数
)
```
| 标的 | 最优参数 | OOS 收益 | Sharpe | 评级 |
|------|---------|--------:|------:|:---:|
| AAPL | n=7 k=2 d=2 | +26% | 1.53 | STABLE |
| NVDA | n=7 k=2 d=2 | +34% | 1.48 | STABLE |
| TSLA | n=14 k=3 d=3 | +92% | 2.61 | ROBUST |
| GOOGL | n=7 k=2 d=2 | +150% | 5.09 | STABLE |
| AMD | n=9 k=2 d=5 | +198% | 3.34 | ROBUST |

> ⚠ `kdj_d=1` 是全局禁区——降到 1 会导致零交易

### 备选策略

```python
# daily_macd_kdj — 日线交易笔数多，适合分散
from strategy import DailyMACD_KDJ
result, _ = run_backtest("AAPL", start="2020-01-01",
    strategy_cls=DailyMACD_KDJ,
    sizing_mode="risk_budget", risk_per_trade=0.02,
    macd_fast=12, macd_slow=21, macd_signal=7,
    kdj_n=14, kdj_k=2, kdj_d=5)
```

| 策略 | 适用场景 | 推荐标的 |
|------|------|------|
| `daily_macd_kdj` | 日线交易笔数多（26笔），分散风险 | AAPL / TSLA |
| `turtle_trading` | 中长线趋势，ETF 优先 | SPY / 510300 |
| `atr_breakout` | 均衡，收益/回撤都不错 | AAPL / NVDA |
| `donchian_breakout` | 波动率敏感，牛市趋势 | QQQ |

## 仓位管理

两种仓位模式，通过 `sizing_mode` 切换：

### fixed_capital（默认）

策略自行决定仓位大小（通常为可用资金的 95%）：

```python
result, _ = run_backtest("AAPL", "2020-01-01", strategy_cls=WeeklyMACD_KDJ)
```

### risk_budget（风险预算，推荐实盘使用）

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

风控拦截日志示例：
```
--- 风控拦截 (12 次) ---
  行业权重: 9次  AAPL, GOOGL, NVDA
  冷却期: 1次  AAPL
  最近拦截:
    2024-04-12 AAPL   行业权重: Technology敞口32% > 30%
```

## 分析工具

### 成本敏感性

扫描佣金 × 滑点网格，输出热力图 + 实盘可行性评级（A+ ~ D-）：

```bash
pipenv run python analysis/cost_sensitivity.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/cost_sensitivity.py --sizing-mode risk_budget --risk-per-trade 0.01
```

### 参数鲁棒性（核心诊断工具）

IS 寻优 → 邻域 ±10%/±20% 扰动 → OOS 分布 → ROBUST~OVERFIT + VIABLE~NEGATIVE 双评级：

```bash
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/param_robustness.py -s enhanced_macd --sizing-mode risk_budget
```

输出解读：
- `✓ 可纳入实盘候选` — ROBUST + VIABLE，放心用
- `△ 参数稳定但策略乏力` — ROBUST + WEAK，换策略优先于调参数
- `✗ 不建议使用` — OVERFIT + NEGATIVE，策略本身不可用

### Monte Carlo 模拟

随机打乱交易顺序 N 次，评估策略对交易序列的敏感性：

```bash
pipenv run python analysis/monte_carlo.py -s weekly_macd_kdj --symbol AAPL --runs 1000
```

### 滚动窗口 α 衰减

检测策略绩效是否随时间退化：

```bash
pipenv run python analysis/rolling_alpha.py -s weekly_macd_kdj --symbol AAPL
```

### 压力测试

用历史极端行情验证策略抗压能力：

```bash
pipenv run python analysis/stress_test.py -s weekly_macd_kdj --symbol AAPL
```

## 实盘前检查清单

在启动 `live_trader.py` 实盘前，逐项确认：

- [ ] **485 测试全部通过** `pytest tests/ -q`
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

### 日志分家

三类日志隔离输出，方便排查：

| 日志文件 | 写入者 | 内容 |
|---------|--------|------|
| `logs/live.log` | live_trader.py | 实盘交易、风控触发、订单状态 |
| `logs/daily.log` | daily.py | 每日扫描、信号输出 |
| `logs/mytrader.log` | 全部模块 | 共享通用日志 |

### 孤儿持仓处理

非 watchlist 标的的持仓自动纳入扫描（只卖不买）。在 `watchlist.toml` 配置兜底策略：

```toml
[defaults]
orphan_strategy = "daily_macd_kdj"
```

### Docker 部署

```bash
docker build -t mytrader .
docker run -d --name mytrader \
  -e FEISHU_WEBHOOK=xxx \
  -v $(pwd)/trading_data.db:/app/trading_data.db \
  mytrader
```

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

## 添加新标的

编辑 `watchlist.toml`：

```toml
[[watchlist]]
symbol = "META"
name = "Meta"
active = "trend_follower"
monitor = ["weekly_macd_kdj"]
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

## 接入券商

```python
from broker.futu import FutuBroker
trader = LiveTrader(broker=FutuBroker(host="127.0.0.1", port=11111))
trader.run()
```

实现 `Broker` 抽象接口即可接入任意券商。
