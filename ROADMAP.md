# traderbridge 整改与开发计划

> 原名 `mytrader`, 2026-05-31 改名为 traderbridge (体现"决策辅助/风险管理"定位)

> 审查日期: 2026-05-23 ｜ 基准: ~16,855 行 Python, 76 源文件, 485 测试
>
> 审查来源: 量化交易架构审查报告 (A) + 架构审查报告 (B) 合并版

---

## 总览

```
阶段一  致命修复     第 1 周    ████████████████   (5 项, 已完成)
阶段二  架构矫正     第 2-3 周  ████████████████   (5/5 项, 已完成)
阶段三  质量加固     第 4 周    ████████████████   (6/6 项, 已完成)
阶段四  MTF 框架     第 5-6 周  ████████████████   (4/4 项, 已完成)
阶段五  策略与组合   第 7-8 周  ████████████████   (4/4 项, 已完成)
阶段六  运维与可观测 第 9-10 周 ████████████████   (6/6 项, 已完成)
阶段七  长期方向     第 11 周+  ░░░░░░░░░░░░░░   (5 方向)
```

---

## 阶段一：致命修复（第 1 周）

### P0-1 统合回测/实盘仓位公式

- [x] 部分完成 — `calc_risk_budget_qty()` 已统一，**但 max_position_pct 上限仍分裂**
- **文件:** `engine/trader.py:106` `live/risk_controller.py:163`
- **问题:** 回测用 `capital × risk_per_trade / (ATR × risk_atr_mult)`，实盘用 `capital × base_risk_pct / (ATR × 2) × vol_scalar`。两者差异 2-5 倍，参数优化结果无法迁移到实盘
- **方案:** 提取统一函数到 `utils/risk.py` 或新建 `utils/sizing.py`，两端共用
- **2026-05-31 复评发现遗留问题:** risk-budget 公式已统一为 `utils/sizing.calc_risk_budget_qty()`，
  但 `max_position_pct` 上限仍分裂：策略层 (`MACDKDJParams.max_position_pct=0.95`) 与
  RiskLimits (`max_position_pct=0.30`) 同时存在，回测走 95% / 实盘走 30%，仓位差约 3 倍。
  剩余统一工作记为 **P2-9**（架构性，待单独立项）。

### P0-2 peak_equity 恢复加日期校验

- [x] 完成
- **文件:** `live/risk_controller.py:59`
- **问题:** 当前无条件从 SQLite 恢复 peak_equity。若历史峰值高但今日已大幅回撤，恢复旧值会误触发市场熔断（max_total_drawdown_pct = 30%）
- **方案:** 恢复时校验 `stored_date == today`，跨日则重新初始化为当日 equity

### P0-3 Config YAML 异常改为 warning + fallback

- [x] 完成
- **文件:** `config.py:155-159`
- **问题:** `except Exception: return` 静默丢弃解析错误。配置 YAML 缩进/格式写错时完全无感知，系统以默认值运行
- **方案:** `logger.warning("config.yaml 解析失败: %s", e)` 后继续使用默认值

### P0-4 Monte Carlo n_sims 参数覆盖

- [x] 完成
- **文件:** `analysis/monte_carlo.py:74`
- **问题:** 函数签名声明了 `n_sims` 参数，但第 74 行无条件覆盖为 `n_sims = 2000`。调用者无论传入任何值都被忽略
- **方案:** `n_sims = n_sims or 2000`

### P0-5 删除/归档 3 个失效策略

- [x] 完成
- **文件:** `strategy/` `watchlist.toml` `engine/optimize.py`
- **影响范围:**
  - `bollinger_mean_reversion` — 已知零交易
  - `bollinger_squeeze` — 已知零交易
  - `enhanced_macd` — 已知过拟合（0 星）
- **方案:**
  1. 文件移至 `archive/strategies/` 保留历史引用
  2. 从 `strategy/__init__.py` 的 `STRATEGY_MAP` 和 imports 中移除
  3. 从 `engine/optimize.py` 的 `PARAM_GRIDS` 中移除
  4. 从 `watchlist.toml` 的 monitor 列表中移除引用

---

## 阶段二：架构矫正（第 2-3 周）

### P1-1 消除尾随止损 ×6 重复

- [x] 完成
- **文件:** `strategy/trend_follower.py` `strategy/atr_breakout.py` `strategy/donchian_breakout.py` `strategy/bollinger_squeeze.py` `strategy/turtle_trading.py` `strategy/daily_macd_kdj.py` `strategy/base.py`
- **问题:** Chandelier 尾随止损逻辑 (`price <= highest - trail_atr_mult × ATR`) 在 6 个策略中一字不差重复
- **方案:** 提取 `ChandelierTrailingExit` Mixin 到 `base.py`，各策略仅需声明 `trail_atr_mult` 参数

### P1-2 合并 weekly / daily_macd_kdj

- [x] 完成
- **文件:** `strategy/weekly_macd_kdj.py` `strategy/daily_macd_kdj.py`
- **问题:** 两者 80% 代码重复。差异仅：周/日重采样 + ATR 尾随止损
- **方案:** 合并为单类 `MACDKDJStrategy`，参数控制 `freq="W"|"D"` 和 `use_atr_stop=True|False`。旧文件改为 re-export，`WeeklyMACD_KDJ` / `DailyMACD_KDJ` 保持向后兼容

### P1-3 消除 MarketState 循环依赖

- [x] 完成
- **文件:** `utils/market_state.py:43`
- **问题:** `from strategy import STRATEGY_MAP` 导入了全部 10 个策略，策略文件可能间接依赖 market_state，形成隐式循环
- **方案:** `is_trend_strategy()` / `is_mean_reversion_strategy()` 接收 `regime_map` 参数，由 `signal_gate.py` 注入

### P1-4 Tencent 单源加 fallback 链

- [x] 完成
- **文件:** `data/sources.py` `data/protocol.py` `data/cache.py`
- **问题:** `SOURCE_PRIORITY["us"] = ["tencent"]` — 腾讯 API 为唯一 US 数据源，无 SLA，随时可能变动或限流
- **方案:**
  1. 新增 `SinaUSSource` — 新浪美股日K，回溯至 1984 年，实测可用
  2. 新增 `YahooChartSource` — Yahoo v8 chart API + cookie 流，作为第三级回退
  3. 回退链: `sina_us → tencent → yahoo_chart`
  4. 彻底移除 `yfinance` 依赖和 `YFinanceSource`
  5. `missing_ranges()` 新增缺口合并逻辑，避免几十个小缺口触发级联请求
- **变更:** +208/-75 行源码, +167 行测试, 576 passed

### P1-5 CacheManager 按职责拆分

- [x] 完成
- **文件:** `data/cache.py` (~420 行)
- **问题:** `CacheManager` 管理 6 种数据类型（OHLCV、signal_history、risk_state、entry_prices、trade_pnl、ops_log），`init_schema()` 每次连接执行 11 CREATE TABLE + 6 ALTER TABLE + 1 CREATE INDEX
- **方案:** 拆为 3 个类 + 1 个 facade：
  - `OhlcvCache(_CacheBase)` — OHLCV load/save/date_range/missing_ranges
  - `StateStore(_CacheBase)` — risk_state / entry_prices / trade_pnl
  - `OpsLogger(_CacheBase)` — ops_log / order_log / slippage_log
  - `CacheManager(OhlcvCache, StateStore, OpsLogger)` — 全功能向后兼容 + signal_history

---

## 阶段三：质量加固（第 4 周）

### P2-1 回测引擎主循环重构

- [x] 完成
- **文件:** `engine/trader.py` `run()` (~100 行)
- **方案:** 提取状态机子方法：`_process_pending_order()` / `_check_exit_signal()` / `_check_entry_signal()` / `_apply_stop_cooldown()`

### P2-2 组合回测主循环拆分

- [x] 完成
- **文件:** `engine/portfolio.py` `run()` (~180 行)
- **方案:** 拆分子方法：`_handle_pending_order()` / `_check_leg_exit()` / `_check_leg_entry()`

### P2-3 RSI 计算提取到 base.py

- [x] 完成
- **文件:** `strategy/enhanced_macd.py` `strategy/bollinger_mean_reversion.py` `strategy/base.py`
- **问题:** RSI 计算逻辑在两处独立实现，参数和边界处理可能不一致
- **方案:** `compute_rsi()` 加入 `base.py` 帮助函数组，两策略各 -5 行

### P2-4 硬编码分裂调校外置

- [x] 完成
- **文件:** `data/sources.py` TencentSource
- **问题:** AAPL/NVDA/TSLA/AMZN/GOOGL 的分裂调整因子硬编码在源码中，每次拆股需要手动改代码
- **方案:** 移至 `data/splits.json` 配置文件，`_load_splits()` 自动加载

### P2-5 金标测试扩展到全策略

- [x] 完成
- **文件:** `tests/test_golden.py`
- **新增覆盖:** `atr_breakout` `donchian_breakout` `daily_macd_kdj` `weekly_macd` `MACDKDJStrategy`
- **参数:** seed=42, 300 bars, $10k capital, tolerance ±0.01, 17 tests total

### P2-6 实盘 BUY 后刷新风控

- [x] 完成
- **文件:** `live_trader.py`
- **问题:** `check_global()` 当前仅在 SELL 后调用，BUY 后未刷新。总敞口超限检查滞后一个循环周期
- **方案:** BUY/SELL 成交后统一调用 `check_global()`

---

## 阶段四：Multi-Timeframe 框架（第 5-6 周）

> 中期最有价值改进，从根本提升信号质量

### M-1 重构 BaseStrategy 接口

- [x] 完成
- **文件:** `strategy/base.py` 及所有策略
- **方案:** `calculate_indicators(df, df_weekly=None)` 策略可同时接收日线和周线
- **向后兼容:** 默认 `df_weekly=None`，现有策略无需改动

### M-2 weekly_macd_kdj 迁移为示范

- [x] 完成
- **文件:** 合并后的 `strategy/macd_kdj.py`
- **方案:** freq="W" + df_weekly 时，指标在周线计算后 ffill 映射回日线时间轴。freq="D" 保持原逻辑

### M-3 SignalScanner 跨频率对齐

- [x] 完成
- **文件:** `utils/signal_scanner.py`
- **方案:** `_fetch_weekly()` 自动重采样 → `calculate_indicators(df, df_weekly=...)` → TypeError fallback 兼容旧策略

### M-4 DataProvider 按需喂多频率

- [x] 完成
- **文件:** `data/provider.py`
- **方案:** `get_data(symbol, freqs=["D","W"])` 返回 `{"D": df, "W": df_weekly}`，内部缓存重采样

---

## 阶段五：策略与组合增强（第 7-8 周）

### E-1 StrategyEnsemble 信号组合

- [x] 完成
- **文件:** 新建 `strategy/ensemble.py`
- **方案:** 按市场状态自动加权投票。StrategyEnsemble 继承 BaseStrategy，封装多个子策略 + MarketStateClassifier

### E-2 批量仓位分配（替代顺序处理）

- [x] 完成
- **文件:** `live/order_manager.py`
- **方案:** 三阶段：①处理卖单 → ②收集买入候选 → ③等风险均分 capital/n 批量下单

### E-3 DataQuality 层

- [x] 完成
- **文件:** 新建 `data/quality.py`
- **功能:** flag_missing / flag_price_jumps / flag_non_trading / validate_ohlcv / quality_report / clean

### E-4 参数滚动优化接入实盘

- [x] 完成
- **文件:** `daily.py` + `utils/env.py`
- **方案:** `--optimize` 标记触发 walk-forward。OOS Sharpe 下滑 >30% 自动更新 watchlist.toml。新增 save_toml()

---

## 阶段六：运维与可观测性（第 9-10 周）

### O-1 Docker Compose

- [x] 完成
- **文件:** 新增 `docker-compose.yml`
- **内容:**
  - `services: traderbridge` + `futu-opend`（FutuOpenD 容器）
  - 环境变量注入: `FEISHU_*` `TRADERBRIDGE_DB` (legacy `MYTRADER_DB` 也接受) `FUTU_HOST`
  - 持久化卷: `trading_data.db` `logs/` `reports/`

### O-2 daemon 健康检查

- [x] 完成
- **文件:** `live_trader.py` 内置 HTTP server
- **方案:** `GET /health` → `{"status":"ok","last_tick":"2026-05-23T14:30:00Z","paused":false}`
- **Docker:** `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health`

### O-3 结构化日志

- [x] 完成
- **文件:** `utils/logging.py`
- **方案:** 统一日志格式为 JSON 行（`{"ts":"...","level":"INFO","logger":"live","event":"trade_filled",...}`），供 Loki / ELK 解析

### O-4 飞书日报增强

- [x] 完成
- **文件:** `utils/notify.py`
- **当前:** 简单信号/交易通知
- **方案:** 日终汇总卡片：
  - 昨日 PnL 归因（按策略分解 / 按标的分解）
  - 风控事件（熔断次数、滑点超标次数、连亏计数）
  - 当日预扫描信号预览
  - 当前持仓摘要 + 浮盈/浮亏

### O-5 DB migration 版本化

- [x] 完成
- **文件:** `data/cache.py`
- **方案:** 引入 `schema_version` 表 + 版本化迁移函数列表。替换现有 `try: ALTER TABLE except: pass` 模式

### O-6 DB 路径绝对化

- [x] 完成
- **文件:** `data/cache.py`
- **问题:** `os.environ.get("MYTRADER_DB", "trading_data.db")` 相对路径依赖 CWD。cron 触发 `daily.py` 时工作目录可能不一致
- **方案:** `Path(...).resolve()` 或基于 `PROJECT_ROOT` 生成绝对路径

---

## 阶段七：长期方向（第 11 周+）

### 因子分析框架

- [ ] 开始研究
- **门槛:** 需积累 ≥1 年实盘交易记录
- **功能:** Beta 暴露分解、Rank IC 分析 (按日/周)、因子相关性热力图、IC decay 曲线

### 日内策略

- [ ] 开始研究
- **门槛:** FutuOpenD Level-2 数据权限
- **功能:** 分钟级数据管道、MomentumBreakout 策略、TWAP/VWAP 算法订单执行

### ML 辅助信号

- [ ] 开始研究
- **门槛:** ≥2000 笔历史交易记录
- **方案:** LightGBM 预测未来 N 日方向（特征=现有指标列）→ 输出置信度权重 → 与规则策略加权融合。规则策略保持为基线

### 黑天鹅预案

- [ ] 开始研究
- **功能:**
  - 2020-03 / 2022-06 式闪崩自动识别并强制平仓
  - 极端波动（VIX > 40）自动降低持仓至 30% 上限
  - 可选：深度虚值 put 对冲信号

### 多账户管理

- [ ] 开始研究
- **门槛:** broker 接口需支持多连接
- **功能:** 多 Futu 账户同时管理、跨账户风控聚合、账户间资金调拨

---

## 专业风险管理平台对标 (2026-06-01 评估)

按 Aladdin / Barra / Bloomberg POMS 对标, traderbridge 缺失的模块,
按价值排序. 标 ✅ 的本批次做.

### 第一批 — 本批次实施

- ✅ **VaR / Expected Shortfall** — Historical/Parametric/Conditional VaR, 1d 95%/99%
- ✅ **历史场景压力测试** — 2008/2020/2022 重放 + 当前持仓重算
- ✅ **集中度指标** — HHI, Top-N, Effective N, 行业暴露, 相关性 HHI

### 第二批 — 实盘相关

- [ ] **Kill Switch / 紧急平仓** — 一键 market-sell-all + VIX>50 自动触发.
  实盘前必备, 半天
- [ ] **流动性风险** — Days-to-Liquidate (持仓/ADV), Position vs ADV %.
  中小盘 / 大仓位时出场预估

### 第三批 — 业绩分析深化

- [ ] **Risk-Adjusted Metrics 深化** — Sortino / Calmar / MAR / Omega.
  当前只有 Sharpe. Sortino 只看下行波动, 更贴近"风险"
- [ ] **Drawdown Analytics 深化** — Underwater curve, Time-to-recover,
  Pain Index. 当前只有 MaxDD 数字
- [ ] **Brinson Performance Attribution** — 资产配置 vs 选股 vs 交互效应
  分解. 不同于现有的因子暴露归因
- [ ] **Realized vs Unrealized PnL 拆分** — trade_pnl 表已有 realized,
  dashboard 没正式拆开展示

### 第四批 — 报告与合规

- [ ] **风险报告自动生成** — 周报 / 月报 PDF, 含 VaR / Beta / 暴露 /
  持仓 / PnL 归因, 飞书推送
- [ ] **税务批次会计 (FIFO/LIFO/HIFO)** — 报税需要, 但富途自带, 可暂缓
- [ ] **Style drift detection** — 策略在不同 regime 下风格漂移监测

---

## 优先级决策矩阵

```
             ┌── 高影响 ────┬── 中影响 ────┐
             │              │              │
低难度 ───── P0-1 公式对齐 │ P0-3 配置日志 │
             P0-4 MC 修复  │ P2-6 BUY刷新  │
             P0-5 删失效   │ P2-3 RSI提取  │
             ──────────────┼───────────────┤
中难度 ───── P1-4 多源链   │ P1-3 环依赖   │
             P1-1 止损去重 │ P2-1 引擎重构  │
             P1-2 KDJ合并  │ P2-5 金标扩展  │
             ──────────────┼───────────────┤
高难度 ───── P1-5 缓存拆分 │ M-1 MTF 框架  │
             ──────────────┤ E-1 策略组合   │
                           │ O-4 日报增强   │
```

## 进度记录

| 日期 | 阶段 | 项目 | 状态 | 备注 |
|------|------|------|------|------|
| 2026-05-23 | 阶段一 | P0-1 ~ P0-5 | 已完成 | 5/5 全绿, 571 passed, 0 failed |
| 2026-05-23 | 阶段一 | P0-1 统合回测/实盘仓位公式 | 完成 | 新增 utils/sizing.py, 两端共用 calc_risk_budget_qty() |
| 2026-05-23 | 阶段一 | P0-2 peak_equity 日期校验 | 完成 | 恢复逻辑移入 stored_date==today 分支内 |
| 2026-05-23 | 阶段一 | P0-3 Config YAML 异常警告 | 完成 | 改用 logging.warning() 记录解析错误 |
| 2026-05-23 | 阶段一 | P0-4 Monte Carlo 参数覆盖 | 完成 | n_sims = n_sims or 2000 |
| 2026-05-23 | 阶段一 | P0-5 失效策略移除 | 完成 | 从 STRATEGY_MAP/watchlist.toml 移除, 保留源文件供测试导入 |
| 2026-05-23 | 阶段二 | P1-4 数据源多链 | 完成 | SinaUSSource + YahooChartSource, 移除 yfinance, 回退链 sina_us→tencent→yahoo_chart, missing_ranges 缺口合并 |
| 2026-05-25 | 阶段二 | P1-1 尾随止损去重 | 完成 | ChandelierTrailingExit Mixin 提取到 base.py, 6 策略复用 |
| 2026-05-25 | 阶段二 | P1-2 KDJ合并 | 完成 | MACDKDJStrategy 统一类, freq="W"\|"D" + use_atr_stop, 旧文件re-export |
| 2026-05-25 | 阶段二 | P1-3 循环依赖 | 完成 | is_trend_strategy/is_mean_reversion_strategy 接收 regime_map 参数 |
| 2026-05-25 | 阶段二 | P1-5 缓存拆分 | 完成 | OhlcvCache / StateStore / OpsLogger + CacheManager facade |
| 2026-05-25 | 阶段三 | P2-1 回测引擎重构 | 完成 | 4 个子方法提取，run() 主循环 ~50 行 |
| 2026-05-25 | 阶段三 | P2-2 组合回测重构 | 完成 | _handle_pending_order / _check_leg_exit / _check_leg_entry |
| 2026-05-25 | 阶段三 | P2-3 RSI 提取 | 完成 | compute_rsi() 加入 base.py，双策略复用 |
| 2026-05-25 | 阶段三 | P2-4 分裂外置 | 完成 | splits.json + _load_splits() |
| 2026-05-25 | 阶段三 | P2-5 金标扩展 | 完成 | 5 策略新增 → 17 golden tests, 586 passed |
| 2026-05-25 | 阶段三 | P2-6 BUY刷新风控 | 完成 | check_global() 统一在 BUY/SELL 成交后调用 |
| 2026-05-25 | 阶段四 | M-1 BaseStrategy 接口 | 完成 | calculate_indicators(df, df_weekly=None) 多频签名 |
| 2026-05-25 | 阶段四 | M-2 macd_kdj MTF | 完成 | freq="W"+df_weekly → 周线指标 ffill 映射日线 |
| 2026-05-25 | 阶段四 | M-3 SignalScanner | 完成 | _fetch_weekly + TypeError fallback |
| 2026-05-25 | 阶段四 | M-4 DataProvider 多频 | 完成 | get_data(freqs=["D","W"]) |
| 2026-05-25 | 阶段五 | E-1 StrategyEnsemble | 完成 | 策略组合加权投票, MarketRegime 自适应权重 |
| 2026-05-25 | 阶段五 | E-2 批量仓位分配 | 完成 | 三阶段: 卖单→收集→等风险均分批量下单 |
| 2026-05-25 | 阶段五 | E-3 DataQuality 层 | 完成 | quality.py: flag/clean/validate 全套检查 |
| 2026-05-25 | 阶段五 | E-4 滚动优化接入 | 完成 | daily.py --optimize + save_toml |
| 2026-05-25 | 阶段六 | O-1 Docker Compose | 完成 | docker-compose.yml + HEALTHCHECK |
| 2026-05-25 | 阶段六 | O-2 健康检查 | 完成 | live_trader.py 内嵌 HTTP /health |
| 2026-05-25 | 阶段六 | O-3 结构化日志 | 完成 | JsonFormatter → JSON 行日志 |
| 2026-05-25 | 阶段六 | O-4 日报增强 | 完成 | daily_card() PnL归因+风控+持仓 |
| 2026-05-25 | 阶段六 | O-5 DB migration | 完成 | schema_version + _run_migrations() |
| 2026-05-25 | 阶段六 | O-6 DB路径绝对化 | 完成 | Path(...).resolve() |
| 2026-05-31 | 复评-P0 | S1 做空虚增买入力 | 完成 (`589756a`) | 引入 available_cash + short_margin_ratio=1.5 (Reg-T), 开多检查改用 available_cash |
| 2026-05-31 | 复评-P0 | S2 ensemble 死代码 + 索引错位 | 完成 (`589756a`) | reindex 到 df.index, long/short_mask 真冲突仲裁, member_weights 归一化除均值 |
| 2026-05-31 | 复评-P0 | S3 SPYMABreakout 死代码 | 完成 (`589756a`) | 移除 short 信号 + SHORT 出场 + N_day_low |
| 2026-05-31 | 复评-P0 | S4 默认策略 | 完成 (`589756a`) | run_backtest 默认 enhanced_macd → MACDKDJStrategy |
| 2026-05-31 | 复评-P0 | S5 turtle O(N²) → O(1) | 完成 (`589756a`) | 实例缓存 _cur_entry_atr/_cur_highest_high/_cur_lowest_low 增量更新 |
| 2026-05-31 | 复评-P1 | P1-3 peak_equity 跨日 | 完成 (`b5f3165`) | peak_equity + consecutive_losses 从 today 分支移出 |
| 2026-05-31 | 复评-P1 | P1-5 max_slippage_pct | 完成 (`b5f3165`) | 2% → 0.5% (50bp) |
| 2026-05-31 | 复评-P1 | P1-6 SignalGate 动态化 | 完成 (`b5f3165`) | 移除模块期固化, dataclass field + 懒加载 _build_regime_map |
| 2026-05-31 | 复评-P1 | P1-7 跨源校验 | 完成 (`b5f3165`) | _check_cross_source_drift: 重叠 close >1% 或边界 >20% warn |
| 2026-05-31 | 复评-P2 | P2-2/3/4/6/7/8/10 | 完成 (`c9f9eff`) | 公式精确化 / 缺口阈值 3→7 / health 刷新 / 融券利息 / ensemble typo warn / MTF min_bars×5 / 废弃清理 |
| 2026-05-31 | 复评-P3 | P3-1/2/4/5/7 | 完成 | Windows signal 兼容 / sector_map fallback warn / print-logger 约定文档化 / SQLite 读路径加锁 / 本表同步 |

## 复评后剩余待办

| 编号 | 题 | 状态 |
|------|----|------|
| P2-5 | PortfolioBacktest → PortfolioState 重构 | ✅ 完成 (`252a5e8`) |
| P2-9 | max_position_pct 组合回测与实盘对齐 | ✅ 完成 (`0c47f3b`) |
| P2-1 | 回测 vs 实盘信号时点对齐 | 📌 已立项, 暂不修 |

### P2-1 决策记录 (2026-05-31)

**决策**: 暂不动代码, 仅做文档化标注。

**理由**:
- 当前 watchlist 主力 `weekly_macd_kdj` 周线信号稀疏, 日内闪烁概率极低,
  实盘行为实际安全。
- 出场即时响应 (跳空止损保护) 是用户明确需要的实盘特性, 不能为对齐回测牺牲。

**未来触发修复的场景** (任一即必修):
1. 把 daily_macd_kdj / rsi2 / spy_ma_breakout 任一设为 active
2. 启动多因子模型 / Ensemble 实盘下单
3. 接入日内策略
4. 想让回测优化的参数可严格迁移到实盘

**修复方向草案**:
- 入场延次日开盘提交 (MARKET 单, 与回测 next_open 对齐)
- 出场保持即时响应
- 新增 pending_orders 表 + post-close / pre-market 双时点 cron / daemon

---

> 此文档将随开发进度持续更新。每完成一项，勾选其 checkbox 并在进度表中记录。
