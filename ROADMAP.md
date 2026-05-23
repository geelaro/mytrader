# mytrader 整改与开发计划

> 审查日期: 2026-05-23 ｜ 基准: ~16,855 行 Python, 76 源文件, 485 测试
>
> 审查来源: 量化交易架构审查报告 (A) + 架构审查报告 (B) 合并版

---

## 总览

```
阶段一  致命修复     第 1 周    ████████████████   (5 项, 已完成)
阶段二  架构矫正     第 2-3 周  ██░░░░░░░░░░░░░░   (1/5 项, P1)
阶段三  质量加固     第 4 周    ░░░░░░░░░░░░░░   (6 项, P2)
阶段四  MTF 框架     第 5-6 周  ░░░░░░░░░░░░░░   (4 项)
阶段五  策略与组合   第 7-8 周  ░░░░░░░░░░░░░░   (4 项)
阶段六  运维与可观测 第 9-10 周 ░░░░░░░░░░░░░░   (6 项)
阶段七  长期方向     第 11 周+  ░░░░░░░░░░░░░░   (5 方向)
```

---

## 阶段一：致命修复（第 1 周）

### P0-1 统合回测/实盘仓位公式

- [x] 完成
- **文件:** `engine/trader.py:106` `live/risk_controller.py:163`
- **问题:** 回测用 `capital × risk_per_trade / (ATR × risk_atr_mult)`，实盘用 `capital × base_risk_pct / (ATR × 2) × vol_scalar`。两者差异 2-5 倍，参数优化结果无法迁移到实盘
- **方案:** 提取统一函数到 `utils/risk.py` 或新建 `utils/sizing.py`，两端共用

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

- [ ] 完成
- **文件:** `strategy/trend_follower.py` `strategy/atr_breakout.py` `strategy/donchian_breakout.py` `strategy/bollinger_squeeze.py` `strategy/turtle_trading.py` `strategy/daily_macd_kdj.py` `strategy/base.py`
- **问题:** Chandelier 尾随止损逻辑 (`price <= highest - trail_atr_mult × ATR`) 在 6 个策略中一字不差重复
- **方案:** 提取 `ChandelierTrailingExit` Mixin 到 `base.py`，各策略仅需声明 `trail_atr_mult` 参数

### P1-2 合并 weekly / daily_macd_kdj

- [ ] 完成
- **文件:** `strategy/weekly_macd_kdj.py` `strategy/daily_macd_kdj.py`
- **问题:** 两者 80% 代码重复。差异仅：周/日重采样 + ATR 尾随止损
- **方案:** 合并为单类 `MACDKDJStrategy`，参数控制 `freq="W"|"D"` 和 `use_atr_stop=True|False`

### P1-3 消除 MarketState 循环依赖

- [ ] 完成
- **文件:** `utils/market_state.py:43`
- **问题:** `from strategy import STRATEGY_MAP` 导入了全部 10 个策略，策略文件可能间接依赖 market_state，形成隐式循环
- **方案:** `is_trend_strategy()` 改为接收 regime 映射表作为参数，由调用者注入

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

- [ ] 完成
- **文件:** `data/cache.py` (511 行)
- **问题:** `CacheManager` 管理 6 种数据类型（OHLCV、signal_history、risk_state、entry_prices、trade_pnl、ops_log），`init_schema()` 每次连接执行 11 CREATE TABLE + 6 ALTER TABLE + 1 CREATE INDEX
- **方案:** 拆为 3 个类：
  - `OhlcvCache` — OHLCV 存储和增量更新
  - `StateStore` — risk_state / entry_prices / trade_pnl 持久化
  - `OpsLogger` — ops_log / order_log / slippage_log 写入

---

## 阶段三：质量加固（第 4 周）

### P2-1 回测引擎主循环重构

- [ ] 完成
- **文件:** `engine/trader.py` `run()` (~100 行)
- **方案:** 提取状态机子方法：
  - `_process_pending_order()`
  - `_check_entry_signal()`
  - `_check_exit_signal()`
  - `_apply_stop_cooldown()`

### P2-2 组合回测主循环拆分

- [ ] 完成
- **文件:** `engine/portfolio.py` `run()` (~180 行)
- **方案:** 拆分子方法：
  - `_process_bar_for_leg()`
  - `_handle_pending_order()`
  - `_compute_equity_snapshot()`

### P2-3 RSI 计算提取到 base.py

- [ ] 完成
- **文件:** `strategy/enhanced_macd.py` `strategy/bollinger_mean_reversion.py` `strategy/base.py`
- **问题:** RSI 计算逻辑在两处独立实现，参数和边界处理可能不一致
- **方案:** `compute_rsi()` 加入 `base.py` 帮助函数组

### P2-4 硬编码分裂调校外置

- [ ] 完成
- **文件:** `data/sources.py` TencentSource
- **问题:** AAPL/NVDA/TSLA/AMZN/GOOGL 的分裂调整因子硬编码在源码中，每次拆股需要手动改代码
- **方案:** 移至 `data/splits.json` 配置文件。后续考虑从 yfinance 自动获取分裂因子

### P2-5 金标测试扩展到全策略

- [ ] 完成
- **文件:** `tests/test_golden.py`
- **当前覆盖:** `weekly_macd_kdj` `trend_follower` `turtle_trading`
- **待新增:** `atr_breakout` `donchian_breakout` `daily_macd_kdj` `weekly_macd` `MACDKDJStrategy`(合并后)
- **参数:** seed=42, 300 bars, $10k capital, tolerance ±0.01

### P2-6 实盘 BUY 后刷新风控

- [ ] 完成
- **文件:** `live_trader.py`
- **问题:** `check_global()` 当前仅在 SELL 后调用 (L229-231)，BUY 后未刷新。总敞口超限检查滞后一个循环周期
- **方案:** BUY 下单后也调用 `check_global()` 立即检查

---

## 阶段四：Multi-Timeframe 框架（第 5-6 周）

> 中期最有价值改进，从根本提升信号质量

### M-1 重构 BaseStrategy 接口

- [ ] 完成
- **文件:** `strategy/base.py` 及所有策略
- **方案:** `calculate_indicators(df_daily, df_weekly) → df` 策略可同时接收日线和周线，内部做跨周期信号对齐
- **向后兼容:** 保留 `calculate_indicators(df)` 的单参数签名作为 fallback

### M-2 weekly_macd_kdj 迁移为示范

- [ ] 完成
- **文件:** 合并后的 `strategy/macd_kdj.py`
- **方案:** 周线 MACD/KDJ 指标计算 → 映射回日线时间轴 → 日线精确入场执行。消除策略内部 `resample_weekly` 隐式约定

### M-3 SignalScanner 跨频率对齐

- [ ] 完成
- **文件:** `utils/signal_scanner.py`
- **方案:** 同一标的的日线信号和周线上下文按时间戳对齐后输出。扫描引擎适配多频数据输入

### M-4 DataProvider 按需喂多频率

- [ ] 完成
- **文件:** `data/provider.py`
- **方案:** `get_data(symbol, freqs=["D","W"])` 内部缓存周线重采样结果，避免多策略重复计算

---

## 阶段五：策略与组合增强（第 7-8 周）

### E-1 StrategyEnsemble 信号组合

- [ ] 完成
- **方案:** 按市场状态自动加权投票
  - TRENDING_UP: `(趋势策略 × 0.7) + (均值回归 × 0.3)`
  - RANGING: `(均值回归 × 0.6) + (趋势策略 × 0.4)`
  - HIGH_VOL: `(趋势策略 × 0.6) + 清仓信号优先`

### E-2 批量仓位分配（替代顺序处理）

- [ ] 完成
- **文件:** `live_trader.py`
- **方案:** 收集所有标的的当日信号 → 统一最优分配（等风险加权）→ 批量下单。消除先到先得的资金分配偏差

### E-3 DataQuality 层

- [ ] 完成
- **位置:** 新建 `data/quality.py`
- **功能:**
  - 缺失值标记（NaN → 标记列）
  - 异常价格跳变检测（单日涨跌 >20% 标记）
  - 停牌日/无成交日标记（is_trading_day 列）

### E-4 参数滚动优化接入实盘

- [ ] 完成
- **文件:** `engine/optimize.py` → daily 流程集成
- **方案:** `daily.py` 运行后可选触发 walk-forward 重优化。若 OOS 性能下滑超阈值则自动更新 `watchlist.toml` 参数段

---

## 阶段六：运维与可观测性（第 9-10 周）

### O-1 Docker Compose

- [ ] 完成
- **文件:** 新增 `docker-compose.yml`
- **内容:**
  - `services: mytrader` + `futu-opend`（FutuOpenD 容器）
  - 环境变量注入: `FEISHU_*` `MYTRADER_DB` `FUTU_HOST`
  - 持久化卷: `trading_data.db` `logs/` `reports/`

### O-2 daemon 健康检查

- [ ] 完成
- **文件:** `live_trader.py` 内置 HTTP server
- **方案:** `GET /health` → `{"status":"ok","last_tick":"2026-05-23T14:30:00Z","paused":false}`
- **Docker:** `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health`

### O-3 结构化日志

- [ ] 完成
- **文件:** `utils/logging.py`
- **方案:** 统一日志格式为 JSON 行（`{"ts":"...","level":"INFO","logger":"live","event":"trade_filled",...}`），供 Loki / ELK 解析

### O-4 飞书日报增强

- [ ] 完成
- **文件:** `utils/notify.py`
- **当前:** 简单信号/交易通知
- **方案:** 日终汇总卡片：
  - 昨日 PnL 归因（按策略分解 / 按标的分解）
  - 风控事件（熔断次数、滑点超标次数、连亏计数）
  - 当日预扫描信号预览
  - 当前持仓摘要 + 浮盈/浮亏

### O-5 DB migration 版本化

- [ ] 完成
- **文件:** `data/cache.py`
- **方案:** 引入 `schema_version` 表 + 版本化迁移函数列表。替换现有 `try: ALTER TABLE except: pass` 模式

### O-6 DB 路径绝对化

- [ ] 完成
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

---

> 此文档将随开发进度持续更新。每完成一项，勾选其 checkbox 并在进度表中记录。
