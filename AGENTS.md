# AGENTS.md — traderbridge

## Prerequisites

- **Python 3.10+** (enforced by Pipfile)
- Package manager: **Pipenv** (not pip/poetry/uv)
- Install: `pipenv install --dev`

## Verify

```bash
pipenv run pytest -q                            # full suite (1043 tests, ~40s)
pipenv run pytest tests/test_golden.py -v        # CI gate only
pipenv run pytest tests/test_strategy.py -v      # single file
pipenv run pytest tests/test_strategy.py::TestWeeklyMACDKDJ -v  # single class

# Coverage (matches CI)
pipenv run pytest --cov=. --cov-report=term --cov-config=.coveragerc
```

CI runs on push/PR via `.github/workflows/ci.yml`, uploads coverage to
Codecov (`.codecov.yml`, badge in README).  Coverage threshold:
project 60% (current ~76%), patch 70%, drop tolerance 2pp.

No lint, formatter, or typechecker is configured. Tests + coverage are the verification steps.

## Project architecture

Flat single-package layout — everything is at the top level. No `src/` layout, no namespace packages.
**Strict single-direction dependencies, no cycles.** See `docs/architecture.md` for the full layer diagram.

| Directory | LOC | Role |
|-----------|----:|------|
| `data/` | 2081 | Market data pipeline — 6 sources + SQLite cache + cross-source drift check + `apply_us_splits` shared helper + realtime VIX bypass |
| `strategy/` | 2051 | Strategy library — `BaseStrategy` + `ChandelierTrailingExit` mixin + 9 active strategies + `StrategyEnsemble` |
| `broker/` | 960 | Broker abstraction — `Broker` ABC + `MockBroker` (dry-run) + `FutuBroker` (real) |
| `engine/` | 2525 | Backtest engines — single-symbol + portfolio + walk-forward optimize |
| `analysis/` | **5299** | ★ **22 risk/performance analysis modules** — VaR / EVT / Stress / Concentration / Correlation / Brinson / Drawdown / Marginal VaR / What-If / Risk Report aggregator / …  Pure compute, no I/O side effects |
| `live/` | 1146 | Live trading bridge — `RiskController` / `OrderManager` / `RiskAlerter` / **`KillSwitch`** / `position_stops` |
| `dashboard/` | 3139 | Streamlit UI — **11 tabs**, pure render |
| `utils/` | 1787 | Cross-cutting — notify (Feishu) / sizing / risk / market_state / signal_gate / logging / sectors |
| `scripts/` | 364 | Cron entry — `weekly_risk_report.py` etc. |
| `tests/` | **11890** | **1043 tests, all offline.** 75.9% coverage on runtime code |
| `docs/` | — | `architecture.md` (full design doc) |

Entry points: `live_trader.py`, `daily.py`, `dashboard.py`, `scripts/weekly_risk_report.py`. None are importable as libraries — they're scripts.

## Critical: import side-effects

**`import utils` MUST be done before any `matplotlib` import.** The `utils/__init__.py` triggers `utils/env.py` on import, which:
- Fixes Windows GBK encoding
- Sets matplotlib backend to `"Agg"` (headless, required on servers/Docker)
- Loads `.env` via dotenv

If you import matplotlib or pyplot before `import utils`, the backend won't be `Agg` and charts will fail in headless environments.

## Config system

Load order: `DEFAULT_CONFIG` (hardcoded in `config.py`) → `config.yaml` (optional, at project root) → environment variables (`.env`).

`watchlist.toml` is the canonical source for symbols, active strategies, monitor strategies, strategy params, risk params, and market state config. It's lazy-loaded via `config.watchlist_data` (calls `utils.env.load_toml`). Don't edit watchlist data in `config.py`.

Access config: `from config import config` (singleton), then `config.risk.max_position_pct`, `config.feishu.webhook`, etc.

## Strategy conventions

- All strategies extend `BaseStrategy` from `strategy/base.py`
- Must implement `calculate_indicators(df) → df` — adds a **`Signal` column** with int values: `1` = buy, `-1` = sell, `0` = hold
- Must define `min_bars` property (minimum data needed before first signal)
- Exit logic lives in the strategy (via `check_exit`), not in the engine
- `StrategyParams` is a dataclass for parameter defaults

### Adding a new strategy requires 3 changes in 3 files:
1. **`strategy/__init__.py`** — add to `STRATEGY_MAP` dict and imports
2. **`engine/optimize.py`** — add params import and `PARAM_GRIDS` entry
3. **`watchlist.toml`** — add `[strategy.xxx]` section with default params

## Golden tests — CI blocking

`tests/test_golden.py` contains exact numeric assertions (seed=42, 300 bars, $10k capital). If you change any strategy calculation logic, these WILL fail. The values are hardcoded for 3 strategies in both `fixed_capital` and `risk_budget` modes. Tolerance is ±0.01 absolute.

## Sizing modes

- **`fixed_capital`** (default) — strategy sets position size (~95% of cash). High returns, high drawdowns. For research.
- **`risk_budget`** — engine computes `qty = capital × risk_per_trade / (ATR × risk_atr_mult)`. Lower returns, controlled drawdowns (<10%). For live trading.

## Database

- SQLite, WAL mode. Production DB: `trading_data.db` (gitignored).
- Test DB: `conftest.py` sets `TRADERBRIDGE_DB` env var to a temp file before any import to isolate from production.
- `data/cache.py` — `CacheManager` reads `TRADERBRIDGE_DB` env var first, falls back to legacy `MYTRADER_DB`, then `trading_data.db`.
- DB tables: `ohlcv_daily`, `signals`, `order_log`, `slippage_log`, `ops_log`, `entry_prices`, `trade_pnl`, `risk_state`.

## Undo / rollback workflow

When reverting changes made in earlier work, **prefer git operations over manual file editing**:

```bash
# 1. Undo the last commit but keep changes staged
git reset --soft HEAD~1

# 2. Unstage the specific parts you want to remove
git reset HEAD -- file_to_keep.py

# 3. Checkout the files you want to revert
git checkout -- file_to_revert.py

# 4. Commit the corrected set
git add -A && git commit -m "..."
```

This avoids error-prone manual edits across many files. Use `git stash` for
work-in-progress you want to temporarily set aside.

## Logging conventions — print vs logger

Two channels coexist by design — don't try to "deduplicate" them:

- **`print()`** — operator console UX. Progress bars, position tables, summary
  cards, trade ticker. Format is human-oriented (emojis, columns, separators).
  Always goes to stdout. Lost when piped to log aggregators — that's OK,
  it's not the audit trail.
- **`logger.<level>(...)`** — structured audit / alerting channel. JSON
  format (`utils/logging.JsonFormatter`), goes to file + Loki / ELK. Use
  for: orders, risk events (pause/resume/circuit breaker), rejections,
  data-source failures, fills. Include structured fields, not pre-formatted
  strings, when possible.

Some events are written to both intentionally (e.g. order fills — operator
needs to see them, audit pipeline needs them too). Both writes are fine
**as long as the logger call carries useful structured fields beyond the
human-readable string**. If a `logger.info(msg)` literally repeats a
`print(msg)` with no extra structure, drop it to `logger.debug`.

## Known traps

- **`kdj_d=1`** — causes zero trades. Documented in README, don't set it.
- **`bollinger_mean_reversion`** and **`bollinger_squeeze`** — rated 0 stars, known to produce zero trades. Don't use for live.
- **`enhanced_macd`** — rated 0 stars, documented as overfitted. Don't use for live.
- **Streamlit requires `--server.headless true`** on servers/headless machines.

## Critical design rules

### 1. New US data source MUST call `apply_us_splits()`

`data/sources.py:apply_us_splits()` is a shared helper.  All US-equity
sources (Tencent, SinaUS, YahooChart) call it at the end of `fetch()`.
**Skipping it = cross-source price cliffs in cache** (the NVDA 2023-12-26
909% jump bug).  Same applies when adding a new US source — call
`apply_us_splits(df, sym)` before returning.

### 2. Updating `splits.json` requires force-refresh

`apply_us_splits` is applied **at fetch time** and the result is cached.
If `splits.json` is updated AFTER data is already cached, the old data
stays in the old (unadjusted) scale.  After adding entries:

```python
provider.get_daily(sym, ..., force_refresh=True)
# or just delete the rows: DELETE FROM ohlcv_daily WHERE symbol IN (...)
```

### 3. Realtime VIX uses an isolated session

`data/realtime.py:_realtime_session()` is **separate** from
`data/sources.py:_yahoo_session()`.  Why: `_yahoo_session` warms with
a `fc.yahoo.com` cookie call required by Yahoo's historical chart
endpoint.  That cookie marks the session for **strict rate limiting** —
spark/chart 429 within seconds.  Without the cookie, spark/chart work
fine.

**Any new Yahoo realtime endpoint must use `_realtime_session`**, NOT
`_yahoo_session`.

### 4. Never auto-trigger Kill Switch on VIX / drawdown

Empirical study (CBOE 1990+, see `live/kill_switch.py` docstring):
VIX > 50 has happened 5 times historically; SPY's 250-day forward
return after those events averaged **+44.6%** (vs baseline +11.4%).
**VIX > 50 is a bottom signal, not a panic-sell signal.**

Same logic applies to any "X% drawdown → auto liquidate" trigger.
Manual control only.  This is a `live/kill_switch.py` invariant.

### 5. Streamlit cross-button state needs `st.session_state`

Every button click in Streamlit re-runs the page from top.  Local
variables are reset.  If a tab has two buttons (e.g. "Generate" then
"Send"), the "Send" branch will never see the data built in the
"Generate" branch unless that data is stashed in `st.session_state`.

See `dashboard/risk_report.py` for the canonical pattern (`_rr_report`,
`_rr_data`, `_rr_md` keys).

### 6. `NotifyLogHandler` is NOT auto-installed

`utils/notify.install_notify_log_handler()` is only called in tests.
In production code, do **not** use `logger.error(...)` or
`logger.exception(...)` for expected failures — those would be picked
up if/when the handler is installed and spam Feishu.

Use `logger.warning(...)` for recoverable / try-except'd errors that
the caller already surfaces to the user.  Use `logger.error(...)`
only for genuine unexpected errors that warrant Feishu attention
(when the handler is installed in `live_trader.py`).

### 7. Dashboard tabs are stateless renders

`dashboard/*.py` files only render — never persist state outside
`st.session_state` or `cache.save_*`.  Provider and cache are shared
singletons via `@st.cache_resource`.  Don't introduce module-level
mutable state in dashboard modules.

## Module boundaries (layer contract)

Single repo, but strict layering — the project deliberately stays in one
codebase (no split between "risk system" and "trading system") because
the shared substrate (data/cache, broker abstraction, strategy library,
engine) is large and splitting would force duplication or a shared lib
with hidden coupling.  Instead, layer discipline carries the same
benefits at much lower maintenance cost.

### Layers (low → high)

```
utils       ← cross-cutting; no internal deps
data        ← market data + SQLite cache; deps: utils
broker      ← broker abstraction (MockBroker, FutuBroker, RetryingBroker); deps: utils
strategy    ← strategies; deps: data, utils
engine      ← backtest engines; deps: strategy, data, utils
analysis    ← pure-compute risk/perf modules; deps: engine, strategy, data, utils
                  ★ MUST NOT import live/ or broker/
live        ← live trading bridge (risk_controller, order_manager,
              kill_switch, decision_logger, risk_alerts); deps: broker,
              strategy, data, utils, analysis (read-only)
dashboard   ← Streamlit UI; deps: analysis, engine, data, broker, live, utils
scripts     ← cron entry; deps: most modules
```

No cycles.  Currently verified manually (see [Layer-import enforcement](#layer-import-enforcement) below);
CI gate via import-linter is planned but not yet wired.

### `analysis/` is pure compute

No I/O writes (no `cache.save_*`), no broker calls, no network.  Inputs
are `pd.Series` / `pd.DataFrame` / dicts; outputs are dicts / DataFrames.
This lets `analysis/` be unit-tested without DB or broker fixtures
(currently ~5300 LOC; the bulk of the suite).

**Known violation, slated for refactor**: `analysis/risk_report.py`
imports `live.position_stops.compute_hypothetical_positions` (2 sites).
The function itself is pure compute (config + target_date + provider →
positions list, no broker access) and is misplaced in `live/`.  Planned
fix: move to `analysis/hypothetical_positions.py`, keep a re-export in
`live/position_stops.py` for the other 7 callers.  Until then, this is
the only sanctioned `analysis → live` import.

### `live/` is the only writer of trading-state tables

| Table | Writers | Readers |
|-------|---------|---------|
| `ohlcv_daily`, `signal_history` | `data/` providers, `daily.py` | everywhere |
| `trade_pnl`, `order_log`, `slippage_log` | `live/order_manager` (via `live_trader.py`) | dashboard, analysis, reports |
| `decision_history` | `live/decision_logger` (via `live_trader.py` + KillSwitch) | dashboard `decision_review` |
| `alert_history` | `live/risk_alerts`, `live/kill_switch` | dashboard `alert_history` |
| `risk_state` | `live/risk_controller`, `live/kill_switch` | dashboard, daemon next tick |
| `entry_prices` | `live/risk_controller` | engine, analysis |
| `ops_log` | utility code across `live/`, `broker/` | dashboard `ops` tab |

`dashboard/`, `analysis/`, `scripts/` are **read-only** with respect to
all of the above, **with one principled exception** — see Kill Switch.

If you want to write a new table from `dashboard/` or `analysis/`,
reconsider.  The correct path is: dashboard writes a *request flag* to
`risk_state` → daemon next tick reads the flag → daemon performs the
action → daemon writes the table.

### Sanctioned exception: Kill Switch is dashboard→broker direct

`dashboard/kill_switch.py` constructs `KillSwitch(broker, ...)` directly
and calls `ks.trigger()`, which executes `broker.submit_order()` from
the **dashboard process** — bypassing the `live_trader` daemon entirely.

This is **by design**, not a layering violation:

- KillSwitch must work when the daemon is dead, hung, or unreachable.
  Routing it through the daemon would make daemon failure also disable
  the emergency exit — the single worst possible failure mode for a
  safety control.
- The `KillSwitch` class itself lives in `live/`, so dashboard depends
  on `live/`, not the reverse.  Layer direction is preserved.
- Audit trail flows the same way it would have through the daemon:
  `KillSwitch.trigger` writes to `alert_history` AND `decision_history`,
  and pauses the daemon via `risk_state` so the daemon won't open new
  positions when it next ticks.
- Locking is via SQLite + WAL; no in-process coordination needed.

**Do not generalise this pattern.**  KillSwitch is the only sanctioned
dashboard→broker write path.  Every other dashboard action must follow
the "write a request flag" pattern above.

### Layer-import enforcement

The rule distinguishes **broker value types** (`Order`, `OrderSide`,
`OrderStatus`, `OrderType` — pure dataclasses / enums) from **broker
implementations** (`Broker`, `MockBroker`, `FutuBroker`, `RetryingBroker`
— actual order-submitting code).  Value types are shared vocabulary;
engine and strategy may use them.  Implementations may only be touched
by `live/`, `dashboard/` (KillSwitch only), `scripts/`, and tests.

Until import-linter is wired into CI, these greps are the spot-checks:

```bash
# analysis/ MUST NOT import live/  (one known violation, slated for fix)
grep -rn "from live\|^import live\b" analysis/

# strategy/, data/, utils/ MUST NOT import live/ or broker/
grep -rn "from live\|^import live\b" strategy/ data/ utils/      # empty
grep -rn "from broker\|^import broker\b" strategy/ data/ utils/  # empty

# engine/ may import broker VALUE TYPES only — Order/OrderSide/OrderStatus/OrderType
# It must NOT import Broker / MockBroker / FutuBroker / RetryingBroker
grep -rn "Broker\b" engine/ \
  | grep -v "OrderSide\|OrderStatus\|OrderType\|order_type\|order side\|Order\b"
# (manual review — any line matching a *Broker class is a violation)

# Who may IMPORT broker — broader (need types/factory)
# Who may CALL broker.submit_order — narrower (only live_trader daemon + KillSwitch)
grep -rn "from broker\|^import broker\b" \
  | grep -v "^live/\|^dashboard/\|^scripts/\|^broker/\|^tests/\|^engine/\|^live_trader.py\|^daily.py\|^analysis/"
# should be empty

# Critical: broker.submit_order calls — should only appear in:
#   live/order_manager.py  (daemon path)
#   live/kill_switch.py    (emergency path)
#   broker/middleware.py   (RetryingBroker passthrough)
#   tests/                 (test fixtures)
grep -rn "\.submit_order\b" \
  | grep -v "^live/order_manager\|^live/kill_switch\|^broker/middleware\|^tests/\|^broker/base"
# should be empty
```

If you add an import that breaks these rules, you're crossing a layer
boundary — refactor or add a justification comment with a follow-up to
CI.  Don't suppress.

## Windows / Docker quirks

- `utils/env.py` fixes GBK encoding on Windows: `sys.stdout.reconfigure(encoding="utf-8")`
- Dockerfile uses `host.docker.internal` for FutuOpenD connection (Windows/Mac specific; Linux needs `--add-host`)
- Tsinghua PyPI mirror is the primary Pipfile source; CI overrides with `PIP_INDEX_URL=https://pypi.org/simple`

## Live trading workflow

1. Run all tests first: `pipenv run pytest tests/ -v`
2. Check golden tests pass (CI gate)
3. Dry-run with mock broker: `pipenv run python live_trader.py --broker mock`
4. Verify signals with daily scan: `pipenv run python daily.py`
5. Only then run with futu: `pipenv run python live_trader.py --broker futu --daemon`

## Dashboard

```bash
pipenv run streamlit run dashboard.py --server.port 8501 --server.headless true
```

Access at `http://localhost:8501`. **11 tabs**:

1. 单标的回测 — single-symbol backtest + risk-adjusted metrics
2. 组合回测 — portfolio backtest + PnL attribution + Monte Carlo expander
3. 因子归因 — 6-factor OLS + Newey-West HAC, Jensen α + β
4. 业绩归因 Brinson — sector allocation / selection / interaction
5. 盈亏分析 — Realized + Unrealized PnL breakdown
6. 信号有效性 — Forward return distribution
7. 风险量化 — VaR + EVT + Stress + Concentration + Correlation + Marginal VaR + What-If
8. 风险告警历史 — alert timeline + per-type daily bar chart
9. 📑 风险报告 — 9-section weekly report + Feishu push
10. 🚨 Kill Switch — emergency liquidation (manual + double confirm + dry-run)
11. 配置管理 — watchlist editor

## Weekly risk report (cron)

```bash
# Manual run + Feishu push
pipenv run python scripts/weekly_risk_report.py

# Dry-run (Markdown only)
pipenv run python scripts/weekly_risk_report.py --dry-run

# Cron schedule
# 0 9 * * 1 cd /path/to/traderbridge && pipenv run python scripts/weekly_risk_report.py
```

## Proxy config (China)

Yahoo + GitHub are unreachable from mainland China without a proxy.
Add to `.env` (gitignored):

```
HTTPS_PROXY=http://127.0.0.1:7897
HTTP_PROXY=http://127.0.0.1:7897
NO_PROXY=localhost,127.0.0.1
```

Pipenv loads `.env` automatically.  Git pushes use SSH (`git@github.com`)
to bypass HTTPS proxy issues.
