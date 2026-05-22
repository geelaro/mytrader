# AGENTS.md — mytrader

## Prerequisites

- **Python 3.10+** (enforced by Pipfile)
- Package manager: **Pipenv** (not pip/poetry/uv)
- Install: `pipenv install --dev`

## Verify

```bash
pipenv run pytest tests/ -v                    # full suite
pipenv run pytest tests/test_golden.py -v       # CI gate only
pipenv run pytest tests/test_strategy.py -v     # single file
pipenv run pytest tests/test_strategy.py::TestWeeklyMACDKDJ -v  # single class
```

No lint, formatter, or typechecker is configured. Tests are the only verification step.

## Project architecture

Flat single-package layout — everything is at the top level. No `src/` layout, no namespace packages.

| Directory | Role |
|-----------|------|
| `data/` | Market data pipeline (sources → cache → provider) |
| `strategy/` | Strategy library (ABC + 10 implementations) |
| `broker/` | Broker abstraction (MockBroker + FutuBroker) |
| `engine/` | Backtest engines (single-symbol + portfolio + optimization) |
| `analysis/` | Offline analysis tools (robustness, sensitivity, Monte Carlo) |
| `utils/` | Logging, notifications, market state, signal gate, sectors |
| `tests/` | 485 tests, all offline (synthetic data) |

Entry points: `live_trader.py`, `daily.py`, `dashboard.py`. None are importable as libraries — they're scripts.

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
- Test DB: `conftest.py` sets `MYTRADER_DB` env var to a temp file before any import to isolate from production.
- `data/cache.py` — `CacheManager` reads `MYTRADER_DB` env var; falls back to `trading_data.db`.
- DB tables: `ohlcv_daily`, `signals`, `order_log`, `slippage_log`, `ops_log`, `entry_prices`, `trade_pnl`, `risk_state`.

## Known traps

- **`kdj_d=1`** — causes zero trades. Documented in README, don't set it.
- **`bollinger_mean_reversion`** and **`bollinger_squeeze`** — rated 0 stars, known to produce zero trades. Don't use for live.
- **`enhanced_macd`** — rated 0 stars, documented as overfitted. Don't use for live.
- **Streamlit requires `--server.headless true`** on servers/headless machines.

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

Access at `http://localhost:8501`. Has tabs for single-symbol backtest, portfolio backtest, risk dashboard, Monte Carlo, and live trade history.
