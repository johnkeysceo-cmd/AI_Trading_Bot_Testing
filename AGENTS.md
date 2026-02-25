## Cursor Cloud specific instructions

### Project Overview

This is **Monster Crypto AI Hub**, a multi-agent autonomous cryptocurrency trading system in Python. The core project lives in `crypto_ai_command_hub_new/`. Vendored external bots (Freqtrade, Jesse, OctoBot, OpenTrader, RLTrader, AI-CryptoTrader) are under `agents/external/` and are not required for core functionality.

### Running Core Entry Points

All commands should be run from `crypto_ai_command_hub_new/`:

- **Backtest**: `python3 run_phase2_backtest.py` — runs Phase 1 vs Phase 2 comparison with synthetic data. Completes in <1 second.
- **ULTIMATE Bot simulation**: `python3 -c "import sys; sys.path.insert(0,'.'); from models.ULTIMATE_10_10_TRADING_BOT import ULTIMATE_10_10_TradingBot; bot = ULTIMATE_10_10_TradingBot(initial_capital=10000, use_real_data=False, lookback_days=30); bot.run_simulation()"`
- **Live runner (paper mode)**: `python3 scripts/live_runner.py --symbols BTC ETH --lookback-days 7 --interval 30 --capital 500` — polls CoinGecko for real prices. Press Ctrl+C to stop.

### Key Gotchas

1. **CCXT version**: Must use `ccxt==3.1.60` (not latest). Newer CCXT versions removed `coinbasepro` exchange which `execution/ccxt_executor.py` references. Current CCXT 4.x only has `coinbase`.

2. **MetaAgent / test_meta_agent.py is broken**: The `agents/meta_agent/agent_selector.py` imports all adapter modules at module level. Several adapters import vendored external bots that have incompatible internal imports (e.g., `opentrader` is a Node.js project, jesse's vendored `__init__.py` triggers cascading import issues). This is a pre-existing codebase issue.

3. **Empty tests**: `tests/test_agents.py`, `test_data_integrity.py`, `test_execution.py`, `test_risk.py` are all empty files. Only `test_meta_agent.py` has content but cannot run (see #2).

4. **No editable install**: `pip install -e .` fails due to flat-layout package discovery (multiple top-level packages). Install dependencies directly instead.

5. **numpy version**: Use `numpy==2.2.6` for best compatibility across the dependency tree (numba, pandas-ta, etc.).

### Linting

No linter is configured for the core project. Use `ruff check` for ad-hoc linting. The `pyproject.toml` does not include ruff/flake8/mypy config (only the vendored freqtrade has linter config).

### Configuration

- Trading mode: `config/trading_mode.py` — defaults to `"paper"` (safe, no real money).
- API keys: `config/api_keys.json` — contains placeholder keys for Binance, Kraken, CoinbasePro. Only used in paper mode (no real API calls).
- CoinGecko API: Used by `scripts/live_runner.py` to fetch real market prices. Works without an API key (rate-limited).
