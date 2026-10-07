# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Crypto trading backtesting system with 13 built-in strategies, historical data fetching from Binance/OKX, and visualization. All documentation, comments, and log messages are in English only.

**Primary reference**: `src/strategy_and_backtest_tutorial.md` is the canonical deep-dive for signal patterns, indicator math, and risk management. Read it before implementing or modifying a strategy.

## Commands

```bash
# Install
pip install -r requirements.txt

# Backtest (default: BTC, multi_factor strategy, 730 days, 1h interval)
python run_backtest.py
python run_backtest.py --coin eth --strategy ma_cross
python run_backtest.py --coin all --compare              # Compare all strategies
python run_backtest.py --interval 4h --days 365 --capital 50000
python run_backtest.py --source okx --no-viz             # OKX data, no charts

# Real-time data
python src/main.py                    # REST polling
python src/main.py --websocket        # WebSocket streaming

# Tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
pytest tests/ -v -m "not slow and not integration"
pytest tests/test_risk_management.py::TestStopLoss -v   # Single test class

# Format
black src/ tests/ --line-length 100
```

## Architecture

Four-layer pipeline: **Data → Strategy → Backtest → Visualization**

### Data Layer
- `src/historical_data.py` — Fetches Binance/OKX K-line data (BTC/ETH/SOL, intervals 1m–1d). Outputs DataFrame with columns: timestamp, open, high, low, close, volume.
- `src/data_fetcher.py` — REST API fetcher (CoinGecko/Binance) for real-time prices.
- `src/websocket_client.py` — Binance WebSocket streaming.

### Strategy Layer (`src/strategies/`)
- Base class `TradingStrategy` in `_base.py` — implement `generate_signals(df) → df` (adds `signal` column: 1=buy, -1=sell, 0=hold).
- Factory `get_strategy(name, **kwargs)` in `__init__.py` registers all 13 strategies.
- Strategies: ma_cross, rsi, bollinger, multi_factor, mean_reversion, macd, breakout, vwap, momentum, atr_stop, stochastic, grid, martingale.
  - Notable: `VWAPStrategy` uses ATR-based dynamic deviation (`VWAP_DYNAMIC_MULTIPLIER=1.5`, `VWAP_MIN_DEVIATION=0.005`) instead of a fixed percentage — it automatically tightens/widens bands with volatility.
  - Notable: `GridStrategy` is a range-trading strategy fed lower/upper bounds from the first 100 bars (plus 10% margin) by `run_backtest.py` — avoids look-ahead bias.
- Strategy hyperparameters live in `constants.py` — no magic numbers in strategy modules. (Backtest-level params like `commission_rate`, `slippage`, `position_size` are hardcoded in `BacktestEngine.__init__()`; only risk-management knobs flow from `constants.py`.)
- Shared helpers in `_helpers.py`:
  - `forward_fill_position(df)` — derive `position` column from `signal`.
  - `detect_crossover(df, fast_col, slow_col)` — golden/death cross signals.
  - `calculate_rsi(prices, period)` — RSI computation.
  - `convert_to_event_signals(df)` — collapse repeated state signals into single events.
  - `add_trend_filter(df, trend_window, trend_tolerance)` — suppress mean-reversion signals in strong trends.

**Critical signal pattern**: emit **event-based** signals, not state-based. Setting `signal=1` on every bar where a condition is true causes massive over-trading (we saw -90% returns from this bug). Use `convert_to_event_signals()` to keep only the first bar of each state change. Mean-reversion strategies (RSI, Bollinger, VWAP, mean_reversion) should also use `add_trend_filter()` to skip signals in trending markets.

**To add a strategy**: create `src/strategies/<name>.py`, subclass `TradingStrategy`, call `convert_to_event_signals()` at the end of `generate_signals()`, add defaults to `constants.py`, register in `get_strategy()` dict.

### Backtest Layer (`src/backtest.py`)
`BacktestEngine` simulates trading and returns a `BacktestResult` dataclass (trades, equity_curve, daily_returns, metrics, decision_log).

Risk management knobs and engine params:
- `commission_rate=0.001` (hardcoded), `slippage=0.001` (hardcoded), `position_size=0.95` (hardcoded)
- `min_holding_bars=5` (from `constants.py`) — block sells before N bars after entry.
- `max_trades_per_day=6` (from `constants.py`) — daily trade cap, resets on new day.
- `stop_loss_pct=0.05` (from `constants.py`) — auto-sell if drawdown from entry ≥ 5%.
- `max_drawdown_pct=0.20` (from `constants.py`) — circuit breaker that halts trading once equity drawdown ≥ 20%.
- `log_decisions=False` (hardcoded) — when `True`, appends a per-bar dict to `decision_log` (~N allocations; defaults to `False` for performance).

Signal values: `1` (buy), `-1` (sell), `0` (hold), `-2` (`FORCED_SELL_SIGNAL` — marks stop-loss / end-of-data forced liquidations in trade records).

Metrics: `total_return_pct`, `annual_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate_pct`, `total_trades`, plus cost analysis (`total_cost`, `cost_drag`).

### Visualization Layer (`src/visualization/`)
- `Visualizer` class composed via mixins (`equity`, `price_signals`, `monthly`, `comparison`, `report`).
- `VisualizerBase` in `_base.py` provides the `_save_figure()` helper.
- `html_report.py` — standalone `HTMLReportGenerator` for single-strategy and comparison HTML reports (used by `run_backtest.py`).
- Constants in `_constants.py` (figure sizes, DPI, color thresholds).

## Test Organization

- `tests/test_crypto_trading.py` — core data fetcher, strategy, backtest, and end-to-end pipeline tests.
- `tests/test_extended.py` — extended strategy coverage (grid, martingale, ATR stop, MACD, breakout, VWAP dynamic deviation, momentum, stochastic) and HTML report rendering.
- `tests/test_risk_management.py` — `BacktestEngine` risk features (min holding, daily trade cap, stop-loss, drawdown circuit breaker, cost analysis, reset behavior).

## Key Conventions

- Black formatter (line-length=100), full type hints, Google-style docstrings. Config in `pyproject.toml`.
- **All comments, docstrings, and log messages in English only — no Chinese.** (Note: `README.md` is the exception — it's a bilingual project introduction for a Chinese-speaking audience. Write new documentation in English.)
- Strategy signals: 1 (buy), -1 (sell), 0 (hold). Use event-based signals, not state-based.
- Backtest logs as JSON in `logs/`, charts and HTML reports in `results/`.
- Config at `config/settings.yaml` for data sources, logging, storage paths.

## Network Configuration (restricted regions)

Binance is geo-blocked (HTTP 451) in some regions. OKX is the recommended fallback (no proxy needed):

```bash
# OKX as data source
python run_backtest.py --source okx --coin btc --strategy ma_cross --no-viz

# Or set as default
export CRYPTO_DATA_SOURCE=okx
```

Proxy-based access to Binance:

```bash
export CRYPTO_PROXY=http://127.0.0.1:7890     # HTTP/HTTPS/SOCKS5
export HTTPS_PROXY=http://127.0.0.1:7890      # Standard env var also supported
export CRYPTO_BINANCE_BASE=https://api1.binance.com/api/v3   # Mirror endpoint
export CRYPTO_WS_BASE=wss://stream.binance.com:443
export CRYPTO_DISABLE_SSL=1                    # Not for production
```

The fetcher auto-detects an unreachable proxy and falls back to direct connection.
