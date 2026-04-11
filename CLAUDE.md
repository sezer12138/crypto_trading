# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto trading backtesting system with 13 built-in strategies, historical data fetching from Binance, and visualization. All documentation and comments are in English.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (default: BTC, multi_factor strategy, 2 years, 1h interval)
python run_backtest.py
python run_backtest.py --coin eth --strategy ma_cross
python run_backtest.py --coin all --compare          # Compare all strategies
python run_backtest.py --interval 4h --days 365 --capital 50000

# Real-time data fetching
python src/main.py                    # REST polling mode
python src/main.py --websocket        # WebSocket streaming

# Tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
pytest tests/ -v -m "not slow and not integration"

# Format
black src/ tests/
```

## Architecture

Four-layer pipeline: **Data → Strategy → Backtest → Visualization**

### Data Layer
- `src/historical_data.py` — Fetches Binance K-line data (BTC/ETH/SOL, intervals: 1m–1d). Outputs DataFrame with columns: timestamp, open, high, low, close, volume.
- `src/data_fetcher.py` — REST API fetcher (CoinGecko/Binance) for real-time prices.
- `src/websocket_client.py` — WebSocket streaming from Binance.

### Strategy Layer (`src/strategies/` package)
- Base class `TradingStrategy` in `_base.py` with `generate_signals(df) → df` (adds signal column: 1=buy, -1=sell, 0=hold).
- Factory function `get_strategy(name, **kwargs)` in `__init__.py` registers all strategies.
- 13 strategies in separate modules: ma_cross, rsi, bollinger, multi_factor, mean_reversion, macd, breakout, vwap, momentum, atr_stop, stochastic, grid, martingale.
- Named constants in `constants.py` for all strategy parameters.
- Shared helpers in `_helpers.py`: `forward_fill_position()`, `detect_crossover()`, `calculate_rsi()`.
- To add a strategy: create new module in `src/strategies/`, subclass `TradingStrategy`, implement `generate_signals()`, add to `get_strategy()` dict in `__init__.py`.

### Backtest Layer (`src/backtest.py`)
- `BacktestEngine(initial_capital, commission_rate, slippage, position_size)` simulates trading.
- Returns `BacktestResult` dataclass with trades, equity_curve, daily_returns, metrics (total_return, annual_return, sharpe_ratio, max_drawdown, win_rate, total_trades), and decision_log.

### Visualization Layer (`src/visualization/` package)
- `Visualizer` class composed from mixin classes in `__init__.py`.
- Mixin modules: `equity.py`, `price_signals.py`, `monthly.py`, `comparison.py`, `report.py`.
- Base class `VisualizerBase` in `_base.py` with `_save_figure()` helper.
- Named constants in `_constants.py` for figure sizes, DPI, color thresholds.

## Key Conventions

- Code style: Black formatter (line-length=100), type hints, Google-style docstrings.
- **All comments, docstrings, and log messages must be in English only — no Chinese.**
- Strategy signals use numeric codes: 1 (buy), -1 (sell), 0 (hold).
- Config at `config/settings.yaml` for data sources, logging, storage paths.
- Backtest logs saved as JSON in `logs/`, chart outputs in `results/`.

## Network Configuration (for restricted regions)

Binance API may be geo-blocked (HTTP 451) in some regions (e.g. China). Use these environment variables to configure proxy and alternative endpoints:

```bash
# Set HTTP proxy (supports HTTP/HTTPS/SOCKS5)
export CRYPTO_PROXY=http://127.0.0.1:7890

# Or use standard proxy env vars (also supported)
export HTTPS_PROXY=http://127.0.0.1:7890

# Override Binance API base URL (if using a mirror)
export CRYPTO_BINANCE_BASE=https://api1.binance.com/api/v3

# Override WebSocket base URL
export CRYPTO_WS_BASE=wss://stream.binance.com:443

# Disable SSL verification (not recommended for production)
export CRYPTO_DISABLE_SSL=1

# Run with proxy configured
export CRYPTO_PROXY=http://127.0.0.1:7890
python run_backtest.py --coin btc --strategy ma_cross --no-viz
```
