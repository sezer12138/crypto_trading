# Strategy & Backtest Tutorial

A complete guide to every built-in trading strategy and how to run backtests.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Strategy Base Class](#strategy-base-class)
- [Signal Convention](#signal-convention)
- [Strategy Categories](#strategy-categories)
- [Trend-Following Strategies](#trend-following-strategies)
  - [1. MA Cross (ma_cross)](#1-ma-cross-ma_cross)
  - [2. MACD (macd)](#2-macd-macd)
  - [3. Breakout (breakout)](#3-breakout-breakout)
  - [4. Momentum (momentum)](#4-momentum-momentum)
  - [5. ATR Stop-Loss (atr_stop)](#5-atr-stop-loss-atr_stop)
- [Mean-Reversion Strategies](#mean-reversion-strategies)
  - [6. RSI (rsi)](#6-rsi-rsi)
  - [7. Bollinger Bands (bollinger)](#7-bollinger-bands-bollinger)
  - [8. Mean Reversion (mean_reversion)](#8-mean-reversion-mean_reversion)
  - [9. Stochastic (stochastic)](#9-stochastic-stochastic)
  - [10. VWAP (vwap)](#10-vwap-vwap)
- [Composite & Special Strategies](#composite--special-strategies)
  - [11. Multi-Factor (multi_factor)](#11-multi-factor-multi_factor)
  - [12. Grid (grid)](#12-grid-grid)
  - [13. Martingale (martingale)](#13-martingale-martingale)
- [Strategy Helpers](#strategy-helpers)
  - [detect_crossover](#detect_crossover)
  - [convert_to_event_signals](#convert_to_event_signals)
  - [add_trend_filter](#add_trend_filter)
  - [forward_fill_position](#forward_fill_position)
  - [calculate_rsi](#calculate_rsi)
- [Running a Backtest](#running-a-backtest)
  - [CLI Usage](#cli-usage)
  - [Python API Usage](#python-api-usage)
  - [Backtest Engine Internals](#backtest-engine-internals)
  - [Risk Management](#risk-management)
  - [Performance Metrics](#performance-metrics)
- [Choosing a Strategy](#choosing-a-strategy)

---

## Architecture Overview

The system follows a four-layer pipeline:

```
Data Layer --> Strategy Layer --> Backtest Layer --> Visualization Layer
```

1. **Data Layer** (`src/historical_data.py`) fetches OHLCV candles from Binance or OKX.
2. **Strategy Layer** (`src/strategies/`) computes technical indicators and emits buy/sell signals.
3. **Backtest Layer** (`src/backtest.py`) simulates trading with commission, slippage, and risk controls.
4. **Visualization Layer** (`src/visualization/`) renders equity curves, comparison charts, and HTML reports.

---

## Strategy Base Class

All strategies inherit from `TradingStrategy` in `src/strategies/_base.py`:

```python
class TradingStrategy:
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
```

The only required method is `generate_signals(df)`, which must:
- Accept a DataFrame with columns: `open`, `high`, `low`, `close`, `volume`
- Return the same DataFrame with at least a `signal` column added (and optionally `position`)
- Set `signal` to **1** (buy), **-1** (sell), or **0** (hold)

---

## Signal Convention

| Value | Meaning |
|-------|---------|
| `1`   | Buy signal |
| `-1`  | Sell signal |
| `0`   | Hold / no signal |

Strategies use one of two signal patterns:

| Pattern | Description | Example Strategies |
|---------|-------------|-------------------|
| **Event-based** | Signal fires only on the first bar where a condition becomes true (e.g., a crossover). Subsequent bars with the same condition emit `0`. | ma_cross, macd, momentum |
| **Threshold-based** | Signal fires on every bar where a condition is true. These strategies call `convert_to_event_signals()` to collapse consecutive identical signals into a single event. | mean_reversion, vwap, multi_factor |

The backtest engine reads the `signal` column on every bar. It buys when `signal == 1` and it has no position, and sells when `signal == -1` and it holds a position. Consecutive buy signals while already in position are ignored, and vice versa.

---

## Strategy Categories

| Category | Strategies | Best For |
|----------|-----------|----------|
| **Trend-Following** | ma_cross, macd, breakout, momentum, atr_stop | Markets with clear directional trends |
| **Mean-Reversion** | rsi, bollinger, mean_reversion, stochastic, vwap | Sideways / range-bound markets |
| **Composite** | multi_factor | All conditions (combines trend + mean-reversion factors) |
| **Special** | grid, martingale | Specific scenarios (sideways grid, high-risk doubling) |

---

## Trend-Following Strategies

### 1. MA Cross (`ma_cross`)

**File**: `src/strategies/ma_cross.py`

**How it works**: Compares a short-period simple moving average (SMA) against a long-period SMA. When the short SMA crosses above the long SMA, it signals upward momentum (Golden Cross). When it crosses below, it signals downward momentum (Death Cross).

**Buy condition** (Golden Cross):
```
ma_short[i] > ma_long[i]  AND  ma_short[i-1] <= ma_long[i-1]
```

**Sell condition** (Death Cross):
```
ma_short[i] < ma_long[i]  AND  ma_short[i-1] >= ma_long[i-1]
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `short_window` | 10 | Short-term SMA window |
| `long_window` | 30 | Long-term SMA window |

**Code reference**:
```python
# src/strategies/ma_cross.py:53-57 — Indicator calculation
df["ma_short"] = df["close"].rolling(window=self.short_window).mean()
df["ma_long"] = df["close"].rolling(window=self.long_window).mean()

# src/strategies/ma_cross.py:72-73 — Signal generation via crossover helper
df = detect_crossover(df, "ma_short", "ma_long")
```

**Usage**:
```python
from strategies import get_strategy
strategy = get_strategy("ma_cross", short_window=10, long_window=30)
df = strategy.generate_signals(df)
```

---

### 2. MACD (`macd`)

**File**: `src/strategies/macd.py`

**How it works**: MACD (Moving Average Convergence Divergence) uses the difference between a fast and slow exponential moving average (EMA). This difference line is then smoothed with a signal line (another EMA). Crossovers between the MACD line and the signal line generate trading signals.

**Indicators**:
```
ema_fast  = EMA(close, fast_period)       # default period=12
ema_slow  = EMA(close, slow_period)       # default period=26
macd      = ema_fast - ema_slow
macd_signal = EMA(macd, signal_period)     # default period=9
macd_hist = macd - macd_signal            # histogram (momentum strength)
```

**Buy condition**: MACD line crosses above the signal line.
**Sell condition**: MACD line crosses below the signal line.

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `fast` | 12 | Fast EMA period |
| `slow` | 26 | Slow EMA period |
| `signal` | 9 | Signal line EMA period |

**Code reference**:
```python
# src/strategies/macd.py:60-65 — Indicator calculation
df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
df["macd"] = df["ema_fast"] - df["ema_slow"]
df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()

# src/strategies/macd.py:81 — Signal via crossover
df = detect_crossover(df, "macd", "macd_signal")
```

---

### 3. Breakout (`breakout`)

**File**: `src/strategies/breakout.py`

**How it works**: Identifies when price breaks out of its recent N-period range. In confirmation mode (default), a breakout is only triggered when the **close** price exceeds the previous bar's N-period high/low. In instant mode, the **high/low** price triggers the signal immediately.

**Buy condition** (confirmation mode):
```
close[i] > high_n[i-1]  AND  close[i-1] <= high_n[i-2]
```
where `high_n = rolling_max(high, window)`

**Sell condition** (confirmation mode):
```
close[i] < low_n[i-1]  AND  close[i-1] >= low_n[i-2]
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Lookback window for high/low calculation |
| `confirmation` | `True` | Require close-price confirmation |

**Code reference**:
```python
# src/strategies/breakout.py:53-54 — Rolling high/low
df["high_n"] = df["high"].rolling(window=self.window).max()
df["low_n"] = df["low"].rolling(window=self.window).min()

# src/strategies/breakout.py:59-63 — Confirmation mode buy signal
df.loc[
    (df["close"] > df["high_n"].shift(1))
    & (df["close"].shift(1) <= df["high_n"].shift(2)),
    "signal",
] = 1
```

---

### 4. Momentum (`momentum`)

**File**: `src/strategies/momentum.py`

**How it works**: Combines two momentum indicators — Rate of Change (ROC) and raw momentum — and requires both to agree. A buy fires only when ROC turns positive (crosses above the threshold) AND momentum is positive.

**Indicators**:
```
roc            = (close - close[roc_period]) / close[roc_period]
momentum       = close - close[momentum_period]
momentum_norm  = momentum / close * 100
```

**Buy condition**:
```
roc > threshold  AND  momentum_norm > 0  AND  roc[i-1] <= threshold
```

**Sell condition**:
```
roc < -threshold  AND  momentum_norm < 0  AND  roc[i-1] >= -threshold
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `roc_period` | 10 | ROC lookback period |
| `momentum_period` | 14 | Momentum lookback period |
| `threshold` | 0.02 | ROC threshold (2%) to filter noise |

**Code reference**:
```python
# src/strategies/momentum.py:56-60 — Indicator calculation
shifted = df["close"].shift(self.roc_period)
df["roc"] = (df["close"] - shifted) / shifted.replace(0, float("nan"))
df["momentum"] = df["close"] - df["close"].shift(self.momentum_period)
df["momentum_norm"] = df["momentum"] / df["close"] * 100

# src/strategies/momentum.py:79-84 — Buy signal
df.loc[
    (df["roc"] > self.threshold)
    & (df["momentum_norm"] > 0)
    & (df["roc"].shift(1) <= self.threshold),
    "signal",
] = 1
```

---

### 5. ATR Stop-Loss (`atr_stop`)

**File**: `src/strategies/atr_stop.py`

**How it works**: Uses the Average True Range (ATR) to define dynamic support/resistance levels. In an uptrend (price above trend MA), it looks for pullbacks where the low touches the ATR-based support line. In a downtrend (price below trend MA), it looks for bounces where the high touches the ATR-based resistance line.

**Indicators**:
```
tr        = max(high-low, |high-prev_close|, |low-prev_close|)
atr       = rolling_mean(tr, atr_period)
trend_ma  = SMA(close, trend_ma_period)
support   = close - atr * multiplier
resistance = close + atr * multiplier
```

**Buy condition** (uptrend pullback to support):
```
close > trend_ma  AND  low < support[i-1]  AND  close > support[i-1]
```

**Sell condition** (downtrend bounce to resistance):
```
close < trend_ma  AND  high > resistance[i-1]  AND  close < resistance[i-1]
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `multiplier` | 2.0 | ATR multiplier for support/resistance bands |
| `trend_ma` | 50 | Trend determination MA period |

**Code reference**:
```python
# src/strategies/atr_stop.py:59-63 — ATR calculation
high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift(1)).abs()
low_close = (df["low"] - df["close"].shift(1)).abs()
tr = np.maximum(np.maximum(high_low, high_close), low_close)
return tr.rolling(window=self.atr_period).mean()

# src/strategies/atr_stop.py:84-97 — Signal generation
uptrend = df["close"] > df["trend_ma"]
support = df["close"] - df["atr"] * self.multiplier
df.loc[uptrend & (df["low"] < support.shift(1)) & (df["close"] > support.shift(1)), "signal"] = 1
```

---

## Mean-Reversion Strategies

### 6. RSI (`rsi`)

**File**: `src/strategies/rsi.py`

**How it works**: RSI (Relative Strength Index) measures the speed of price changes on a 0-100 scale. Values below 30 indicate oversold conditions; values above 70 indicate overbought conditions. The strategy buys when RSI crosses back above the oversold line and sells when it crosses back below the overbought line.

**Buy condition** (RSI exits oversold):
```
rsi > oversold  AND  rsi[i-1] <= oversold     # default oversold=30
```

**Sell condition** (RSI exits overbought):
```
rsi < overbought  AND  rsi[i-1] >= overbought  # default overbought=70
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `period` | 14 | RSI calculation period |
| `oversold` | 30 | Oversold threshold |
| `overbought` | 70 | Overbought threshold |
| `trend_filter_enabled` | `False` | Suppress signals in strong trends |
| `trend_filter_window` | 50 | Trend filter MA window |
| `trend_filter_tolerance` | 0.03 | Max deviation from MA (3%) |

**Code reference**:
```python
# src/strategies/_helpers.py:99-106 — RSI calculation
delta = prices.diff()
gain = delta.clip(lower=0).rolling(window=period).mean()
loss = (-delta.clip(upper=0)).rolling(window=period).mean()
rs = gain / loss.replace(0, np.nan)
rsi = 100 - (100 / (1 + rs))

# src/strategies/rsi.py:79 — Buy signal
df.loc[(df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold), "signal"] = 1

# src/strategies/rsi.py:88-89 — Trend filter (optional)
if self.trend_filter_enabled:
    df = add_trend_filter(df, self.trend_filter_window, self.trend_filter_tolerance)
    df.loc[~df["trend_filter"], "signal"] = 0
```

---

### 7. Bollinger Bands (`bollinger`)

**File**: `src/strategies/bollinger.py`

**How it works**: Bollinger Bands consist of a middle band (SMA) and upper/lower bands at +/- N standard deviations. When price breaks below the lower band and recovers, it signals a buy. When price breaks above the upper band and retreats, it signals a sell.

**Indicators**:
```
middle_band = SMA(close, window)         # default window=20
std         = rolling_std(close, window)
upper_band  = middle_band + std * num_std  # default num_std=2.0
lower_band  = middle_band - std * num_std
```

**Buy condition** (price recovers above lower band):
```
close > lower_band  AND  close[i-1] <= lower_band[i-1]
```

**Sell condition** (price retreats below upper band):
```
close < upper_band  AND  close[i-1] >= upper_band[i-1]
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Moving average / band calculation window |
| `num_std` | 2.0 | Standard deviation multiplier |
| `trend_filter_enabled` | `True` | Suppress signals in strong trends |
| `trend_filter_window` | 50 | Trend filter MA window |
| `trend_filter_tolerance` | 0.03 | Max deviation from MA (3%) |

**Code reference**:
```python
# src/strategies/bollinger.py:54-58 — Band calculation
df["middle_band"] = df["close"].rolling(window=self.window).mean()
df["std"] = df["close"].rolling(window=self.window).std()
df["upper_band"] = df["middle_band"] + (df["std"] * self.num_std)
df["lower_band"] = df["middle_band"] - (df["std"] * self.num_std)

# src/strategies/bollinger.py:79-82 — Buy signal (bounce off lower band)
df.loc[
    (df["close"] > df["lower_band"]) & (df["close"].shift(1) <= df["lower_band"].shift(1)),
    "signal",
] = 1
```

---

### 8. Mean Reversion (`mean_reversion`)

**File**: `src/strategies/mean_reversion.py`

**How it works**: Calculates the Z-score of price relative to its rolling mean. When the Z-score drops below -entry_z (price is significantly below average), it buys. When the Z-score rises above +entry_z (price is significantly above average), it sells. The exit_z parameter closes the position when price reverts to within the exit zone. Signals are converted to event-based to prevent over-trading.

**Indicators**:
```
mean   = SMA(close, window)            # default window=20
std    = rolling_std(close, window)
zscore = (close - mean) / std
```

**Buy condition**: `zscore < -entry_z` (default entry_z=2.0)
**Sell condition**: `zscore > entry_z`
**Exit condition**: `abs(zscore) < exit_z` (default exit_z=0.5, overrides buy/sell)

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Rolling mean/std window |
| `entry_z` | 2.0 | Z-score entry threshold |
| `exit_z` | 0.5 | Z-score exit threshold |
| `trend_filter_enabled` | `True` | Suppress signals in strong trends |
| `trend_filter_window` | 50 | Trend filter MA window |
| `trend_filter_tolerance` | 0.03 | Max deviation from MA (3%) |

**Code reference**:
```python
# src/strategies/mean_reversion.py:98-102 — Z-score calculation
df["mean"] = df["close"].rolling(window=self.window).mean()
df["std"] = df["close"].rolling(window=self.window).std()
df["zscore"] = (df["close"] - df["mean"]) / df["std"].replace(0, float("nan"))

# src/strategies/mean_reversion.py:108-114 — Signal generation
df.loc[df["zscore"] < -self.entry_z, "signal"] = 1    # oversold buy
df.loc[df["zscore"] > self.entry_z, "signal"] = -1     # overbought sell
df.loc[abs(df["zscore"]) < self.exit_z, "signal"] = 0  # exit on reversion

# src/strategies/mean_reversion.py:117 — Convert to event-based signals
df = convert_to_event_signals(df)
```

---

### 9. Stochastic (`stochastic`)

**File**: `src/strategies/stochastic.py`

**How it works**: The Stochastic Oscillator measures where the current close sits within the recent N-bar range (0-100 scale). The %K line is the raw value; the %D line is a smoothed version. Buy when %K crosses above %D in the oversold zone (< 20). Sell when %K crosses below %D in the overbought zone (> 80).

**Indicators**:
```
k = 100 * (close - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period))
d = SMA(k, d_period)
```

**Buy condition**:
```
k > d  AND  k[i-1] <= d[i-1]  AND  k < 20     # crossover in oversold zone
```

**Sell condition**:
```
k < d  AND  k[i-1] >= d[i-1]  AND  k > 80     # crossover in overbought zone
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `k_period` | 14 | %K calculation period |
| `d_period` | 3 | %D smoothing period |
| `smooth` | 3 | %K pre-smoothing period |
| `trend_filter_enabled` | `True` | Suppress signals in strong trends |
| `trend_filter_window` | 50 | Trend filter MA window |
| `trend_filter_tolerance` | 0.03 | Max deviation from MA (3%) |

**Code reference**:
```python
# src/strategies/stochastic.py:59-62 — %K and %D calculation
lowest_low = df["low"].rolling(window=self.k_period).min()
highest_high = df["high"].rolling(window=self.k_period).max()
df["k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
df["d"] = df["k"].rolling(window=self.d_period).mean()

# src/strategies/stochastic.py:91-93 — Buy signal
df.loc[
    (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1)) & (df["k"] < 20),
    "signal",
] = 1
```

---

### 10. VWAP (`vwap`)

**File**: `src/strategies/vwap.py`

**How it works**: VWAP (Volume Weighted Average Price) is the volume-weighted average price over a rolling window. When price deviates significantly below VWAP, the strategy expects a reversion upward (buy). When price deviates significantly above VWAP, it expects a reversion downward (sell). The deviation threshold is dynamic by default, based on ATR, adapting to current market volatility.

**Indicators**:
```
typical_price = (high + low + close) / 3
vwap          = rolling_sum(typical_price * volume, window) / rolling_sum(volume, window)
vwap_dev      = (close - vwap) / vwap

# Dynamic deviation (when dynamic_deviation=True):
atr           = rolling_mean(high - low, atr_window)
dynamic_dev   = max(atr / close * 1.5, 0.005)   # 1.5x ATR, floored at 0.5%
```

**Buy condition**: `vwap_dev < -deviation` (price significantly below VWAP)
**Sell condition**: `vwap_dev > deviation` (price significantly above VWAP)

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | VWAP calculation window |
| `deviation` | 0.01 | Fixed deviation threshold (1%), used when `dynamic_deviation=False` |
| `dynamic_deviation` | `True` | Use ATR-based dynamic threshold |
| `atr_window` | 20 | ATR window for dynamic deviation |

**Code reference**:
```python
# src/strategies/vwap.py:85-88 — VWAP calculation
typical_price = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (typical_price * df["volume"]).rolling(window=self.window).sum() / \
             df["volume"].rolling(window=self.window).sum()
df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, float("nan"))

# src/strategies/vwap.py:122-128 — Dynamic signal generation
df.loc[df["vwap_dev"] < -dev_threshold, "signal"] = 1
df.loc[df["vwap_dev"] > dev_threshold, "signal"] = -1

# src/strategies/vwap.py:135 — Convert to event-based
df = convert_to_event_signals(df)
```

---

## Composite & Special Strategies

### 11. Multi-Factor (`multi_factor`)

**File**: `src/strategies/multi_factor.py`

**How it works**: Combines four independent factors into a single composite score, then generates buy/sell signals based on score thresholds. Each factor contributes a weighted portion:

| Factor | Weight | Logic |
|--------|--------|-------|
| MA Trend | 30% | +0.3 if short MA > long MA, -0.3 if below |
| RSI | 30% | RSI normalized to [-1, 1], scaled by weight |
| Volume | 20% | +0.2 if volume ratio > threshold, -0.2 if below |
| Volatility | 20% | -0.2 if volatility > 70th percentile (penalty) |

**Buy condition**: `score > 0.5`
**Sell condition**: `score < -0.5`

Signals are converted to event-based to prevent over-trading.

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `ma_short` | 5 | Short-term MA window |
| `ma_long` | 20 | Long-term MA window |
| `rsi_period` | 14 | RSI calculation period |
| `volume_threshold` | 1.5 | Volume ratio threshold |

**Code reference**:
```python
# src/strategies/multi_factor.py:116-131 — Score calculation
df["score"] = 0.0
df.loc[df["ma_short"] > df["ma_long"], "score"] += 0.3      # MA trend
df.loc[df["ma_short"] < df["ma_long"], "score"] -= 0.3
df["score"] += df["rsi_norm"] * 0.3                          # RSI
df.loc[df["volume_ratio"] > 1.5, "score"] += 0.2            # Volume
df.loc[df["volume_ratio"] < 0.5, "score"] -= 0.2
df.loc[df["volatility"] > vol_threshold, "score"] -= 0.2     # Volatility penalty

# src/strategies/multi_factor.py:164-166 — Signal from score
df.loc[df["score"] > 0.5, "signal"] = 1
df.loc[df["score"] < -0.5, "signal"] = -1
df = convert_to_event_signals(df)
```

---

### 12. Grid (`grid`)

**File**: `src/strategies/grid.py`

**How it works**: Places evenly-spaced grid lines between a lower and upper price. When price drops through a grid line, it buys a fixed amount. When price rises through a grid line, it sells a fixed amount. This strategy profits from sideways oscillation between grid levels.

**Buy condition**: Price crosses below a grid line (from above).
**Sell condition**: Price crosses above a grid line (from below).

The grid lines are calculated as:
```
grid_prices = linspace(lower_price, upper_price, grid_num)
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `lower_price` | required | Grid lower bound |
| `upper_price` | required | Grid upper bound |
| `grid_num` | 10 | Number of grid levels |
| `amount_per_grid` | 0.01 | Trade amount per grid level (in coin units) |

**Code reference**:
```python
# src/strategies/grid.py:52 — Grid line calculation
self.grid_prices = np.linspace(lower_price, upper_price, grid_num)

# src/strategies/grid.py:84-90 — Per-bar signal generation
for grid_price in self.grid_prices:
    if last_price > grid_price and current_price <= grid_price:
        signals[i] = 1                      # Buy on downward cross
    elif last_price < grid_price and current_price >= grid_price:
        signals[i] = -1                     # Sell on upward cross
```

> **Note**: The `run_backtest.py` runner calculates grid bounds from the first 100 bars (with 10% margin) to avoid look-ahead bias. See `run_backtest.py:231-240`.

---

### 13. Martingale (`martingale`)

**File**: `src/strategies/martingale.py`

**How it works**: Starts with an initial buy. If price drops by a stop-loss percentage, the strategy doubles the position size (updates the average entry price). This continues until the price recovers to the take-profit target, or the maximum number of doubling steps is reached, at which point a forced stop-loss exit occurs.

**WARNING**: This is an extremely high-risk strategy. It is included for backtesting research only. Never use it in live trading.

**Decision flow**:
```
1. No position  -->  Buy base_amount
2. In position:
   a. Price change >= target_profit  -->  Sell all (take profit)
   b. Price change <= -stop_loss/(step+1)  -->  Double down (if step < max_steps)
   c. Step >= max_steps AND stop-loss hit  -->  Force sell (stop loss)
   d. Otherwise  -->  Hold
```

**Parameters**:

| Param | Default | Description |
|-------|---------|-------------|
| `base_amount` | 0.001 | Initial buy amount (coin units) |
| `multiplier` | 2.0 | Position size multiplier per doubling step |
| `max_steps` | 5 | Maximum number of doubling steps |
| `target_profit` | 0.01 | Take-profit target (1%) |
| `stop_loss` | 0.05 | Initial stop-loss trigger (5%) |

**Code reference**:
```python
# src/strategies/martingale.py:86-105 — Core decision logic
price_change = (current_price - entry_price) / entry_price
if price_change >= self.target_profit:
    return -1, 0.0, 0, False           # Take profit
stop_threshold = self.stop_loss / (current_step + 1)
if price_change <= -stop_threshold:
    if current_step < self.max_steps:
        return 1, new_entry_price, current_step + 1, True  # Double down
    else:
        return -1, 0.0, 0, False       # Max steps reached, stop loss
return 0, entry_price, current_step, True                   # Hold
```

---

## Strategy Helpers

The following helper functions in `src/strategies/_helpers.py` are shared across strategies:

### `detect_crossover(df, fast_col, slow_col)`

Detects when `fast_col` crosses above or below `slow_col`. Sets `signal=1` on Golden Cross (fast crosses above slow) and `signal=-1` on Death Cross (fast crosses below slow). Only fires on the exact crossover bar.

```python
# Used by: ma_cross, macd
df = detect_crossover(df, "ma_short", "ma_long")
```

### `convert_to_event_signals(df)`

Converts state-based signals (where `signal=1` appears on every bar a condition is true) into event-based signals (only the first bar of each state change). Consecutive identical non-zero signals are collapsed to a single event.

```python
# Used by: mean_reversion, vwap, multi_factor
# Before: [0, 0, 1, 1, 1, -1, -1, 0, 1, 1]
# After:  [0, 0, 1, 0, 0, -1, 0, 0, 1, 0]
df = convert_to_event_signals(df)
```

### `add_trend_filter(df, trend_window, trend_tolerance)`

Adds a `trend_filter` boolean column. `True` where price is within `trend_tolerance` (default 3%) of the trend MA, indicating a ranging market. Strategies can zero out signals where `trend_filter` is `False` to avoid mean-reversion trades during strong trends.

```python
# Used by: rsi, bollinger, stochastic, mean_reversion (when trend_filter_enabled=True)
df = add_trend_filter(df, trend_window=50, trend_tolerance=0.03)
df.loc[~df["trend_filter"], "signal"] = 0
```

### `forward_fill_position(df)`

Creates a `position` column by forward-filling non-zero signals. After a buy signal (1), position stays at 1 until a sell signal (-1) appears. This is mainly for visualization; the backtest engine manages its own position state internally.

```python
# Used by: most strategies
df = forward_fill_position(df)
```

### `calculate_rsi(prices, period)`

Calculates the RSI indicator. Returns a series in the range [0, 100].

```python
# Used by: rsi, multi_factor
df["rsi"] = calculate_rsi(df["close"], period=14)
```

---

## Running a Backtest

### CLI Usage

The `run_backtest.py` script is the main entry point.

```bash
# Default: BTC with multi_factor strategy, 2 years, 1h interval
python run_backtest.py

# Single strategy backtest
python run_backtest.py --coin eth --strategy ma_cross

# Custom parameters
python run_backtest.py --coin btc --strategy rsi --interval 4h --days 365 --capital 50000

# Compare all 13 strategies
python run_backtest.py --coin btc --compare

# Skip chart generation
python run_backtest.py --coin btc --strategy bollinger --no-viz

# Use OKX data source (for restricted regions)
python run_backtest.py --source okx --coin btc --strategy ma_cross --no-viz
```

**CLI arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--coin` | `btc` | Coin symbol: btc, eth, sol, or all |
| `--strategy` | `multi_factor` | Strategy name (see factory below) |
| `--days` | `730` | Number of days to backtest |
| `--interval` | `1h` | K-line interval: 1m, 5m, 15m, 1h, 4h, 1d |
| `--capital` | `10000` | Initial capital in USD |
| `--compare` | off | Compare all strategies |
| `--no-viz` | off | Skip visualization charts |
| `--disable-drawdown-breaker` | off | Disable maximum-drawdown forced liquidation and trading halt |
| `--source` | `binance` | Data source: binance or okx |

### Python API Usage

```python
import pandas as pd
from backtest import BacktestEngine
from strategies import get_strategy

# 1. Prepare data (OHLCV DataFrame with timestamp index)
df = pd.read_csv("data/historical/btc_1h_730d.csv",
                  index_col="timestamp", parse_dates=True)

# 2. Create strategy
strategy = get_strategy("ma_cross", short_window=10, long_window=30)

# 3. Create engine and run backtest
engine = BacktestEngine(
    initial_capital=10000.0,
    commission_rate=0.001,   # 0.1% commission
    slippage=0.001,          # 0.1% slippage
    position_size=0.95,      # Use 95% of cash per trade
)
result = engine.run_backtest(df, strategy, coin="BTC")

# 4. Read results
print(f"Total return: {result.metrics['total_return_pct']:.2f}%")
print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max drawdown: {result.metrics['max_drawdown_pct']:.2f}%")
print(f"Win rate: {result.metrics['win_rate_pct']:.2f}%")
print(f"Total trades: {result.metrics['total_trades']}")
print(f"Cost drag: {result.metrics['cost_drag_pct']:.2f}%")
```

### Strategy Factory

All 13 strategies are available through the factory function:

```python
from strategies import get_strategy

# By name (default parameters)
strategy = get_strategy("rsi")

# With custom parameters
strategy = get_strategy("bollinger", window=20, num_std=2.0, trend_filter_enabled=True)

# Available names:
# ma_cross, rsi, bollinger, multi_factor, mean_reversion,
# macd, breakout, vwap, momentum, atr_stop, stochastic,
# grid, martingale
```

The factory passes all `**kwargs` to the strategy constructor, so any constructor parameter can be specified by name.

### Backtest Engine Internals

The `BacktestEngine` in `src/backtest.py` processes each bar in sequence. Here is the execution flow:

```
For each bar i in the data:
  1. DAY TRACKING: Reset daily trade count on new day
  2. DRAWDOWN CHECK: If drawdown_breaker_enabled and drawdown >= max_drawdown_pct,
     force-liquidate and stop trading
  3. STOP-LOSS CHECK: If position loss >= stop_loss_pct, force sell
  4. SIGNAL CHECK:
     - If signal==1 and no position and trades_today < max_trades_per_day --> BUY
     - If signal==-1 and in position and bars_since_entry >= min_holding_bars --> SELL
  5. EQUITY TRACKING: Record total_value = cash + position * price
```

**Code reference** (`src/backtest.py:389-461`):
```python
for i in range(len(df)):
    # ... risk management checks ...

    # Min holding period check
    can_sell = self._entry_bar < 0 or (i - self._entry_bar >= self.min_holding_bars)

    if signal == SIGNAL_BUY and self.position == 0 and self._trades_today < self.max_trades_per_day:
        self._execute_buy(timestamp, price, coin, result, df.iloc[i])
        self._entry_bar = i
        self._trades_today += 1

    elif signal == SIGNAL_SELL and self.position > 0 and can_sell:
        self._execute_sell(timestamp, price, coin, result, df.iloc[i])
        self._trades_today += 1
```

**Buy execution** (`src/backtest.py:496-538`):
```python
executed_price = price * (1 + self.slippage)          # Slippage increases buy price
position_value = self.cash * self.position_size         # 95% of cash
commission = position_value * self.commission_rate      # 0.1% commission
quantity = (position_value - commission) / executed_price
self.cash -= position_value
self.position = quantity
```

**Sell execution** (`src/backtest.py:540-593`):
```python
executed_price = price * (1 - self.slippage)           # Slippage decreases sell price
sell_value = self.position * executed_price
commission = sell_value * self.commission_rate
net_value = sell_value - commission
self.cash += net_value
self.position = 0.0
```

### Risk Management

The backtest engine includes four risk management controls, all configurable via constructor parameters:

| Control | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| Min holding period | `min_holding_bars` | 5 | Minimum bars to hold before selling |
| Max trades per day | `max_trades_per_day` | 6 | Hard cap on daily trade count |
| Per-trade stop-loss | `stop_loss_pct` | 0.05 (5%) | Force-sell if unrealized loss exceeds threshold |
| Drawdown circuit breaker | `drawdown_breaker_enabled`, `max_drawdown_pct` | enabled; 0.20 (20%) | Force-liquidate and halt trading if total drawdown exceeds threshold |

```python
# Strict risk management example
engine = BacktestEngine(
    initial_capital=10000,
    min_holding_bars=10,        # Hold at least 10 bars
    max_trades_per_day=3,       # Max 3 round-trips per day
    stop_loss_pct=0.03,         # 3% stop-loss per trade
    max_drawdown_pct=0.15,      # Stop everything at 15% drawdown
)
```

The drawdown circuit breaker is enabled by default. Disable it to continue trading through
drawdowns while retaining the other risk controls:

```python
engine = BacktestEngine(
    initial_capital=10000,
    drawdown_breaker_enabled=False,
)
```

```bash
python run_backtest.py --strategy momentum --disable-drawdown-breaker
python run_backtest.py --compare --disable-drawdown-breaker
```

Disabling this breaker does not disable stop-losses and does not remove the
`max_drawdown_pct` performance metric.

### Performance Metrics

After a backtest, `result.metrics` contains:

| Metric | Key | Description |
|--------|-----|-------------|
| Total Return | `total_return_pct` | `(final_equity / initial_capital - 1) * 100` |
| Annual Return | `annual_return_pct` | Annualized total return |
| Volatility | `volatility_pct` | Annualized standard deviation of daily returns |
| Sharpe Ratio | `sharpe_ratio` | `(annual_return - risk_free_rate) / volatility` |
| Max Drawdown | `max_drawdown_pct` | Largest peak-to-trough decline |
| Win Rate | `win_rate_pct` | Percentage of sell trades that are profitable |
| Total Trades | `total_trades` | Total number of buy/sell actions |
| Trades/Month | `trades_per_month` | Average trades per calendar month |
| Total Cost | `total_cost` | Sum of all commissions and slippage |
| Cost Drag | `cost_drag_pct` | Total cost as percentage of initial capital |

**Code reference** (`src/backtest.py:159-241`):
```python
# Win rate calculation
for trade in self.trades:
    if trade.action == "buy":
        last_buy_price = trade.price
    elif trade.action == "sell":
        total_sells += 1
        if last_buy_price is not None and trade.price > last_buy_price:
            profitable_sells += 1

# Sharpe ratio
sharpe_ratio = (annual_return / 100 - RISK_FREE_RATE) / (volatility / 100)

# Cost analysis (added after calculate_metrics)
total_cost = sum(trade.value * self.commission_rate + trade.value * self.slippage
                 for trade in result.trades)
```

---

## Choosing a Strategy

| Market Condition | Recommended Strategies | Avoid |
|-----------------|----------------------|-------|
| **Strong uptrend** | ma_cross, macd, momentum, atr_stop | mean_reversion, stochastic |
| **Strong downtrend** | (short-side: macd, breakout) | mean_reversion, martingale |
| **Sideways / range-bound** | rsi, bollinger, grid, vwap | momentum, breakout |
| **Volatile / choppy** | atr_stop (low multiplier) | martingale, grid |
| **Low volatility** | bollinger (tight bands), ma_cross | breakout, momentum |

**Tips**:
- Use `1h` or `4h` intervals for reliable results. Sub-hourly intervals (1m, 5m, 15m) generate excessive trades and costs.
- Enable `trend_filter_enabled=True` on mean-reversion strategies to avoid trading against trends.
- Always review the `cost_drag_pct` metric — if it exceeds 10%, the strategy is over-trading for the given interval.
- The `multi_factor` strategy combines multiple signals and tends to be more robust than single-indicator strategies.
