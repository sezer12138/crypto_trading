"""
Strategy Parameter Constants

Centrally defines default parameter values for all strategies for unified management and tuning.
Each constant name clearly expresses its meaning, replacing magic numbers in the code.
"""

# ============================================================
# Moving Average strategy defaults
# ============================================================
DEFAULT_MA_SHORT = 10  # Short-term moving average window
DEFAULT_MA_LONG = 30  # Long-term moving average window

# ============================================================
# RSI indicator defaults
# ============================================================
DEFAULT_RSI_PERIOD = 14  # RSI calculation period
DEFAULT_RSI_OVERSOLD = 30  # RSI oversold threshold
DEFAULT_RSI_OVERBOUGHT = 70  # RSI overbought threshold

# ============================================================
# Bollinger Bands defaults
# ============================================================
DEFAULT_BB_WINDOW = 20  # Bollinger Bands moving window
DEFAULT_BB_NUM_STD = 2.0  # Bollinger Bands standard deviation multiplier

# ============================================================
# MACD default periods
# ============================================================
DEFAULT_MACD_FAST = 12  # MACD fast line period
DEFAULT_MACD_SLOW = 26  # MACD slow line period
DEFAULT_MACD_SIGNAL = 9  # MACD signal line period

# ============================================================
# Momentum strategy defaults
# ============================================================
DEFAULT_MOMENTUM_BUY_ROC_PERIOD = 16  # Buy rate-of-change lookback period
DEFAULT_MOMENTUM_BUY_PERIOD = 12  # Buy price momentum lookback period
DEFAULT_MOMENTUM_BUY_THRESHOLD = 0.055  # Buy ROC signal threshold (5.5%)
DEFAULT_MOMENTUM_SELL_ROC_PERIOD = 16  # Sell rate-of-change lookback period
DEFAULT_MOMENTUM_SELL_PERIOD = 12  # Sell price momentum lookback period
DEFAULT_MOMENTUM_SELL_THRESHOLD = 0.055  # Sell ROC signal threshold (5.5%)

# ============================================================
# Multi-factor strategy weights
# ============================================================
WEIGHT_MA_TREND = 0.3  # Moving average trend weight
WEIGHT_RSI = 0.3  # RSI factor weight
WEIGHT_VOLUME = 0.2  # Volume factor weight
WEIGHT_VOLATILITY = 0.2  # Volatility factor weight

# ============================================================
# Multi-factor strategy thresholds
# ============================================================
SCORE_BUY_THRESHOLD = 0.5  # Composite score buy threshold
SCORE_SELL_THRESHOLD = -0.5  # Composite score sell threshold
DEFAULT_VOLUME_THRESHOLD = 1.5  # Volume ratio threshold
VOLUME_LOW_RATIO = 0.5  # Low volume ratio threshold
VOLATILITY_QUANTILE = 0.7  # High volatility quantile threshold
DEFAULT_VOLUME_MA_WINDOW = 20  # Volume moving average window
DEFAULT_VOLATILITY_WINDOW = 20  # Volatility calculation window

# ============================================================
# Mean Reversion strategy thresholds
# ============================================================
DEFAULT_MEAN_REVERSION_WINDOW = 20  # Mean reversion calculation window
DEFAULT_ENTRY_Z = 2.0  # Z-score entry threshold
DEFAULT_EXIT_Z = 0.5  # Z-score exit threshold

# ============================================================
# Stochastic Oscillator thresholds
# ============================================================
STOCHASTIC_OVERSOLD = 20  # Stochastic oversold threshold
STOCHASTIC_OVERBOUGHT = 80  # Stochastic overbought threshold

# ============================================================
# Trend filter parameters (for mean-reversion strategies)
# ============================================================
TREND_FILTER_WINDOW = 50  # Window for trend MA calculation
TREND_FILTER_TOLERANCE = 0.03  # Max deviation from MA for ranging market (3%)

# ============================================================
# Backtest engine risk management
# ============================================================
DEFAULT_MIN_HOLDING_BARS = 5  # Minimum holding period (bars) after entry
DEFAULT_MAX_TRADES_PER_DAY = 6  # Maximum number of trades per day
DEFAULT_STOP_LOSS_PCT = 0.05  # Per-trade stop-loss (5%)
DEFAULT_MAX_DRAWDOWN_PCT = 0.20  # Max drawdown circuit breaker (20%)
DEFAULT_ATR_STOP_LOSS_MULTIPLIER = 2.0  # ATR multiplier for dynamic stop-loss
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3  # Max consecutive stop-loss hits before cooldown
DEFAULT_CONSECUTIVE_LOSS_COOLDOWN = 24  # Bars to pause after consecutive losses (24h for 1h data)
DEFAULT_BREAKER_COOLDOWN_BARS = 720  # Bars before breaker resets (30 days for 1h data)

# ============================================================
# VWAP dynamic deviation parameters
# ============================================================
VWAP_DYNAMIC_MULTIPLIER = 1.5  # Multiplier for ATR-based dynamic deviation
VWAP_MIN_DEVIATION = 0.005  # Minimum deviation threshold (0.5%)
VWAP_ATR_WINDOW = 20  # ATR window for dynamic deviation calculation

# BTC/1h profiles accepted by the 2026-10-07 40-trial chronological search.
# Validation used next_open_intrabar_v2, 0.1% commission and slippage, 95% sizing,
# 5-bar minimum holding, 6 daily trades, 5% stops, and disabled cooldown/breaker.
BTC_1H_RSI_PARAMETERS: dict[str, int | float | bool] = {
    "trend_filter_enabled": True,
    "trend_filter_window": 672,
    "trend_filter_tolerance": 0.006128353716190624,
    "period": 12,
    "oversold": 30,
    "overbought": 85,
}
BTC_1H_VWAP_PARAMETERS: dict[str, int | float | bool] = {
    "trend_filter_enabled": True,
    "trend_filter_window": 48,
    "trend_filter_tolerance": 0.0077084678522454965,
    "window": 672,
    "atr_window": 168,
    "deviation_multiplier": 4.0,
    "min_deviation": 0.006769730483419706,
}
