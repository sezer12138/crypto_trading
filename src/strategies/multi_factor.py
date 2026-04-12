"""
Multi-Factor Composite Strategy

Combines multiple factors including moving average trend, RSI, volume, and volatility for composite scoring.
A composite scoring strategy suitable for high-frequency trading.

Scoring System:
    - Moving Average Trend (weight 30%): Direction of short-term vs long-term moving average
    - RSI Factor (weight 30%): Normalized RSI value (range -1 to 1)
    - Volume Confirmation (weight 20%): Volume ratio compared to threshold
    - Volatility Filter (weight 20%): Penalty for high volatility

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('multi_factor', ma_short=5, ma_long=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import calculate_rsi, convert_to_event_signals, forward_fill_position
from strategies.constants import (
    DEFAULT_MA_SHORT,
    DEFAULT_MA_LONG,
    DEFAULT_RSI_PERIOD,
    DEFAULT_VOLUME_THRESHOLD,
    WEIGHT_MA_TREND,
    WEIGHT_RSI,
    WEIGHT_VOLUME,
    WEIGHT_VOLATILITY,
    SCORE_BUY_THRESHOLD,
    SCORE_SELL_THRESHOLD,
    VOLUME_LOW_RATIO,
    VOLATILITY_QUANTILE,
    DEFAULT_VOLUME_MA_WINDOW,
    DEFAULT_VOLATILITY_WINDOW,
)


class MultiFactorStrategy(TradingStrategy):
    """
    Multi-Factor Composite Strategy (High Frequency)

    Combines multiple factors including moving average, RSI, volume, and volatility,
    generating signals through a weighted scoring system.
    Buy when composite score > 0.5, sell when composite score < -0.5.

    Args:
        ma_short: Short-term moving average window (default 5)
        ma_long: Long-term moving average window (default 20)
        rsi_period: RSI calculation period (default 14)
        volume_threshold: Volume ratio threshold (default 1.5)

    Generated indicator columns:
        ma_short, ma_long, ma_trend: Moving averages and trend direction
        rsi, rsi_norm: RSI and its normalized value
        volume_ma, volume_ratio: Volume moving average and ratio
        returns, volatility: Return rate and volatility
        score: Composite score (range approximately -1 to 1)
    """

    def __init__(
        self,
        ma_short: int = DEFAULT_MA_SHORT,
        ma_long: int = DEFAULT_MA_LONG,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
    ):
        super().__init__("Multi_Factor")
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.volume_threshold = volume_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Calculates moving average system, RSI, volume factor, and volatility indicators in sequence.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with indicator columns added
        """
        df = df.copy()

        # Moving average system
        df["ma_short"] = df["close"].rolling(window=self.ma_short).mean()
        df["ma_long"] = df["close"].rolling(window=self.ma_long).mean()
        df["ma_trend"] = (df["ma_short"] > df["ma_long"]).astype(int)

        # RSI (using shared RSI calculation function)
        df["rsi"] = calculate_rsi(df["close"], self.rsi_period)
        df["rsi_norm"] = (df["rsi"] - 50) / 50  # Normalized to [-1, 1]

        # Volume factor
        df["volume_ma"] = df["volume"].rolling(window=DEFAULT_VOLUME_MA_WINDOW).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, float("nan"))

        # Volatility
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=DEFAULT_VOLATILITY_WINDOW).std()

        return df

    def _calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multi-factor composite score

        Scoring system:
            - Moving average trend (weight 30%): Positive when short MA is above long MA, negative when below
            - RSI (weight 30%): Normalized RSI value (range -1 to 1)
            - Volume confirmation (weight 20%): Positive when volume expands, negative when it shrinks
            - Volatility filter (weight 20%): Penalty in high volatility environments

        Args:
            df: DataFrame containing technical indicators

        Returns:
            DataFrame with a 'score' column added, range approximately [-1, 1]
        """
        df["score"] = 0.0

        # Moving average trend (weight 30%)
        df.loc[df["ma_short"] > df["ma_long"], "score"] += WEIGHT_MA_TREND
        df.loc[df["ma_short"] < df["ma_long"], "score"] -= WEIGHT_MA_TREND

        # RSI (weight 30%)
        df["score"] += df["rsi_norm"] * WEIGHT_RSI

        # Volume confirmation (weight 20%)
        df.loc[df["volume_ratio"] > self.volume_threshold, "score"] += WEIGHT_VOLUME
        df.loc[df["volume_ratio"] < VOLUME_LOW_RATIO, "score"] -= WEIGHT_VOLUME

        # Volatility filter (weight 20%)
        vol_threshold = df["volatility"].quantile(VOLATILITY_QUANTILE)
        df.loc[df["volatility"] > vol_threshold, "score"] -= WEIGHT_VOLATILITY

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy signal when composite score > buy threshold (0.5),
        sell signal when composite score < sell threshold (-0.5).

        Uses event-based signal conversion to prevent over-trading: only the first bar
        of each signal state change emits a signal, consecutive bars with the same
        state are suppressed.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with score, signal, position columns added
        """
        df = self.calculate_indicators(df)
        df = self._calculate_score(df)

        # Generate signals from score thresholds
        df["signal"] = 0
        df.loc[df["score"] > SCORE_BUY_THRESHOLD, "signal"] = 1
        df.loc[df["score"] < SCORE_SELL_THRESHOLD, "signal"] = -1

        # Convert state-based signals to event-based (keep only first bar of each state change)
        df = convert_to_event_signals(df)

        # Forward fill position state
        df = forward_fill_position(df)

        return df
