"""
Dual Moving Average Crossover Strategy

Buy when the short-term moving average crosses above the long-term moving average (Golden Cross),
sell when it crosses below (Death Cross).
A trend-following strategy suitable for medium-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('ma_cross', short_window=5, long_window=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, detect_crossover
from strategies.constants import DEFAULT_MA_SHORT, DEFAULT_MA_LONG


class MovingAverageCrossStrategy(TradingStrategy):
    """
    Dual Moving Average Crossover Strategy (Medium Frequency)

    Buy when the short-term moving average crosses above the long-term moving average,
    sell when it crosses below.
    Suitable for markets with clear trends; prone to false signals in sideways markets.

    Args:
        short_window: Short-term moving average window (default 10)
        long_window: Long-term moving average window (default 30)

    Generated indicator columns:
        ma_short: Short-term moving average value
        ma_long: Long-term moving average value
        ma_diff: Difference between short-term and long-term moving averages
    """

    def __init__(self, short_window: int = DEFAULT_MA_SHORT, long_window: int = DEFAULT_MA_LONG):
        super().__init__("MA_Cross")
        self.short_window = short_window
        self.long_window = long_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average indicators

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with ma_short, ma_long, ma_diff columns added
        """
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["ma_long"] = df["close"].rolling(window=self.long_window).mean()
        df["ma_diff"] = df["ma_short"] - df["ma_long"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Golden Cross (short MA crosses above long MA) generates a buy signal,
        Death Cross (short MA crosses below long MA) generates a sell signal.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with signal and position columns added
        """
        df = self.calculate_indicators(df)
        df = detect_crossover(df, "ma_short", "ma_long")
        df = forward_fill_position(df)
        return df
