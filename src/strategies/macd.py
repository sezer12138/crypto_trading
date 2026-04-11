"""
MACD Trend Strategy

Buy when the MACD line crosses above the signal line, sell when it crosses below.
A trend-following strategy suitable for medium-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('macd', fast=12, slow=26, signal=9)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, detect_crossover
from strategies.constants import DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, DEFAULT_MACD_SIGNAL


class MACDStrategy(TradingStrategy):
    """
    MACD Trend Strategy (Medium Frequency)

    MACD (Moving Average Convergence Divergence) determines trend direction through the crossover
    of the difference between fast and slow EMAs and its signal line.

    Args:
        fast: Fast line EMA period (default 12)
        slow: Slow line EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Generated indicator columns:
        ema_fast: Fast EMA
        ema_slow: Slow EMA
        macd: MACD line (fast line - slow line)
        macd_signal: Signal line (EMA of MACD)
        macd_hist: MACD histogram (MACD - signal line)
    """

    def __init__(self, fast: int = DEFAULT_MACD_FAST, slow: int = DEFAULT_MACD_SLOW, signal: int = DEFAULT_MACD_SIGNAL):
        super().__init__("MACD_Strategy")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with MACD related columns added
        """
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when MACD line crosses above signal line, sell when it crosses below.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with signal and position columns added
        """
        df = self.calculate_indicators(df)
        df = detect_crossover(df, "macd", "macd_signal")
        df = forward_fill_position(df)
        return df
