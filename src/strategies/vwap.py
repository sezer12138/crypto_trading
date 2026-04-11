"""
VWAP Mean Reversion Strategy

Buy when price falls below VWAP by a certain margin, sell when price rises above VWAP by a certain margin.
A mean reversion strategy suitable for high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('vwap', window=20, deviation=0.01)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class VWAPStrategy(TradingStrategy):
    """
    VWAP Mean Reversion Strategy (High Frequency)

    VWAP (Volume Weighted Average Price) is the volume-weighted average price.
    When price deviates from VWAP beyond the set margin, it is expected to revert to VWAP.

    Args:
        window: VWAP calculation window (default 20)
        deviation: Deviation threshold (default 0.01 = 1%)

    Generated indicator columns:
        vwap: Volume-weighted average price
        vwap_dev: Percentage deviation of price from VWAP
    """

    def __init__(self, window: int = 20, deviation: float = 0.01):
        super().__init__("VWAP_Strategy")
        self.window = window
        self.deviation = deviation

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP indicator

        Args:
            df: DataFrame containing high, low, close, volume columns

        Returns:
            DataFrame with vwap, vwap_dev columns added
        """
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=self.window).sum() / \
                     df["volume"].rolling(window=self.window).sum()
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, float("nan"))
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when price falls below VWAP beyond the deviation threshold,
        sell when price rises above VWAP beyond the deviation threshold.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with vwap, vwap_dev, signal, position columns added
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # Buy when price falls below VWAP by a certain margin
        df.loc[df["vwap_dev"] < -self.deviation, "signal"] = 1
        # Sell when price rises above VWAP by a certain margin
        df.loc[df["vwap_dev"] > self.deviation, "signal"] = -1

        df = forward_fill_position(df)
        return df
