"""
ATR Dynamic Stop-Loss Strategy

Dynamic stop-loss and take-profit strategy based on ATR (Average True Range).
Buy on pullbacks to ATR support levels during uptrends, sell on bounces to ATR resistance levels during downtrends.
A trend-following strategy suitable for high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('atr_stop', atr_period=14, multiplier=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import numpy as np
import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class ATRStopLossStrategy(TradingStrategy):
    """
    ATR Dynamic Stop-Loss Strategy (High Frequency)

    ATR measures market volatility; the strategy sets support/resistance levels based on ATR multiples:
    - In uptrends, buy when price pulls back to ATR support level
    - In downtrends, sell when price bounces to ATR resistance level

    Args:
        atr_period: ATR calculation period (default 14)
        multiplier: ATR multiplier (default 2.0)
        trend_ma: Trend determination moving average period (default 50)

    Generated indicator columns:
        atr: Average True Range
        trend_ma: Trend determination moving average
    """

    def __init__(self, atr_period: int = 14, multiplier: float = 2.0, trend_ma: int = 50):
        super().__init__("ATR_StopLoss_Strategy")
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.trend_ma = trend_ma

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        ATR is a measure of market volatility, taking the moving average of the maximum of:
        1. Current day high - current day low
        2. |Current day high - previous day close|
        3. |Current day low - previous day close|

        Args:
            df: DataFrame containing high, low, close columns

        Returns:
            ATR value series
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        return tr.rolling(window=self.atr_period).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy on pullback to ATR support level during uptrend,
        sell on bounce to ATR resistance level during downtrend.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with atr, trend_ma, signal, position columns added
        """
        df = df.copy()
        df["atr"] = self.calculate_atr(df)
        df["trend_ma"] = df["close"].rolling(window=self.trend_ma).mean()
        df["signal"] = 0

        # Trend determination
        uptrend = df["close"] > df["trend_ma"]
        downtrend = df["close"] < df["trend_ma"]

        # In uptrend, buy on pullback to ATR support level
        support = df["close"] - df["atr"] * self.multiplier
        resistance = df["close"] + df["atr"] * self.multiplier

        df.loc[
            uptrend & (df["low"] < support.shift(1)) & (df["close"] > support.shift(1)), "signal"
        ] = 1
        df.loc[
            downtrend & (df["high"] > resistance.shift(1)) & (df["close"] < resistance.shift(1)),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
