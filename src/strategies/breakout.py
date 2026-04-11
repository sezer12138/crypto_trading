"""
Breakout Strategy

Buy when price breaks above the N-period high, sell when price breaks below the N-period low.
A trend-following strategy suitable for high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('breakout', window=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class BreakoutStrategy(TradingStrategy):
    """
    Breakout Strategy (High Frequency)

    Buy when price breaks above the N-period highest price, sell when price breaks below the N-period lowest price.
    Supports confirmation mode (close price confirmation) and instant mode.

    Args:
        window: Lookback window (default 20)
        confirmation: Whether close price confirmation is required for breakout (default True)

    Generated indicator columns:
        high_n: N-period highest price
        low_n: N-period lowest price
    """

    def __init__(self, window: int = 20, confirmation: bool = True):
        super().__init__("Breakout_Strategy")
        self.window = window
        self.confirmation = confirmation

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Confirmation mode: Signals generated when close price breaks the previous day's N-period high/low.
        Instant mode: Signals generated immediately when intraday price breaks through.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with high_n, low_n, signal, position columns added
        """
        df = df.copy()
        df["high_n"] = df["high"].rolling(window=self.window).max()
        df["low_n"] = df["low"].rolling(window=self.window).min()
        df["signal"] = 0

        if self.confirmation:
            # Close price confirmation required for breakout
            df.loc[
                (df["close"] > df["high_n"].shift(1)) &
                (df["close"].shift(1) <= df["high_n"].shift(2)),
                "signal",
            ] = 1
            df.loc[
                (df["close"] < df["low_n"].shift(1)) &
                (df["close"].shift(1) >= df["low_n"].shift(2)),
                "signal",
            ] = -1
        else:
            # Instant breakout
            df.loc[df["high"] > df["high_n"].shift(1), "signal"] = 1
            df.loc[df["low"] < df["low_n"].shift(1), "signal"] = -1

        df = forward_fill_position(df)
        return df
