"""
Momentum Strategy

Trading strategy based on Rate of Change (ROC) and momentum indicators.
Buy when ROC turns positive and momentum is positive, sell when ROC turns negative and momentum is negative.
A trend-following strategy suitable for medium-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('momentum', roc_period=10, threshold=0.02)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class MomentumStrategy(TradingStrategy):
    """
    Momentum Strategy (Medium Frequency)

    Combines Rate of Change (ROC) and momentum indicators:
    - Buy when ROC turns positive and momentum is positive
    - Sell when ROC turns negative and momentum is negative

    Args:
        roc_period: Rate of change calculation period (default 10)
        momentum_period: Momentum calculation period (default 14)
        threshold: ROC threshold (default 0.02 = 2%)

    Generated indicator columns:
        roc: Rate of change (percentage)
        momentum: Momentum value (price difference)
        momentum_norm: Normalized momentum (percentage)
    """

    def __init__(self, roc_period: int = 10, momentum_period: int = 14, threshold: float = 0.02):
        super().__init__("Momentum_Strategy")
        self.roc_period = roc_period
        self.momentum_period = momentum_period
        self.threshold = threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with roc, momentum, momentum_norm columns added
        """
        df = df.copy()
        # Rate of change
        shifted = df["close"].shift(self.roc_period)
        df["roc"] = (df["close"] - shifted) / shifted.replace(0, float("nan"))
        # Momentum indicator
        df["momentum"] = df["close"] - df["close"].shift(self.momentum_period)
        df["momentum_norm"] = df["momentum"] / df["close"] * 100
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when ROC turns positive and momentum is positive, sell when ROC turns negative and momentum is negative.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with signal and position columns added
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # Buy when ROC turns positive and momentum is positive
        df.loc[
            (df["roc"] > self.threshold) & (df["momentum_norm"] > 0) &
            (df["roc"].shift(1) <= self.threshold),
            "signal",
        ] = 1

        # Sell when ROC turns negative and momentum is negative
        df.loc[
            (df["roc"] < -self.threshold) & (df["momentum_norm"] < 0) &
            (df["roc"].shift(1) >= -self.threshold),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
