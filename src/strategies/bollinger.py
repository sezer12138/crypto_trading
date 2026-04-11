"""
Bollinger Bands Strategy

Buy when price bounces off the lower band, sell when price retreats from the upper band.
A mean reversion strategy suitable for medium-to-high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('bollinger', window=20, num_std=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position
from strategies.constants import DEFAULT_BB_WINDOW, DEFAULT_BB_NUM_STD


class BollingerBandsStrategy(TradingStrategy):
    """
    Bollinger Bands Strategy (Medium-High Frequency)

    Bollinger Bands consist of a middle band (moving average) and upper/lower bands (±N standard deviations).
    Buy when price bounces off the lower band, sell when price retreats from the upper band.

    Args:
        window: Moving average window (default 20)
        num_std: Standard deviation multiplier (default 2.0)

    Generated indicator columns:
        middle_band: Middle band (moving average)
        std: Rolling standard deviation
        upper_band: Upper band
        lower_band: Lower band
        bandwidth: Bandwidth (upper-lower band difference / middle band)
    """

    def __init__(self, window: int = DEFAULT_BB_WINDOW, num_std: float = DEFAULT_BB_NUM_STD):
        super().__init__("Bollinger_Bands")
        self.window = window
        self.num_std = num_std

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands indicators

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with Bollinger Bands related columns added
        """
        df = df.copy()
        df["middle_band"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        df["upper_band"] = df["middle_band"] + (df["std"] * self.num_std)
        df["lower_band"] = df["middle_band"] - (df["std"] * self.num_std)
        df["bandwidth"] = (df["upper_band"] - df["lower_band"]) / df["middle_band"].replace(0, float("nan"))
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when price breaks below the lower band and recovers,
        sell when price breaks above the upper band and retreats.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with signal and position columns added
        """
        df = self.calculate_indicators(df)

        df["signal"] = 0

        # Buy when price drops below lower band and recovers
        df.loc[
            (df["close"] > df["lower_band"]) & (df["close"].shift(1) <= df["lower_band"].shift(1)),
            "signal",
        ] = 1

        # Sell when price breaks above upper band and retreats
        df.loc[
            (df["close"] < df["upper_band"]) & (df["close"].shift(1) >= df["upper_band"].shift(1)),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
