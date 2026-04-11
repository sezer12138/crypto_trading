"""
Mean Reversion Strategy

Take contrarian positions when price deviates significantly from the mean (Z-score exceeds threshold),
close position when price reverts to the mean.
A mean reversion strategy suitable for high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('mean_reversion', window=20, entry_z=2.0, exit_z=0.5)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies.constants import DEFAULT_MEAN_REVERSION_WINDOW, DEFAULT_ENTRY_Z, DEFAULT_EXIT_Z


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy (High Frequency)

    Mean reversion strategy based on price Z-score:
    - Buy when Z-score falls below -entry_z (price is significantly below mean)
    - Sell when Z-score rises above entry_z (price is significantly above mean)
    - Close position when Z-score reverts within exit_z

    Args:
        window: Rolling mean and standard deviation calculation window (default 20)
        entry_z: Entry Z-score threshold (default 2.0)
        exit_z: Exit Z-score threshold (default 0.5)

    Generated indicator columns:
        mean: Rolling mean
        std: Rolling standard deviation
        zscore: Z-score value
    """

    def __init__(self, window: int = DEFAULT_MEAN_REVERSION_WINDOW, entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z):
        super().__init__("Mean_Reversion")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when Z-score falls below -entry_z, sell when above entry_z,
        close position (signal=0) when absolute Z-score is less than exit_z.

        Note: This strategy uses 0 instead of -1 for close position signals,
        so forward_fill_position is not used; position is determined dynamically by Z-score.

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with zscore, signal columns added
        """
        df = df.copy()

        # Calculate Z-score
        df["mean"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        # Division-by-zero guard: when std is 0 (flat market), zscore is 0
        df["zscore"] = (df["close"] - df["mean"]) / df["std"].replace(0, float("nan"))
        df["zscore"] = df["zscore"].fillna(0)

        df["signal"] = 0

        # Buy when Z-score is below -entry_z (oversold)
        df.loc[df["zscore"] < -self.entry_z, "signal"] = 1

        # Sell when Z-score is above entry_z (overbought)
        df.loc[df["zscore"] > self.entry_z, "signal"] = -1

        # Close position when reverting to mean
        df.loc[abs(df["zscore"]) < self.exit_z, "signal"] = 0

        return df
