"""
Mean Reversion Strategy

Take contrarian positions when price deviates significantly from the mean (Z-score exceeds threshold),
close position when price reverts to the mean.
Uses event-based signals to avoid over-trading, with an optional trend filter to suppress
signals during strong trending markets.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('mean_reversion', window=20, entry_z=2.0, exit_z=0.5)
    >>> result_df = strategy.generate_signals(df)

    With trend filter enabled:
    >>> strategy = get_strategy('mean_reversion', trend_filter_enabled=True)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import convert_to_event_signals, forward_fill_position, apply_trend_filter
from strategies.constants import (
    DEFAULT_MEAN_REVERSION_WINDOW,
    DEFAULT_ENTRY_Z,
    DEFAULT_EXIT_Z,
    TREND_FILTER_WINDOW,
    TREND_FILTER_TOLERANCE,
)


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy (Event-Based with Optional Trend Filter)

    Mean reversion strategy based on price Z-score:
    - Buy when Z-score falls below -entry_z (price is significantly below mean)
    - Sell when Z-score rises above entry_z (price is significantly above mean)
    - Close position when Z-score reverts within exit_z

    Signals are converted to event-based (only the first bar of each state change)
    to prevent over-trading from consecutive identical signals.

    An optional trend filter can suppress signals when price is far from its
    moving average, avoiding mean-reversion entries during strong trends.

    Args:
        window: Rolling mean and standard deviation calculation window (default 20)
        entry_z: Entry Z-score threshold (default 2.0)
        exit_z: Exit Z-score threshold (default 0.5)
        trend_filter_enabled: Whether to enable the trend filter (default True)
        trend_filter_window: Window for trend filter MA calculation (default 50)
        trend_filter_tolerance: Max deviation from MA for ranging market (default 0.03 = 3%)

    Generated indicator columns:
        mean: Rolling mean
        std: Rolling standard deviation
        zscore: Z-score value
        trend_filter: (only when enabled) True where market is ranging
    """

    def __init__(
        self,
        window: int = DEFAULT_MEAN_REVERSION_WINDOW,
        entry_z: float = DEFAULT_ENTRY_Z,
        exit_z: float = DEFAULT_EXIT_Z,
        trend_filter_enabled: bool = True,
        trend_filter_window: int = TREND_FILTER_WINDOW,
        trend_filter_tolerance: float = TREND_FILTER_TOLERANCE,
    ):
        super().__init__("Mean_Reversion")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_filter_window = trend_filter_window
        self.trend_filter_tolerance = trend_filter_tolerance

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when Z-score falls below -entry_z, sell when above entry_z,
        close position when absolute Z-score is less than exit_z.

        State-based signals are converted to event-based (only the first bar
        of each state change) to avoid over-trading. If trend_filter_enabled,
        signals are suppressed when the market is in a strong trend.

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with zscore, signal, and position columns added
        """
        df = df.copy()

        df["mean"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        # Division-by-zero guard: when std is 0 (flat market), zscore is 0
        df["zscore"] = (df["close"] - df["mean"]) / df["std"].replace(0, float("nan"))
        df["zscore"] = df["zscore"].fillna(0)

        df["signal"] = 0
        df.loc[df["zscore"] < -self.entry_z, "signal"] = 1
        df.loc[df["zscore"] > self.entry_z, "signal"] = -1
        # exit_z overrides entry signals when reverting to mean
        df.loc[abs(df["zscore"]) < self.exit_z, "signal"] = 0

        df = convert_to_event_signals(df)
        df = apply_trend_filter(
            df,
            self.trend_filter_enabled,
            self.trend_filter_window,
            self.trend_filter_tolerance,
        )
        df = forward_fill_position(df)

        return df
