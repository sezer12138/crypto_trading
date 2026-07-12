"""
Stochastic Oscillator Strategy

Buy when the K line crosses above the D line in the oversold zone (K<20),
sell when the K line crosses below the D line in the overbought zone (K>80).
A mean reversion strategy suitable for medium-to-high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('stochastic', k_period=14, d_period=3)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, apply_trend_filter
from strategies.constants import (
    STOCHASTIC_OVERSOLD,
    STOCHASTIC_OVERBOUGHT,
    TREND_FILTER_WINDOW,
    TREND_FILTER_TOLERANCE,
)


class StochasticStrategy(TradingStrategy):
    """
    Stochastic Oscillator Strategy (Medium-High Frequency)

    The Stochastic Oscillator measures the relative position of the current price within a given period's range:
    - K line: Percentage position of current price within the N-period high-low price range
    - D line: Moving average of the K line

    Strategy logic:
    - Buy when K crosses above D and K < 20 (oversold zone)
    - Sell when K crosses below D and K > 80 (overbought zone)

    Args:
        k_period: K line calculation period (default 14)
        d_period: D line smoothing period (default 3)
        smooth: K line pre-smoothing period (default 3)
        trend_filter_enabled: Enable trend filter to suppress signals in strong trends (default True)
        trend_filter_window: Window for trend MA calculation (default 50)
        trend_filter_tolerance: Max deviation from MA for ranging market (default 0.03)

    Generated indicator columns:
        k: K line value (range 0-100)
        d: D line value
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth: int = 3,
        trend_filter_enabled: bool = True,
        trend_filter_window: int = TREND_FILTER_WINDOW,
        trend_filter_tolerance: float = TREND_FILTER_TOLERANCE,
    ):
        super().__init__("Stochastic_Strategy")
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_filter_window = trend_filter_window
        self.trend_filter_tolerance = trend_filter_tolerance

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator indicators

        Args:
            df: DataFrame containing high, low, close columns

        Returns:
            DataFrame with k, d columns added
        """
        df = df.copy()
        lowest_low = df["low"].rolling(window=self.k_period).min()
        highest_high = df["high"].rolling(window=self.k_period).max()
        df["k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        df["d"] = df["k"].rolling(window=self.d_period).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when K crosses above D and K is in the oversold zone (<20),
        sell when K crosses below D and K is in the overbought zone (>80).

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with k, d, signal, position columns added
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # Buy when K crosses above D in oversold zone
        df.loc[
            (df["k"] > df["d"])
            & (df["k"].shift(1) <= df["d"].shift(1))
            & (df["k"] < STOCHASTIC_OVERSOLD),
            "signal",
        ] = 1

        # Sell when K crosses below D in overbought zone
        df.loc[
            (df["k"] < df["d"])
            & (df["k"].shift(1) >= df["d"].shift(1))
            & (df["k"] > STOCHASTIC_OVERBOUGHT),
            "signal",
        ] = -1

        df = apply_trend_filter(
            df,
            self.trend_filter_enabled,
            self.trend_filter_window,
            self.trend_filter_tolerance,
        )
        df = forward_fill_position(df)

        return df
