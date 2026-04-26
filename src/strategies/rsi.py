"""
RSI Overbought/Oversold Strategy

Buy when RSI recovers from the oversold zone, sell when RSI retreats from the overbought zone.
A mean reversion strategy suitable for medium-to-high-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('rsi', period=14, oversold=30, overbought=70)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, calculate_rsi, apply_trend_filter
from strategies.constants import (
    DEFAULT_RSI_PERIOD,
    DEFAULT_RSI_OVERSOLD,
    DEFAULT_RSI_OVERBOUGHT,
    TREND_FILTER_WINDOW,
    TREND_FILTER_TOLERANCE,
)


class RSIStrategy(TradingStrategy):
    """
    RSI Overbought/Oversold Strategy (Medium-High Frequency)

    RSI below the oversold threshold is considered oversold; buy when RSI rises above the oversold line.
    RSI above the overbought threshold is considered overbought; sell when RSI falls below the overbought line.

    Args:
        period: RSI calculation period (default 14)
        oversold: Oversold threshold (default 30)
        overbought: Overbought threshold (default 70)
        trend_filter_enabled: Enable trend filter to suppress signals in strong trends (default False)
        trend_filter_window: Window for trend MA calculation (default 50)
        trend_filter_tolerance: Max deviation from MA for ranging market (default 0.03)

    Generated indicator columns:
        rsi: RSI value (range 0-100)
    """

    def __init__(
        self,
        period: int = DEFAULT_RSI_PERIOD,
        oversold: int = DEFAULT_RSI_OVERSOLD,
        overbought: int = DEFAULT_RSI_OVERBOUGHT,
        trend_filter_enabled: bool = False,
        trend_filter_window: int = TREND_FILTER_WINDOW,
        trend_filter_tolerance: float = TREND_FILTER_TOLERANCE,
    ):
        super().__init__("RSI_Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_filter_window = trend_filter_window
        self.trend_filter_tolerance = trend_filter_tolerance

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Buy when RSI recovers from the oversold zone, sell when RSI retreats from the overbought zone.

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with rsi, signal, position columns added
        """
        df = df.copy()
        df["rsi"] = calculate_rsi(df["close"], self.period)

        df["signal"] = 0

        # Buy when RSI recovers from oversold zone
        df.loc[(df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold), "signal"] = 1

        # Sell when RSI retreats from overbought zone
        df.loc[
            (df["rsi"] < self.overbought) & (df["rsi"].shift(1) >= self.overbought), "signal"
        ] = -1

        df = apply_trend_filter(
            df,
            self.trend_filter_enabled,
            self.trend_filter_window,
            self.trend_filter_tolerance,
        )
        df = forward_fill_position(df)

        return df
