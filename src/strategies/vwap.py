"""
VWAP Mean Reversion Strategy

Buy when price falls below VWAP by a dynamic deviation threshold (ATR-based),
sell when price rises above VWAP by the same threshold. Uses event-based signals
to prevent over-trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('vwap', window=20, deviation=0.01)
    >>> result_df = strategy.generate_signals(df)
    >>> # With dynamic deviation disabled (fixed threshold):
    >>> strategy = get_strategy('vwap', window=20, deviation=0.02, dynamic_deviation=False)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import convert_to_event_signals, forward_fill_position
from strategies.constants import (
    VWAP_ATR_WINDOW,
    VWAP_DYNAMIC_MULTIPLIER,
    VWAP_MIN_DEVIATION,
)


class VWAPStrategy(TradingStrategy):
    """
    VWAP Mean Reversion Strategy with Dynamic Deviation

    VWAP (Volume Weighted Average Price) is the volume-weighted average price.
    When price deviates from VWAP beyond the dynamic threshold, it is expected
    to revert to VWAP.

    Uses ATR-based dynamic deviation thresholds that adapt to market volatility,
    preventing the fixed-threshold problem where a single percentage value is
    too tight for low-volatility periods and too loose for high-volatility ones.

    Args:
        window: VWAP calculation window (default 20)
        deviation: Fixed deviation threshold when dynamic_deviation is False
                   (default 0.01 = 1%)
        dynamic_deviation: Use ATR-based dynamic deviation instead of fixed
                           threshold (default True)
        atr_window: ATR rolling window for dynamic deviation calculation
                    (default VWAP_ATR_WINDOW = 20)

    Generated indicator columns:
        vwap: Volume-weighted average price
        vwap_dev: Percentage deviation of price from VWAP
        atr: Average True Range (when dynamic_deviation is True)
        dynamic_dev: Dynamic deviation threshold per bar (when dynamic_deviation is True)
    """

    def __init__(
        self,
        window: int = 20,
        deviation: float = 0.01,
        dynamic_deviation: bool = True,
        atr_window: int = VWAP_ATR_WINDOW,
    ):
        super().__init__("VWAP_Strategy")
        self.window = window
        self.deviation = deviation
        self.dynamic_deviation = dynamic_deviation
        self.atr_window = atr_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP and deviation indicators.

        When dynamic_deviation is enabled, also calculates ATR and derives a
        per-bar deviation threshold that adapts to recent volatility. The dynamic
        threshold is floored at VWAP_MIN_DEVIATION to avoid triggering on noise.

        Args:
            df: DataFrame containing high, low, close, volume columns

        Returns:
            DataFrame with vwap, vwap_dev, and optionally atr, dynamic_dev
            columns added.
        """
        df = df.copy()

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=self.window).sum() / df[
            "volume"
        ].rolling(window=self.window).sum()
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, float("nan"))

        if self.dynamic_deviation:
            # Calculate ATR as mean of (high - low) over the rolling window
            df["atr"] = (df["high"] - df["low"]).rolling(window=self.atr_window).mean()

            # Derive dynamic deviation: normalized ATR scaled by multiplier, floored
            df["dynamic_dev"] = (
                df["atr"] / df["close"].replace(0, float("nan")) * VWAP_DYNAMIC_MULTIPLIER
            ).clip(lower=VWAP_MIN_DEVIATION)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate event-based trading signals.

        1. Calculate VWAP and deviation indicators.
        2. Assign state-based signals using dynamic or fixed deviation threshold.
        3. Convert to event-based signals (only first bar of each state change).
        4. Forward-fill positions.

        Buy when price falls below VWAP beyond the deviation threshold,
        sell when price rises above VWAP beyond the deviation threshold.

        Args:
            df: DataFrame containing OHLCV data

        Returns:
            DataFrame with vwap, vwap_dev, signal, position columns added.
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        if self.dynamic_deviation:
            # Use dynamic deviation threshold per bar
            dev_threshold = df["dynamic_dev"]
            # Buy when price falls below VWAP beyond the dynamic threshold
            df.loc[df["vwap_dev"] < -dev_threshold, "signal"] = 1
            # Sell when price rises above VWAP beyond the dynamic threshold
            df.loc[df["vwap_dev"] > dev_threshold, "signal"] = -1
        else:
            # Use fixed deviation threshold
            df.loc[df["vwap_dev"] < -self.deviation, "signal"] = 1
            df.loc[df["vwap_dev"] > self.deviation, "signal"] = -1

        # Convert state-based signals to event-based to prevent over-trading
        df = convert_to_event_signals(df)

        df = forward_fill_position(df)
        return df
