"""
Momentum Strategy

Trading strategy based on Rate of Change (ROC) and momentum indicators.
Buy when ROC turns positive and momentum is positive, sell when ROC turns negative and momentum is negative.
A trend-following strategy suitable for medium-frequency trading.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('momentum', buy_roc_period=16, sell_roc_period=12)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position
from strategies.constants import (
    DEFAULT_MOMENTUM_BUY_PERIOD,
    DEFAULT_MOMENTUM_BUY_ROC_PERIOD,
    DEFAULT_MOMENTUM_BUY_THRESHOLD,
    DEFAULT_MOMENTUM_SELL_PERIOD,
    DEFAULT_MOMENTUM_SELL_ROC_PERIOD,
    DEFAULT_MOMENTUM_SELL_THRESHOLD,
)


class MomentumStrategy(TradingStrategy):
    """
    Momentum Strategy (Medium Frequency)

    Combines Rate of Change (ROC) and momentum indicators:
    - Buy when ROC turns positive and momentum is positive
    - Sell when ROC turns negative and momentum is negative

    Args:
        buy_roc_period: Buy rate-of-change calculation period.
        buy_momentum_period: Buy momentum calculation period.
        buy_threshold: Positive ROC threshold for buys.
        sell_roc_period: Sell rate-of-change calculation period.
        sell_momentum_period: Sell momentum calculation period.
        sell_threshold: Absolute negative ROC threshold for sells.

    Generated indicator columns:
        buy_roc, sell_roc: Side-specific rates of change.
        buy_momentum, sell_momentum: Side-specific price differences.
        buy_momentum_norm, sell_momentum_norm: Normalized momentum percentages.
    """

    def __init__(
        self,
        buy_roc_period: int = DEFAULT_MOMENTUM_BUY_ROC_PERIOD,
        buy_momentum_period: int = DEFAULT_MOMENTUM_BUY_PERIOD,
        buy_threshold: float = DEFAULT_MOMENTUM_BUY_THRESHOLD,
        sell_roc_period: int = DEFAULT_MOMENTUM_SELL_ROC_PERIOD,
        sell_momentum_period: int = DEFAULT_MOMENTUM_SELL_PERIOD,
        sell_threshold: float = DEFAULT_MOMENTUM_SELL_THRESHOLD,
    ):
        super().__init__("Momentum_Strategy")
        self.buy_roc_period = buy_roc_period
        self.buy_momentum_period = buy_momentum_period
        self.buy_threshold = buy_threshold
        self.sell_roc_period = sell_roc_period
        self.sell_momentum_period = sell_momentum_period
        self.sell_threshold = sell_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with side-specific ROC and momentum columns added
        """
        df = df.copy()
        for side in ("buy", "sell"):
            roc_period = getattr(self, f"{side}_roc_period")
            momentum_period = getattr(self, f"{side}_momentum_period")
            shifted = df["close"].shift(roc_period)
            df[f"{side}_roc"] = (df["close"] - shifted) / shifted.replace(0, float("nan"))
            df[f"{side}_momentum"] = df["close"] - df["close"].shift(momentum_period)
            df[f"{side}_momentum_norm"] = df[f"{side}_momentum"] / df["close"] * 100
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
            (df["buy_roc"] > self.buy_threshold)
            & (df["buy_momentum_norm"] > 0)
            & (df["buy_roc"].shift(1) <= self.buy_threshold),
            "signal",
        ] = 1

        # Sell when ROC turns negative and momentum is negative
        df.loc[
            (df["sell_roc"] < -self.sell_threshold)
            & (df["sell_momentum_norm"] < 0)
            & (df["sell_roc"].shift(1) >= -self.sell_threshold),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
