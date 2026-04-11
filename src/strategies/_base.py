"""
Trading Strategy Base Class

All trading strategies must inherit TradingStrategy and implement the generate_signals() method.
Signal convention: 1=Buy, -1=Sell, 0=Hold
"""

import pandas as pd
from typing import Dict, List


class TradingStrategy:
    """
    Trading Strategy Base Class

    Abstract base class for all trading strategies, defining the common interface and attributes.

    Attributes:
        name: Strategy name identifier
        positions: Position records dict (for internal strategy use)
        signals: Signal records list (for internal strategy use)

    Subclasses must implement:
        generate_signals(df) -> pd.DataFrame

    Signal column convention:
        signal: 1=Buy signal, -1=Sell signal, 0=No signal (Hold)
        position: Position state (1=In position, 0=No position)
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize strategy

        Args:
            name: Strategy name, used for identification and logging
        """
        self.name = name
        self.positions: Dict = {}
        self.signals: List = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (subclasses must implement)

        Calculate technical indicators based on input price data and generate buy/sell signals.

        Args:
            df: DataFrame containing OHLCV data
                Must contain columns: close, high, low, volume

        Returns:
            DataFrame with 'signal' and 'position' columns added
            signal: 1=Buy, -1=Sell, 0=Hold
            position: Position state (1=In position, 0=No position)

        Raises:
            NotImplementedError: Raised when subclass has not implemented this method
        """
        raise NotImplementedError

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators (optional override)

        Subclasses can override this method to calculate required indicator columns before generate_signals().

        Args:
            df: Raw price data DataFrame

        Returns:
            DataFrame with indicator columns added
        """
        return df
