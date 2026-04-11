"""
Grid Trading Strategy

Place buy and sell orders at regular intervals within a preset price range.
Suitable for sideways markets, not suitable for trending markets.

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('grid', lower_price=50000, upper_price=70000)
    >>> result_df = strategy.generate_signals(df)
"""

import numpy as np
import pandas as pd
from strategies._base import TradingStrategy


class GridStrategy(TradingStrategy):
    """
    Grid Trading Strategy

    Evenly sets grid lines within a price range:
    - Buy when price drops through a grid line
    - Sell when price rises through a grid line

    Suitable for sideways markets; may incur significant losses in trending markets.

    Args:
        lower_price: Grid lower bound price
        upper_price: Grid upper bound price
        grid_num: Number of grid levels (default 10)
        amount_per_grid: Trade amount per grid (default 0.01)

    Generated indicator columns:
        position: Current cumulative position amount
    """

    def __init__(
        self,
        lower_price: float,
        upper_price: float,
        grid_num: int = 10,
        amount_per_grid: float = 0.01,
    ):
        super().__init__("Grid_Strategy")
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.grid_num = grid_num
        self.amount_per_grid = amount_per_grid

        # Calculate grid prices
        self.grid_prices = np.linspace(lower_price, upper_price, grid_num)
        self.buy_grids = [False] * grid_num
        self.sell_grids = [False] * grid_num

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Iterates through price data row by row, checking if grid lines are crossed:
        - Price drops below grid line: Buy signal (1)
        - Price rises above grid line: Sell signal (-1)

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with signal, position columns added
        """
        df = df.copy()
        df["signal"] = 0
        df["position"] = 0.0

        current_position = 0.0
        prices = df["close"].values
        signals = [0] * len(df)
        positions = [0.0] * len(df)

        for i in range(1, len(df)):
            current_price = prices[i]
            last_price = prices[i - 1]

            # Check if any grid line is crossed
            for grid_price in self.grid_prices:
                if last_price > grid_price and current_price <= grid_price:
                    signals[i] = 1
                    current_position += self.amount_per_grid
                elif last_price < grid_price and current_price >= grid_price:
                    signals[i] = -1
                    current_position = max(0, current_position - self.amount_per_grid)

            positions[i] = current_position

        df["signal"] = signals
        df["position"] = positions

        return df
