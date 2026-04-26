"""
Martingale Strategy

Double the position after each loss until a profit is realized. A high-risk gambling strategy intended
only for backtesting research.

WARNING - Risk Alert:
    - The Martingale strategy exponentially increases position size after consecutive losses
    - On the Nth doubling, the single position size is 2^N times the initial amount
    - After multiple consecutive losses, cumulative losses can be enormous
    - This strategy is strictly for backtesting research only; NEVER use it in live trading

Usage example:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('martingale', base_amount=0.001, multiplier=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy


class MartingaleStrategy(TradingStrategy):
    """
    Martingale Strategy

    Core logic:
    1. Initial buy of base_amount
    2. If price drops to the stop-loss percentage, double the buy (multiply by multiplier)
    3. Continue doubling until max_steps is reached or price recovers to take-profit target
    4. Force stop-loss exit after reaching maximum doubling steps

    WARNING: This strategy is extremely high-risk and is intended only for backtesting research.

    Args:
        base_amount: Initial buy amount (default 0.001)
        multiplier: Doubling multiplier (default 2.0)
        max_steps: Maximum number of doubling steps (default 5)
        target_profit: Take-profit target percentage (default 0.01 = 1%)
        stop_loss: Single-step stop-loss trigger percentage (default 0.05 = 5%)

    Generated indicator columns:
        position: Current position intensity (step count + 1), 0 means no position
    """

    def __init__(
        self,
        base_amount: float = 0.001,
        multiplier: float = 2.0,
        max_steps: int = 5,
        target_profit: float = 0.01,
        stop_loss: float = 0.05,
    ):
        super().__init__("Martingale_Strategy")
        self.base_amount = base_amount
        self.multiplier = multiplier
        self.max_steps = max_steps
        self.target_profit = target_profit
        self.stop_loss = stop_loss

    def _update_martingale_position(
        self,
        current_price: float,
        entry_price: float,
        current_step: int,
        in_position: bool,
    ) -> tuple:
        """
        Update Martingale position based on current price

        Checks take-profit/stop-loss conditions and decides whether to close position or double down.

        Args:
            current_price: Current price
            entry_price: Average entry price
            current_step: Current doubling step
            in_position: Whether currently holding a position

        Returns:
            Tuple of (signal, new_entry_price, new_step, new_in_position)
            signal: 0=No action, 1=Double buy, -1=Close position
        """
        if not in_position:
            return 1, current_price, 0, True

        price_change = (current_price - entry_price) / entry_price

        # Target take-profit reached
        if price_change >= self.target_profit:
            return -1, 0.0, 0, False

        # Loss reaches stop-loss trigger line (tightens as steps increase)
        stop_threshold = self.stop_loss / (current_step + 1)
        if price_change <= -stop_threshold:
            if current_step < self.max_steps:
                # Double buy, update average entry price
                total_weight = sum(self.multiplier**j for j in range(current_step + 2))
                last_weight = self.multiplier ** (current_step + 1)
                new_entry_price = (
                    entry_price * (total_weight - last_weight) + current_price * last_weight
                ) / total_weight
                return 1, new_entry_price, current_step + 1, True
            else:
                # Exceeded maximum doubling steps, stop-loss exit
                return -1, 0.0, 0, False

        return 0, entry_price, current_step, True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals

        Executes Martingale logic row by row:
        1. Initial buy when no position
        2. Check take-profit/stop-loss conditions when holding position
        3. Double down on loss, sell all on take-profit

        Args:
            df: DataFrame containing a 'close' column

        Returns:
            DataFrame with signal, position columns added
        """
        df = df.copy()
        prices = df["close"].values
        signals = [0] * len(df)
        positions = [0.0] * len(df)

        current_step = 0
        entry_price = 0.0
        in_position = False

        for i in range(1, len(df)):
            signal, entry_price, current_step, in_position = self._update_martingale_position(
                prices[i], entry_price, current_step, in_position
            )

            signals[i] = signal
            positions[i] = (current_step + 1) if in_position else 0

        df["signal"] = signals
        df["position"] = positions
        return df
