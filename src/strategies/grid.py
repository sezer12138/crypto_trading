"""
网格交易策略

在预设的价格区间内，每隔一定间距挂单买入和卖出。
适合震荡行情，不适合单边趋势市场。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('grid', lower_price=50000, upper_price=70000)
    >>> result_df = strategy.generate_signals(df)
"""

import numpy as np
import pandas as pd
from strategies._base import TradingStrategy


class GridStrategy(TradingStrategy):
    """
    网格交易策略 (Grid Trading)

    在价格区间内均匀设置网格线:
    - 价格下跌穿过网格线时买入
    - 价格上涨穿过网格线时卖出

    适合震荡行情，单边趋势中可能产生较大亏损。

    Args:
        lower_price: 网格下界价格
        upper_price: 网格上界价格
        grid_num: 网格数量 (默认 10)
        amount_per_grid: 每格交易量 (默认 0.01)

    生成的指标列:
        position: 当前累计持仓量
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

        # 计算网格价格
        self.grid_prices = np.linspace(lower_price, upper_price, grid_num)
        self.buy_grids = [False] * grid_num
        self.sell_grids = [False] * grid_num

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        逐行遍历价格数据，检查是否穿过网格线:
        - 价格跌破网格线: 买入信号 (1)
        - 价格突破网格线: 卖出信号 (-1)

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 signal, position 列的 DataFrame
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

            # 检查是否穿过任何网格线
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
