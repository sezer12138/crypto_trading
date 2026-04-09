"""
双均线交叉策略

短期均线上穿长期均线时买入（金叉），下穿时卖出（死叉）。
属于趋势跟踪类策略，适合中频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('ma_cross', short_window=5, long_window=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, detect_crossover
from strategies.constants import DEFAULT_MA_SHORT, DEFAULT_MA_LONG


class MovingAverageCrossStrategy(TradingStrategy):
    """
    双均线交叉策略 (中频)

    短期均线上穿长期均线买入，下穿卖出。
    适合有明显趋势的市场，震荡行情中容易产生虚假信号。

    Args:
        short_window: 短期均线窗口 (默认 10)
        long_window: 长期均线窗口 (默认 30)

    生成的指标列:
        ma_short: 短期移动平均值
        ma_long: 长期移动平均值
        ma_diff: 短期均线与长期均线的差值
    """

    def __init__(self, short_window: int = DEFAULT_MA_SHORT, long_window: int = DEFAULT_MA_LONG):
        super().__init__("MA_Cross")
        self.short_window = short_window
        self.long_window = long_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算移动平均线指标

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 ma_short, ma_long, ma_diff 列的 DataFrame
        """
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["ma_long"] = df["close"].rolling(window=self.long_window).mean()
        df["ma_diff"] = df["ma_short"] - df["ma_long"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        金叉（短均线上穿长均线）产生买入信号，死叉（短均线下穿长均线）产生卖出信号。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 signal 和 position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df = detect_crossover(df, "ma_short", "ma_long")
        df = forward_fill_position(df)
        return df
