"""
突破策略

价格突破 N 周期高点买入，突破 N 周期低点卖出。
属于趋势跟踪类策略，适合高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('breakout', window=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class BreakoutStrategy(TradingStrategy):
    """
    突破策略 (高频)

    价格突破 N 周期最高价时买入，突破 N 周期最低价时卖出。
    支持确认模式（收盘价确认）和即时模式。

    Args:
        window: 回看窗口 (默认 20)
        confirmation: 是否需要收盘价确认突破 (默认 True)

    生成的指标列:
        high_n: N 周期最高价
        low_n: N 周期最低价
    """

    def __init__(self, window: int = 20, confirmation: bool = True):
        super().__init__("Breakout_Strategy")
        self.window = window
        self.confirmation = confirmation

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        确认模式: 收盘价突破前一日 N 周期高低点时产生信号。
        即时模式: 盘中价格突破时立即产生信号。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 high_n, low_n, signal, position 列的 DataFrame
        """
        df = df.copy()
        df["high_n"] = df["high"].rolling(window=self.window).max()
        df["low_n"] = df["low"].rolling(window=self.window).min()
        df["signal"] = 0

        if self.confirmation:
            # 需要收盘价确认突破
            df.loc[
                (df["close"] > df["high_n"].shift(1)) &
                (df["close"].shift(1) <= df["high_n"].shift(2)),
                "signal",
            ] = 1
            df.loc[
                (df["close"] < df["low_n"].shift(1)) &
                (df["close"].shift(1) >= df["low_n"].shift(2)),
                "signal",
            ] = -1
        else:
            # 即时突破
            df.loc[df["high"] > df["high_n"].shift(1), "signal"] = 1
            df.loc[df["low"] < df["low_n"].shift(1), "signal"] = -1

        df = forward_fill_position(df)
        return df
