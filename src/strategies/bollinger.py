"""
布林带策略

价格触及下轨后回升买入，触及上轨后回落卖出。
属于均值回归类策略，适合中高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('bollinger', window=20, num_std=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position
from strategies.constants import DEFAULT_BB_WINDOW, DEFAULT_BB_NUM_STD


class BollingerBandsStrategy(TradingStrategy):
    """
    布林带策略 (中高频)

    布林带由中轨（移动平均线）和上下轨（±N倍标准差）构成。
    价格触及下轨后回升时买入，触及上轨后回落时卖出。

    Args:
        window: 移动平均线窗口 (默认 20)
        num_std: 标准差倍数 (默认 2.0)

    生成的指标列:
        middle_band: 中轨 (移动平均线)
        std: 滚动标准差
        upper_band: 上轨
        lower_band: 下轨
        bandwidth: 带宽 (上下轨差/中轨)
    """

    def __init__(self, window: int = DEFAULT_BB_WINDOW, num_std: float = DEFAULT_BB_NUM_STD):
        super().__init__("Bollinger_Bands")
        self.window = window
        self.num_std = num_std

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了布林带相关列的 DataFrame
        """
        df = df.copy()
        df["middle_band"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        df["upper_band"] = df["middle_band"] + (df["std"] * self.num_std)
        df["lower_band"] = df["middle_band"] - (df["std"] * self.num_std)
        df["bandwidth"] = (df["upper_band"] - df["lower_band"]) / df["middle_band"].replace(0, float("nan"))
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        价格从下方突破下轨后回升买入，从上方突破上轨后回落卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 signal 和 position 列的 DataFrame
        """
        df = self.calculate_indicators(df)

        df["signal"] = 0

        # 价格跌破下轨后回升买入
        df.loc[
            (df["close"] > df["lower_band"]) & (df["close"].shift(1) <= df["lower_band"].shift(1)),
            "signal",
        ] = 1

        # 价格突破上轨后回落卖出
        df.loc[
            (df["close"] < df["upper_band"]) & (df["close"].shift(1) >= df["upper_band"].shift(1)),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
