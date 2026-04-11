"""
VWAP 均值回归策略

价格低于 VWAP 一定幅度时买入，高于 VWAP 一定幅度时卖出。
属于均值回归类策略，适合高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('vwap', window=20, deviation=0.01)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class VWAPStrategy(TradingStrategy):
    """
    VWAP 均值回归策略 (高频)

    VWAP (Volume Weighted Average Price) 是成交量加权平均价。
    价格偏离 VWAP 超过设定幅度时，预期将回归 VWAP。

    Args:
        window: VWAP 计算窗口 (默认 20)
        deviation: 偏离阈值 (默认 0.01 = 1%)

    生成的指标列:
        vwap: 成交量加权平均价
        vwap_dev: 价格偏离 VWAP 的百分比
    """

    def __init__(self, window: int = 20, deviation: float = 0.01):
        super().__init__("VWAP_Strategy")
        self.window = window
        self.deviation = deviation

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 VWAP 指标

        Args:
            df: 包含 high, low, close, volume 列的 DataFrame

        Returns:
            添加了 vwap, vwap_dev 列的 DataFrame
        """
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=self.window).sum() / \
                     df["volume"].rolling(window=self.window).sum()
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, float("nan"))
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        价格低于 VWAP 超过偏离阈值时买入，
        价格高于 VWAP 超过偏离阈值时卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 vwap, vwap_dev, signal, position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # 价格低于 VWAP 一定幅度买入
        df.loc[df["vwap_dev"] < -self.deviation, "signal"] = 1
        # 价格高于 VWAP 一定幅度卖出
        df.loc[df["vwap_dev"] > self.deviation, "signal"] = -1

        df = forward_fill_position(df)
        return df
