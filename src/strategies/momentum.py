"""
动量策略

基于价格变化率 (ROC) 和动量指标的交易策略。
ROC 转正且动量为正时买入，转负且动量为负时卖出。
属于趋势跟踪类策略，适合中频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('momentum', roc_period=10, threshold=0.02)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class MomentumStrategy(TradingStrategy):
    """
    动量策略 (中频)

    结合变化率 (ROC) 和动量指标:
    - ROC 转正且动量为正时买入
    - ROC 转负且动量为负时卖出

    Args:
        roc_period: 变化率计算周期 (默认 10)
        momentum_period: 动量计算周期 (默认 14)
        threshold: ROC 阈值 (默认 0.02 = 2%)

    生成的指标列:
        roc: 变化率 (百分比)
        momentum: 动量值 (价格差)
        momentum_norm: 归一化动量 (百分比)
    """

    def __init__(self, roc_period: int = 10, momentum_period: int = 14, threshold: float = 0.02):
        super().__init__("Momentum_Strategy")
        self.roc_period = roc_period
        self.momentum_period = momentum_period
        self.threshold = threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 roc, momentum, momentum_norm 列的 DataFrame
        """
        df = df.copy()
        # 变化率
        shifted = df["close"].shift(self.roc_period)
        df["roc"] = (df["close"] - shifted) / shifted.replace(0, float("nan"))
        # 动量指标
        df["momentum"] = df["close"] - df["close"].shift(self.momentum_period)
        df["momentum_norm"] = df["momentum"] / df["close"] * 100
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        ROC 转正且动量为正买入，ROC 转负且动量为负卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 signal 和 position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # ROC 转正且动量为正买入
        df.loc[
            (df["roc"] > self.threshold) & (df["momentum_norm"] > 0) &
            (df["roc"].shift(1) <= self.threshold),
            "signal",
        ] = 1

        # ROC 转负且动量为负卖出
        df.loc[
            (df["roc"] < -self.threshold) & (df["momentum_norm"] < 0) &
            (df["roc"].shift(1) >= -self.threshold),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
