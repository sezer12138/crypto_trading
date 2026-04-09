"""
随机指标策略

K 线上穿 D 线且在超卖区 (K<20) 时买入，
K 线下穿 D 线且在超买区 (K>80) 时卖出。
属于均值回归类策略，适合中高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('stochastic', k_period=14, d_period=3)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position
from strategies.constants import STOCHASTIC_OVERSOLD, STOCHASTIC_OVERBOUGHT


class StochasticStrategy(TradingStrategy):
    """
    随机指标策略 (中高频)

    随机指标 (Stochastic Oscillator) 衡量当前价格在一定周期内的相对位置:
    - K 线: 当前价格在 N 周期高低价范围内的百分比位置
    - D 线: K 线的移动平均

    策略逻辑:
    - K 上穿 D 且 K < 20 (超卖区) 买入
    - K 下穿 D 且 K > 80 (超买区) 卖出

    Args:
        k_period: K 线计算周期 (默认 14)
        d_period: D 线平滑周期 (默认 3)
        smooth: K 线预平滑周期 (默认 3)

    生成的指标列:
        k: K 线值 (范围 0-100)
        d: D 线值
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth: int = 3):
        super().__init__("Stochastic_Strategy")
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算随机指标

        Args:
            df: 包含 high, low, close 列的 DataFrame

        Returns:
            添加了 k, d 列的 DataFrame
        """
        df = df.copy()
        lowest_low = df["low"].rolling(window=self.k_period).min()
        highest_high = df["high"].rolling(window=self.k_period).max()
        df["k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        df["d"] = df["k"].rolling(window=self.d_period).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        K 上穿 D 且 K 在超卖区 (<20) 买入，
        K 下穿 D 且 K 在超买区 (>80) 卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 k, d, signal, position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # K 上穿 D 且在超卖区买入
        df.loc[
            (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1)) & (df["k"] < STOCHASTIC_OVERSOLD),
            "signal",
        ] = 1

        # K 下穿 D 且在超买区卖出
        df.loc[
            (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1)) & (df["k"] > STOCHASTIC_OVERBOUGHT),
            "signal",
        ] = -1

        df = forward_fill_position(df)
        return df
