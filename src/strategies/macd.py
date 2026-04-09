"""
MACD 趋势策略

MACD 线上穿信号线买入，下穿卖出。
属于趋势跟踪类策略，适合中频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('macd', fast=12, slow=26, signal=9)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, detect_crossover
from strategies.constants import DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, DEFAULT_MACD_SIGNAL


class MACDStrategy(TradingStrategy):
    """
    MACD 趋势策略 (中频)

    MACD (Moving Average Convergence Divergence) 通过快慢 EMA 的差值
    及其信号线的交叉来判断趋势方向。

    Args:
        fast: 快线 EMA 周期 (默认 12)
        slow: 慢线 EMA 周期 (默认 26)
        signal: 信号线 EMA 周期 (默认 9)

    生成的指标列:
        ema_fast: 快速 EMA
        ema_slow: 慢速 EMA
        macd: MACD 线 (快线 - 慢线)
        macd_signal: 信号线 (MACD 的 EMA)
        macd_hist: MACD 柱状图 (MACD - 信号线)
    """

    def __init__(self, fast: int = DEFAULT_MACD_FAST, slow: int = DEFAULT_MACD_SLOW, signal: int = DEFAULT_MACD_SIGNAL):
        super().__init__("MACD_Strategy")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 MACD 指标

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 MACD 相关列的 DataFrame
        """
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        MACD 线上穿信号线买入，下穿信号线卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 signal 和 position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df = detect_crossover(df, "macd", "macd_signal")
        df = forward_fill_position(df)
        return df
