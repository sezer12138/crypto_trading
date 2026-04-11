"""
RSI 超买超卖策略

RSI 从超卖区回升时买入，从超买区回落时卖出。
属于均值回归类策略，适合中高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('rsi', period=14, oversold=30, overbought=70)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, calculate_rsi
from strategies.constants import DEFAULT_RSI_PERIOD, DEFAULT_RSI_OVERSOLD, DEFAULT_RSI_OVERBOUGHT


class RSIStrategy(TradingStrategy):
    """
    RSI 超买超卖策略 (中高频)

    RSI < 超卖阈值时视为超卖，RSI 回升至超卖线以上时买入。
    RSI > 超买阈值时视为超买，RSI 回落至超买线以下时卖出。

    Args:
        period: RSI 计算周期 (默认 14)
        oversold: 超卖阈值 (默认 30)
        overbought: 超买阈值 (默认 70)

    生成的指标列:
        rsi: RSI 值 (范围 0-100)
    """

    def __init__(self, period: int = DEFAULT_RSI_PERIOD, oversold: int = DEFAULT_RSI_OVERSOLD, overbought: int = DEFAULT_RSI_OVERBOUGHT):
        super().__init__("RSI_Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        RSI 从超卖区回升买入，从超买区回落卖出。

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 rsi, signal, position 列的 DataFrame
        """
        df = df.copy()
        df["rsi"] = calculate_rsi(df["close"], self.period)

        df["signal"] = 0

        # RSI 从超卖区回升买入
        df.loc[(df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold), "signal"] = 1

        # RSI 从超买区回落卖出
        df.loc[
            (df["rsi"] < self.overbought) & (df["rsi"].shift(1) >= self.overbought), "signal"
        ] = -1

        df = forward_fill_position(df)
        return df
