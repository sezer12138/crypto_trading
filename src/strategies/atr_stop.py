"""
ATR 动态止损策略

基于 ATR (Average True Range) 的动态止损和止盈策略。
在上升趋势中回调到 ATR 支撑位时买入，下降趋势中反弹到 ATR 阻力位时卖出。
属于趋势跟踪类策略，适合高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('atr_stop', atr_period=14, multiplier=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import numpy as np
import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position


class ATRStopLossStrategy(TradingStrategy):
    """
    ATR 动态止损策略 (高频)

    ATR 衡量市场波动性，策略根据 ATR 的倍数设定支撑/阻力位:
    - 上升趋势中，价格回调到 ATR 支撑位时买入
    - 下降趋势中，价格反弹到 ATR 阻力位时卖出

    Args:
        atr_period: ATR 计算周期 (默认 14)
        multiplier: ATR 倍数 (默认 2.0)
        trend_ma: 趋势判断均线周期 (默认 50)

    生成的指标列:
        atr: 平均真实范围
        trend_ma: 趋势判断均线
    """

    def __init__(self, atr_period: int = 14, multiplier: float = 2.0, trend_ma: int = 50):
        super().__init__("ATR_StopLoss_Strategy")
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.trend_ma = trend_ma

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算平均真实范围 (ATR)

        ATR 是衡量市场波动性的指标，取以下三者的最大值的移动平均:
        1. 当日最高价 - 当日最低价
        2. |当日最高价 - 昨日收盘价|
        3. |当日最低价 - 昨日收盘价|

        Args:
            df: 包含 high, low, close 列的 DataFrame

        Returns:
            ATR 值序列
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        return tr.rolling(window=self.atr_period).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        上升趋势中价格回调到 ATR 支撑位买入，
        下降趋势中价格反弹到 ATR 阻力位卖出。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 atr, trend_ma, signal, position 列的 DataFrame
        """
        df = df.copy()
        df["atr"] = self.calculate_atr(df)
        df["trend_ma"] = df["close"].rolling(window=self.trend_ma).mean()
        df["signal"] = 0

        # 趋势判断
        uptrend = df["close"] > df["trend_ma"]
        downtrend = df["close"] < df["trend_ma"]

        # 上升趋势中，回调到 ATR 支撑位买入
        support = df["close"] - df["atr"] * self.multiplier
        resistance = df["close"] + df["atr"] * self.multiplier

        df.loc[uptrend & (df["low"] < support.shift(1)) & (df["close"] > support.shift(1)), "signal"] = 1
        df.loc[downtrend & (df["high"] > resistance.shift(1)) & (df["close"] < resistance.shift(1)), "signal"] = -1

        df = forward_fill_position(df)
        return df
