"""
均值回归策略

价格偏离均值过大时（Z-score 超过阈值）反向操作，
回归均值后平仓。属于均值回归类策略，适合高频交易。

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('mean_reversion', window=20, entry_z=2.0, exit_z=0.5)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies.constants import DEFAULT_MEAN_REVERSION_WINDOW, DEFAULT_ENTRY_Z, DEFAULT_EXIT_Z


class MeanReversionStrategy(TradingStrategy):
    """
    均值回归策略 (高频)

    基于价格 Z-score 的均值回归策略:
    - Z-score 低于 -entry_z 时买入（价格显著低于均值）
    - Z-score 高于 entry_z 时卖出（价格显著高于均值）
    - Z-score 回归到 exit_z 以内时平仓

    Args:
        window: 滚动均值和标准差的计算窗口 (默认 20)
        entry_z: 入场 Z-score 阈值 (默认 2.0)
        exit_z: 出场 Z-score 阈值 (默认 0.5)

    生成的指标列:
        mean: 滚动均值
        std: 滚动标准差
        zscore: Z-score 值
    """

    def __init__(self, window: int = DEFAULT_MEAN_REVERSION_WINDOW, entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z):
        super().__init__("Mean_Reversion")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Z-score 低于 -entry_z 时买入，高于 entry_z 时卖出，
        Z-score 绝对值小于 exit_z 时平仓（signal=0）。

        注意: 此策略不平仓信号使用 0 而非 -1，
        因此不使用 forward_fill_position，持仓由 Z-score 动态决定。

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 zscore, signal 列的 DataFrame
        """
        df = df.copy()

        # 计算 Z-score
        df["mean"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        df["zscore"] = (df["close"] - df["mean"]) / df["std"]

        df["signal"] = 0

        # Z-score 低于 -entry_z 买入（超卖）
        df.loc[df["zscore"] < -self.entry_z, "signal"] = 1

        # Z-score 高于 entry_z 卖出（超买）
        df.loc[df["zscore"] > self.entry_z, "signal"] = -1

        # 回归均值平仓
        df.loc[abs(df["zscore"]) < self.exit_z, "signal"] = 0

        return df
