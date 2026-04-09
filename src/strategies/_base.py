"""
交易策略基类

所有交易策略必须继承 TradingStrategy 并实现 generate_signals() 方法。
信号约定: 1=买入, -1=卖出, 0=持有
"""

import pandas as pd
from typing import Dict, List


class TradingStrategy:
    """
    交易策略基类

    所有交易策略的抽象基类，定义了策略的公共接口和属性。

    Attributes:
        name: 策略名称标识符
        positions: 持仓记录字典 (策略内部使用)
        signals: 信号记录列表 (策略内部使用)

    子类必须实现:
        generate_signals(df) -> pd.DataFrame

    信号列约定:
        signal: 1=买入信号, -1=卖出信号, 0=无信号(持有)
        position: 持仓状态 (1=持仓中, 0=空仓)
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        初始化策略

        Args:
            name: 策略名称，用于标识和日志
        """
        self.name = name
        self.positions: Dict = {}
        self.signals: List = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号 (子类必须实现)

        根据输入的价格数据，计算技术指标并生成买卖信号。

        Args:
            df: 包含 OHLCV 数据的 DataFrame
                必须包含列: close, high, low, volume

        Returns:
            添加了 'signal' 和 'position' 列的 DataFrame
            signal: 1=买入, -1=卖出, 0=持有
            position: 持仓状态 (1=持仓, 0=空仓)

        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标 (可选覆盖)

        子类可覆盖此方法，在 generate_signals() 之前计算所需的指标列。

        Args:
            df: 原始价格数据 DataFrame

        Returns:
            添加了指标列的 DataFrame
        """
        return df
