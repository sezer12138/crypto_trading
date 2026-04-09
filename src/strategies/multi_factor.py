"""
多因子组合策略

结合均线趋势、RSI、成交量、波动率等多个因子进行综合评分。
属于综合评分类策略，适合高频交易。

评分体系:
    - 均线趋势 (权重 30%): 短期均线与长期均线的方向
    - RSI 因子 (权重 30%): 归一化 RSI 值 (范围 -1 到 1)
    - 成交量确认 (权重 20%): 成交量比值与阈值比较
    - 波动率过滤 (权重 20%): 高波动率扣分

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('multi_factor', ma_short=5, ma_long=20)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy
from strategies._helpers import calculate_rsi
from strategies.constants import (
    DEFAULT_MA_SHORT, DEFAULT_MA_LONG,
    DEFAULT_RSI_PERIOD, DEFAULT_VOLUME_THRESHOLD,
    WEIGHT_MA_TREND, WEIGHT_RSI, WEIGHT_VOLUME, WEIGHT_VOLATILITY,
    SCORE_BUY_THRESHOLD, SCORE_SELL_THRESHOLD,
    VOLUME_LOW_RATIO, VOLATILITY_QUANTILE,
    DEFAULT_VOLUME_MA_WINDOW, DEFAULT_VOLATILITY_WINDOW,
)


class MultiFactorStrategy(TradingStrategy):
    """
    多因子组合策略 (高频)

    结合均线、RSI、成交量、波动率等多个因子，通过加权评分系统生成信号。
    综合评分 > 0.5 时买入，< -0.5 时卖出。

    Args:
        ma_short: 短期均线窗口 (默认 5)
        ma_long: 长期均线窗口 (默认 20)
        rsi_period: RSI 计算周期 (默认 14)
        volume_threshold: 成交量比值阈值 (默认 1.5)

    生成的指标列:
        ma_short, ma_long, ma_trend: 均线及趋势方向
        rsi, rsi_norm: RSI 及其归一化值
        volume_ma, volume_ratio: 成交量均线及比值
        returns, volatility: 收益率及波动率
        score: 综合评分 (范围约 -1 到 1)
    """

    def __init__(
        self,
        ma_short: int = DEFAULT_MA_SHORT,
        ma_long: int = DEFAULT_MA_LONG,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
    ):
        super().__init__("Multi_Factor")
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.volume_threshold = volume_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标

        依次计算均线系统、RSI、成交量因子和波动率指标。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了各项指标列的 DataFrame
        """
        df = df.copy()

        # 均线系统
        df["ma_short"] = df["close"].rolling(window=self.ma_short).mean()
        df["ma_long"] = df["close"].rolling(window=self.ma_long).mean()
        df["ma_trend"] = (df["ma_short"] > df["ma_long"]).astype(int)

        # RSI (使用共享的 RSI 计算函数)
        df["rsi"] = calculate_rsi(df["close"], self.rsi_period)
        df["rsi_norm"] = (df["rsi"] - 50) / 50  # 归一化到 [-1, 1]

        # 成交量因子
        df["volume_ma"] = df["volume"].rolling(window=DEFAULT_VOLUME_MA_WINDOW).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 波动率
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=DEFAULT_VOLATILITY_WINDOW).std()

        return df

    def _calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多因子综合评分

        评分体系:
            - 均线趋势 (权重 30%): 短期均线在长期均线上方加分，下方减分
            - RSI (权重 30%): 归一化 RSI 值 (范围 -1 到 1)
            - 成交量确认 (权重 20%): 成交量放大加分，缩小减分
            - 波动率过滤 (权重 20%): 高波动率环境下减分

        Args:
            df: 包含技术指标的 DataFrame

        Returns:
            添加了 'score' 列的 DataFrame，范围约 [-1, 1]
        """
        df["score"] = 0.0

        # 均线趋势 (权重 30%)
        df.loc[df["ma_short"] > df["ma_long"], "score"] += WEIGHT_MA_TREND
        df.loc[df["ma_short"] < df["ma_long"], "score"] -= WEIGHT_MA_TREND

        # RSI (权重 30%)
        df["score"] += df["rsi_norm"] * WEIGHT_RSI

        # 成交量确认 (权重 20%)
        df.loc[df["volume_ratio"] > self.volume_threshold, "score"] += WEIGHT_VOLUME
        df.loc[df["volume_ratio"] < VOLUME_LOW_RATIO, "score"] -= WEIGHT_VOLUME

        # 波动率过滤 (权重 20%)
        vol_threshold = df["volatility"].quantile(VOLATILITY_QUANTILE)
        df.loc[df["volatility"] > vol_threshold, "score"] -= WEIGHT_VOLATILITY

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        综合评分 > 买入阈值 (0.5) 时产生买入信号，
        综合评分 < 卖出阈值 (-0.5) 时产生卖出信号。

        与简单策略不同，此策略使用逐行持仓管理，
        因为评分系统可能频繁在阈值附近波动。

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 score, signal, position 列的 DataFrame
        """
        df = self.calculate_indicators(df)
        df = self._calculate_score(df)

        # 生成信号
        df["signal"] = 0
        df.loc[df["score"] > SCORE_BUY_THRESHOLD, "signal"] = 1
        df.loc[df["score"] < SCORE_SELL_THRESHOLD, "signal"] = -1

        # 持仓状态（逐行管理，避免评分波动导致的信号抖动）
        df["position"] = 0
        position = 0
        for i in range(len(df)):
            if df["signal"].iloc[i] == 1:
                position = 1
            elif df["signal"].iloc[i] == -1:
                position = 0
            df.loc[df.index[i], "position"] = position

        return df
