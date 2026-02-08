"""
Trading Strategies
中高频交易策略实现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """交易策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions = {}
        self.signals = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Returns:
            DataFrame with signal column (1:买入, -1:卖出, 0:持有)
        """
        raise NotImplementedError

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        return df


class MovingAverageCrossStrategy(TradingStrategy):
    """
    双均线交叉策略 (中频)
    短期均线上穿长期均线买入，下穿卖出
    """

    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__("MA_Cross")
        self.short_window = short_window
        self.long_window = long_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["ma_long"] = df["close"].rolling(window=self.long_window).mean()
        df["ma_diff"] = df["ma_short"] - df["ma_long"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.calculate_indicators(df)

        # 信号逻辑
        df["signal"] = 0

        # 金叉：短均线上穿长均线
        df.loc[
            (df["ma_short"] > df["ma_long"]) & (df["ma_short"].shift(1) <= df["ma_long"].shift(1)),
            "signal",
        ] = 1

        # 死叉：短均线下穿长均线
        df.loc[
            (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1)),
            "signal",
        ] = -1

        # 持仓状态
        df["position"] = df["signal"].replace(to_replace=0, method="ffill")

        return df


class RSIStrategy(TradingStrategy):
    """
    RSI 超买超卖策略 (中高频)
    RSI < 30 超卖买入，RSI > 70 超买卖出
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI_Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = self.calculate_rsi(df["close"])

        df["signal"] = 0

        # RSI 从超卖区回升买入
        df.loc[(df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold), "signal"] = 1

        # RSI 从超买区回落卖出
        df.loc[
            (df["rsi"] < self.overbought) & (df["rsi"].shift(1) >= self.overbought), "signal"
        ] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")

        return df


class BollingerBandsStrategy(TradingStrategy):
    """
    布林带策略 (中高频)
    价格触及下轨买入，触及上轨卖出
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__("Bollinger_Bands")
        self.window = window
        self.num_std = num_std

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["middle_band"] = df["close"].rolling(window=self.window).mean()
        df["std"] = df["close"].rolling(window=self.window).std()
        df["upper_band"] = df["middle_band"] + (df["std"] * self.num_std)
        df["lower_band"] = df["middle_band"] - (df["std"] * self.num_std)
        df["bandwidth"] = (df["upper_band"] - df["lower_band"]) / df["middle_band"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")

        return df


class MultiFactorStrategy(TradingStrategy):
    """
    多因子组合策略 (高频)
    结合均线、RSI、成交量等多个因子
    """

    def __init__(
        self,
        ma_short: int = 5,
        ma_long: int = 20,
        rsi_period: int = 14,
        volume_threshold: float = 1.5,
    ):
        super().__init__("Multi_Factor")
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.volume_threshold = volume_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 均线系统
        df["ma_short"] = df["close"].rolling(window=self.ma_short).mean()
        df["ma_long"] = df["close"].rolling(window=self.ma_long).mean()
        df["ma_trend"] = (df["ma_short"] > df["ma_long"]).astype(int)

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_norm"] = (df["rsi"] - 50) / 50  # 归一化到 [-1, 1]

        # 成交量因子
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 波动率
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)

        # 综合打分系统 (-1 到 1)
        df["score"] = 0

        # 均线趋势 (权重 0.3)
        df.loc[df["ma_short"] > df["ma_long"], "score"] += 0.3
        df.loc[df["ma_short"] < df["ma_long"], "score"] -= 0.3

        # RSI (权重 0.3)
        df["score"] += df["rsi_norm"] * 0.3

        # 成交量确认 (权重 0.2)
        df.loc[df["volume_ratio"] > self.volume_threshold, "score"] += 0.2
        df.loc[df["volume_ratio"] < 0.5, "score"] -= 0.2

        # 波动率过滤 (权重 0.2)
        vol_threshold = df["volatility"].quantile(0.7)
        df.loc[df["volatility"] > vol_threshold, "score"] -= 0.2

        # 生成信号
        df["signal"] = 0
        df.loc[df["score"] > 0.5, "signal"] = 1
        df.loc[df["score"] < -0.5, "signal"] = -1

        # 持仓状态
        df["position"] = 0
        position = 0
        for i in range(len(df)):
            if df["signal"].iloc[i] == 1:
                position = 1
            elif df["signal"].iloc[i] == -1:
                position = 0
            df.loc[df.index[i], "position"] = position

        return df


class MeanReversionStrategy(TradingStrategy):
    """
    均值回归策略 (高频)
    价格偏离均值过大时反向操作
    """

    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__("Mean_Reversion")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 计算Z-score
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


class MACDStrategy(TradingStrategy):
    """
    MACD趋势策略 (中频)
    MACD线上穿信号线买入，下穿卖出
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD_Strategy")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # MACD上穿信号线买入
        df.loc[
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)),
            "signal",
        ] = 1

        # MACD下穿信号线卖出
        df.loc[
            (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)),
            "signal",
        ] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


class BreakoutStrategy(TradingStrategy):
    """
    突破策略 (高频)
    价格突破N周期高点买入，突破N周期低点卖出
    """

    def __init__(self, window: int = 20, confirmation: bool = True):
        super().__init__("Breakout_Strategy")
        self.window = window
        self.confirmation = confirmation

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["high_n"] = df["high"].rolling(window=self.window).max()
        df["low_n"] = df["low"].rolling(window=self.window).min()
        df["signal"] = 0

        if self.confirmation:
            # 需要收盘价确认突破
            df.loc[
                (df["close"] > df["high_n"].shift(1)) &
                (df["close"].shift(1) <= df["high_n"].shift(2)),
                "signal",
            ] = 1
            df.loc[
                (df["close"] < df["low_n"].shift(1)) &
                (df["close"].shift(1) >= df["low_n"].shift(2)),
                "signal",
            ] = -1
        else:
            # 即时突破
            df.loc[df["high"] > df["high_n"].shift(1), "signal"] = 1
            df.loc[df["low"] < df["low_n"].shift(1), "signal"] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


class VWAPStrategy(TradingStrategy):
    """
    VWAP均值回归策略 (高频)
    价格低于VWAP买入，高于VWAP卖出
    """

    def __init__(self, window: int = 20, deviation: float = 0.01):
        super().__init__("VWAP_Strategy")
        self.window = window
        self.deviation = deviation

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=self.window).sum() / \
                     df["volume"].rolling(window=self.window).sum()
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # 价格低于VWAP一定幅度买入
        df.loc[df["vwap_dev"] < -self.deviation, "signal"] = 1
        # 价格高于VWAP一定幅度卖出
        df.loc[df["vwap_dev"] > self.deviation, "signal"] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


class MomentumStrategy(TradingStrategy):
    """
    动量策略 (中频)
    基于价格变化率(ROC)和动量指标
    """

    def __init__(self, roc_period: int = 10, momentum_period: int = 14, threshold: float = 0.02):
        super().__init__("Momentum_Strategy")
        self.roc_period = roc_period
        self.momentum_period = momentum_period
        self.threshold = threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 变化率
        df["roc"] = (df["close"] - df["close"].shift(self.roc_period)) / df["close"].shift(self.roc_period)
        # 动量指标
        df["momentum"] = df["close"] - df["close"].shift(self.momentum_period)
        df["momentum_norm"] = df["momentum"] / df["close"] * 100
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # ROC转正且动量为正买入
        df.loc[
            (df["roc"] > self.threshold) & (df["momentum_norm"] > 0) &
            (df["roc"].shift(1) <= self.threshold),
            "signal",
        ] = 1

        # ROC转负且动量为负卖出
        df.loc[
            (df["roc"] < -self.threshold) & (df["momentum_norm"] < 0) &
            (df["roc"].shift(1) >= -self.threshold),
            "signal",
        ] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


class ATRStopLossStrategy(TradingStrategy):
    """
    ATR动态止损策略 (高频)
    基于ATR的动态止损和止盈
    """

    def __init__(self, atr_period: int = 14, multiplier: float = 2.0, trend_ma: int = 50):
        super().__init__("ATR_StopLoss_Strategy")
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.trend_ma = trend_ma

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = self.calculate_atr(df)
        df["trend_ma"] = df["close"].rolling(window=self.trend_ma).mean()
        df["signal"] = 0

        # 趋势判断
        uptrend = df["close"] > df["trend_ma"]
        downtrend = df["close"] < df["trend_ma"]

        # 上升趋势中，回调到ATR支撑位买入
        support = df["close"] - df["atr"] * self.multiplier
        resistance = df["close"] + df["atr"] * self.multiplier

        df.loc[uptrend & (df["low"] < support.shift(1)) & (df["close"] > support.shift(1)), "signal"] = 1
        df.loc[downtrend & (df["high"] > resistance.shift(1)) & (df["close"] < resistance.shift(1)), "signal"] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


class StochasticStrategy(TradingStrategy):
    """
    随机指标策略 (中高频)
    K线上穿D线且低于20买入，K线下穿D线且高于80卖出
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth: int = 3):
        super().__init__("Stochastic_Strategy")
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lowest_low = df["low"].rolling(window=self.k_period).min()
        highest_high = df["high"].rolling(window=self.k_period).max()
        df["k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        df["d"] = df["k"].rolling(window=self.d_period).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # K上穿D且在超卖区买入
        df.loc[
            (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1)) & (df["k"] < 20),
            "signal",
        ] = 1

        # K下穿D且在超买区卖出
        df.loc[
            (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1)) & (df["k"] > 80),
            "signal",
        ] = -1

        df["position"] = df["signal"].replace(to_replace=0, method="ffill")
        return df


def get_strategy(name: str, **kwargs) -> TradingStrategy:
    """策略工厂函数"""
    strategies = {
        "ma_cross": MovingAverageCrossStrategy,
        "rsi": RSIStrategy,
        "bollinger": BollingerBandsStrategy,
        "multi_factor": MultiFactorStrategy,
        "mean_reversion": MeanReversionStrategy,
        "macd": MACDStrategy,
        "breakout": BreakoutStrategy,
        "vwap": VWAPStrategy,
        "momentum": MomentumStrategy,
        "atr_stop": ATRStopLossStrategy,
        "stochastic": StochasticStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)


if __name__ == "__main__":
    # 测试策略
    import numpy as np

    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # 测试多因子策略
    strategy = MultiFactorStrategy()
    result = strategy.generate_signals(df)

    print(f"策略: {strategy.name}")
    print(f"买入信号数: {(result['signal'] == 1).sum()}")
    print(f"卖出信号数: {(result['signal'] == -1).sum()}")
    print("\n前5行数据:")
    print(result[["close", "ma_short", "ma_long", "rsi", "score", "signal"]].head())
