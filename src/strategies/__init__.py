"""
交易策略包

提供 13 种内置交易策略，涵盖趋势跟踪、均值回归、动量、网格等类别。

策略分类:
    趋势跟踪: MA Cross, MACD, Breakout, Momentum, ATR StopLoss
    均值回归: RSI, Bollinger Bands, Mean Reversion, VWAP, Stochastic
    综合评分: MultiFactor
    特殊策略: Grid, Martingale

使用工厂函数创建策略:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy("rsi", period=14, oversold=30, overbought=70)
    >>> result_df = strategy.generate_signals(df)

直接导入策略类:
    >>> from strategies import MovingAverageCrossStrategy, RSIStrategy
    >>> strategy = RSIStrategy(period=14)
"""

from strategies._base import TradingStrategy
from strategies._helpers import forward_fill_position, detect_crossover, calculate_rsi

from strategies.ma_cross import MovingAverageCrossStrategy
from strategies.rsi import RSIStrategy
from strategies.bollinger import BollingerBandsStrategy
from strategies.multi_factor import MultiFactorStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.macd import MACDStrategy
from strategies.breakout import BreakoutStrategy
from strategies.vwap import VWAPStrategy
from strategies.momentum import MomentumStrategy
from strategies.atr_stop import ATRStopLossStrategy
from strategies.stochastic import StochasticStrategy
from strategies.grid import GridStrategy
from strategies.martingale import MartingaleStrategy


def get_strategy(name: str, **kwargs) -> TradingStrategy:
    """
    策略工厂函数 - 根据名称创建策略实例

    Args:
        name: 策略名称，支持以下值:
            - ma_cross: 双均线交叉策略
            - rsi: RSI 超买超卖策略
            - bollinger: 布林带策略
            - multi_factor: 多因子组合策略
            - mean_reversion: 均值回归策略
            - macd: MACD 趋势策略
            - breakout: 突破策略
            - vwap: VWAP 均值回归策略
            - momentum: 动量策略
            - atr_stop: ATR 动态止损策略
            - stochastic: 随机指标策略
            - grid: 网格交易策略
            - martingale: 马丁格尔策略
        **kwargs: 传递给策略构造函数的参数

    Returns:
        TradingStrategy 子类实例

    Raises:
        ValueError: 策略名称不存在时抛出

    Example:
        >>> strategy = get_strategy("rsi", period=14, oversold=30, overbought=70)
        >>> result_df = strategy.generate_signals(df)
    """
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
        "grid": GridStrategy,
        "martingale": MartingaleStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)
