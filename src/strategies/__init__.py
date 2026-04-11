"""
Trading Strategy Package

Provides 13 built-in trading strategies, covering trend following, mean reversion, momentum, grid, and other categories.

Strategy Categories:
    Trend Following: MA Cross, MACD, Breakout, Momentum, ATR StopLoss
    Mean Reversion: RSI, Bollinger Bands, Mean Reversion, VWAP, Stochastic
    Composite Scoring: MultiFactor
    Special Strategies: Grid, Martingale

Create strategies using the factory function:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy("rsi", period=14, oversold=30, overbought=70)
    >>> result_df = strategy.generate_signals(df)

Import strategy classes directly:
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
    Strategy factory function - create strategy instance by name

    Args:
        name: Strategy name, supported values:
            - ma_cross: Dual moving average crossover strategy
            - rsi: RSI overbought/oversold strategy
            - bollinger: Bollinger Bands strategy
            - multi_factor: Multi-factor composite strategy
            - mean_reversion: Mean reversion strategy
            - macd: MACD trend strategy
            - breakout: Breakout strategy
            - vwap: VWAP mean reversion strategy
            - momentum: Momentum strategy
            - atr_stop: ATR dynamic stop-loss strategy
            - stochastic: Stochastic Oscillator strategy
            - grid: Grid trading strategy
            - martingale: Martingale strategy
        **kwargs: Parameters passed to the strategy constructor

    Returns:
        TradingStrategy subclass instance

    Raises:
        ValueError: Raised when strategy name does not exist

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
