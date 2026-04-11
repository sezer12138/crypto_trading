"""
Visualization Module

Provides chart generation for backtest results, including:
- Equity curve and drawdown analysis
- Price charts and trading signals
- Monthly returns heatmap
- Multi-strategy performance comparison

Usage:
    >>> from visualization import Visualizer
    >>> viz = Visualizer(style='seaborn-v0_8-darkgrid')
    >>> viz.plot_equity_curve(result, title='BTC Strategy', save_path='equity.png')
"""

from visualization._base import VisualizerBase
from visualization.equity import EquityPlotMixin
from visualization.price_signals import PriceSignalsMixin
from visualization.monthly import MonthlyPlotMixin
from visualization.comparison import ComparisonMixin
from visualization.report import ReportMixin


class Visualizer(
    EquityPlotMixin,
    PriceSignalsMixin,
    MonthlyPlotMixin,
    ComparisonMixin,
    ReportMixin,
    VisualizerBase,
):
    """
    Backtest Result Visualizer

    Composes plotting methods via mixins, providing a single Visualizer interface.

    Args:
        style: Matplotlib style name, defaults to 'seaborn-v0_8'
        fig_dpi: Chart save resolution, defaults to 300

    Usage:
        >>> viz = Visualizer()
        >>> viz.plot_equity_curve(result, title='My Strategy')
        >>> viz.create_full_report(result, df, 'MultiFactor', 'BTC')
    """
    pass
