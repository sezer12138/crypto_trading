"""
可视化模块

提供回测结果的图表生成功能，包括:
- 权益曲线与回撤分析
- 价格图表与交易信号
- 月度收益热力图
- 多策略性能对比

使用示例:
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
    回测结果可视化器

    通过混入组合各类型绘图方法，提供单一 Visualizer 接口。

    Args:
        style: Matplotlib 样式名称，默认为 'seaborn-v0_8'
        fig_dpi: 图表保存分辨率，默认 300

    使用示例:
        >>> viz = Visualizer()
        >>> viz.plot_equity_curve(result, title='My Strategy')
        >>> viz.create_full_report(result, df, 'MultiFactor', 'BTC')
    """
    pass
