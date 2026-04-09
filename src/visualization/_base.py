"""
可视化器基类

提供 Visualizer 的核心初始化逻辑和图表保存功能。
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from visualization._constants import DEFAULT_FIG_DPI, DEFAULT_STYLE

logger = logging.getLogger(__name__)

# 设置中文字体和绘图样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class VisualizerBase:
    """
    可视化器基类

    提供初始化和图表保存功能，由各绘图 Mixin 继承。

    Args:
        style: Matplotlib 样式名称，默认为 'seaborn-v0_8'
        fig_dpi: 图表保存分辨率，默认 300

    Attributes:
        style: 当前使用的 matplotlib 样式
        fig_dpi: 图表分辨率设置
    """

    def __init__(self, style: str = DEFAULT_STYLE, fig_dpi: int = DEFAULT_FIG_DPI):
        """
        初始化可视化器

        Args:
            style: Matplotlib 样式名称
            fig_dpi: 图表保存分辨率
        """
        self.fig_dpi = fig_dpi

        # 尝试设置样式，失败则使用默认
        try:
            plt.style.use(style)
            self.style = style
        except OSError:
            plt.style.use("default")
            self.style = "default"
            logger.warning(f"样式 '{style}' 不可用，使用默认样式")

    def _save_figure(self, fig: plt.Figure, save_path: str = None) -> None:
        """
        保存图表到文件 (内部辅助方法)

        将图表以设定分辨率保存到指定路径，并在日志中记录。

        Args:
            fig: matplotlib Figure 对象
            save_path: 保存路径，为 None 时不保存
        """
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 图表已保存: {save_path}")
