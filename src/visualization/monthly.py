"""
月度收益热力图

将日收益聚合成月度收益，以热力图形式展示各月表现。
"""

import calendar
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._constants import (
    DEFAULT_MONTHLY_FIGSIZE,
    MONTHLY_RETURN_VMIN, MONTHLY_RETURN_VMAX,
    COLOR_TEXT_THRESHOLD,
)


class MonthlyPlotMixin:
    """月度收益绘图方法"""

    def plot_monthly_returns(
        self,
        result: object,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> Optional[plt.Figure]:
        """
        绘制月度收益热力图

        将日收益聚合成月度收益，以热力图形式展示各月表现。
        红色表示亏损，绿色表示盈利，颜色深浅表示幅度。

        Args:
            result: BacktestResult 对象，包含 daily_returns
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）

        Returns:
            matplotlib Figure 对象，如果没有数据则返回 None
        """
        if result.daily_returns is None or len(result.daily_returns) == 0:
            import logging
            logging.getLogger(__name__).warning("没有日收益数据，无法生成月度热力图")
            return None

        # 计算月度收益
        monthly_returns = (
            result.daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
        )

        monthly_df = pd.DataFrame({
            "Year": monthly_returns.index.year,
            "Month": monthly_returns.index.month,
            "Return": monthly_returns.values,
        })

        pivot_table = monthly_df.pivot(index="Month", columns="Year", values="Return")

        fig, ax = plt.subplots(figsize=DEFAULT_MONTHLY_FIGSIZE)

        # 使用 diverging colormap，中心为白色
        im = ax.imshow(
            pivot_table.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=MONTHLY_RETURN_VMIN,
            vmax=MONTHLY_RETURN_VMAX,
        )

        # 设置坐标轴标签
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        # 使用 calendar 模块替代硬编码月份名
        ax.set_yticklabels([calendar.month_abbr[i] for i in pivot_table.index])

        # 在每个单元格添加数值
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    # 绝对值大于阈值时使用白色文字，否则使用黑色
                    color = "white" if abs(value) > COLOR_TEXT_THRESHOLD else "black"
                    ax.text(
                        j, i, f"{value:.1f}%",
                        ha="center", va="center",
                        color=color, fontsize=9,
                    )

        ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Month", fontsize=12)

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Return (%)", rotation=270, labelpad=15)

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig
