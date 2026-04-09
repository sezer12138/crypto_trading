"""
权益曲线可视化

绘制权益曲线、累计收益率和回撤分析的三子图布局。
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from visualization._constants import DEFAULT_EQUITY_FIGSIZE


class EquityPlotMixin:
    """权益曲线绘图方法"""

    def plot_equity_curve(
        self,
        result: object,
        title: str = "Strategy Backtest Result",
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        绘制权益曲线和收益曲线

        生成三子图布局:
        1. 权益曲线 - 展示资金变化
        2. 累计收益率 - 百分比收益
        3. 回撤曲线 - 最大回撤分析

        Args:
            result: BacktestResult 对象，包含 equity_curve 和 metrics
            title: 图表主标题
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）

        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(3, 1, figsize=DEFAULT_EQUITY_FIGSIZE)

        # 1. 权益曲线
        ax1 = axes[0]
        ax1.plot(
            result.equity_curve.index,
            result.equity_curve.values,
            label="Portfolio Value",
            color="#2E86AB",
            linewidth=2,
        )
        ax1.axhline(
            y=result.equity_curve.iloc[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

        # 2. 累计收益率
        ax2 = axes[1]
        ax2.plot(
            result.cumulative_returns.index,
            result.cumulative_returns.values,
            label="Cumulative Return (%)",
            color="#A23B72",
            linewidth=2,
        )
        ax2.fill_between(
            result.cumulative_returns.index,
            0,
            result.cumulative_returns.values,
            alpha=0.3,
            color="#A23B72",
        )
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 3. 回撤曲线
        ax3 = axes[2]
        cummax = result.equity_curve.cummax()
        drawdown = (result.equity_curve - cummax) / cummax * 100

        ax3.fill_between(
            drawdown.index, drawdown.values, 0,
            color="#E74C3C", alpha=0.5, label="Drawdown"
        )
        ax3.plot(drawdown.index, drawdown.values, color="#C0392B", linewidth=1)
        ax3.set_ylabel("Drawdown (%)", fontsize=12)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.legend(loc="lower left")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig
