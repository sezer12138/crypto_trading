"""
Equity Curve Visualization

Plots equity curve, cumulative returns, and drawdown analysis in a three-subplot layout.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from visualization._constants import DEFAULT_EQUITY_FIGSIZE


class EquityPlotMixin:
    """Equity curve plotting methods"""

    def plot_equity_curve(
        self,
        result: object,
        title: str = "Strategy Backtest Result",
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot equity curve and returns

        Generates a three-subplot layout:
        1. Equity curve - shows capital changes
        2. Cumulative returns - percentage returns
        3. Drawdown curve - maximum drawdown analysis

        Args:
            result: BacktestResult object containing equity_curve and metrics
            title: Chart main title
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=DEFAULT_EQUITY_FIGSIZE)

        # 1. Equity curve
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

        # 2. Cumulative returns
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

        # 3. Drawdown curve
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
            plt.close(fig)

        return fig
