"""
Monthly Returns Heatmap

Aggregates daily returns into monthly returns and displays monthly performance as a heatmap.
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
    """Monthly returns plotting methods"""

    def plot_monthly_returns(
        self,
        result: object,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot monthly returns heatmap

        Aggregates daily returns into monthly returns and displays monthly performance as a heatmap.
        Red indicates losses, green indicates profits, color intensity indicates magnitude.

        Args:
            result: BacktestResult object containing daily_returns
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)

        Returns:
            matplotlib Figure object, or None if no data is available
        """
        if result.daily_returns is None or len(result.daily_returns) == 0:
            import logging
            logging.getLogger(__name__).warning("No daily returns data, cannot generate monthly heatmap")
            return None

        # Calculate monthly returns
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

        # Use diverging colormap, center is white
        im = ax.imshow(
            pivot_table.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=MONTHLY_RETURN_VMIN,
            vmax=MONTHLY_RETURN_VMAX,
        )

        # Set axis labels
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        # Use calendar module instead of hardcoded month names
        ax.set_yticklabels([calendar.month_abbr[i] for i in pivot_table.index])

        # Add numeric values to each cell
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    # Use white text when absolute value exceeds threshold, otherwise black
                    color = "white" if abs(value) > COLOR_TEXT_THRESHOLD else "black"
                    ax.text(
                        j, i, f"{value:.1f}%",
                        ha="center", va="center",
                        color=color, fontsize=9,
                    )

        ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Month", fontsize=12)

        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Return (%)", rotation=270, labelpad=15)

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig
