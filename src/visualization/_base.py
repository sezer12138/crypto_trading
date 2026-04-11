"""
Visualizer Base Class

Provides core initialization logic and chart saving functionality for the Visualizer.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from visualization._constants import DEFAULT_FIG_DPI, DEFAULT_STYLE

logger = logging.getLogger(__name__)

# Set font and plot style
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class VisualizerBase:
    """
    Visualizer Base Class

    Provides initialization and chart saving functionality, inherited by plotting Mixins.

    Args:
        style: Matplotlib style name, defaults to 'seaborn-v0_8'
        fig_dpi: Chart save resolution, defaults to 300

    Attributes:
        style: Current matplotlib style in use
        fig_dpi: Chart resolution setting
    """

    def __init__(self, style: str = DEFAULT_STYLE, fig_dpi: int = DEFAULT_FIG_DPI):
        """
        Initialize the visualizer

        Args:
            style: Matplotlib style name
            fig_dpi: Chart save resolution
        """
        self.fig_dpi = fig_dpi

        # Try to set style, fall back to default on failure
        try:
            plt.style.use(style)
            self.style = style
        except OSError:
            plt.style.use("default")
            self.style = "default"
            logger.warning(f"Style '{style}' unavailable, using default")

    def _save_figure(self, fig: plt.Figure, save_path: str = None) -> None:
        """
        Save chart to file (internal helper method)

        Saves the chart at the configured resolution to the specified path and logs the action.

        Args:
            fig: matplotlib Figure object
            save_path: Save path, if None the chart is not saved
        """
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 Chart saved: {save_path}")
