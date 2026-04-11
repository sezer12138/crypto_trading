"""
Price and Trading Signals Visualization

Plots price trends with buy/sell signal annotations and volume display.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from visualization._constants import DEFAULT_PRICE_FIGSIZE, PRICE_HEIGHT_RATIO


class PriceSignalsMixin:
    """Price signal plotting methods"""

    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        result: object,
        title: str = "Price Chart with Trading Signals",
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot price chart and trading signals

        Displays price trends with buy/sell signal annotations and volume.

        Args:
            df: DataFrame containing OHLCV data
            result: BacktestResult object containing trade records
            title: Chart title
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)

        Returns:
            matplotlib Figure object

        Note:
            Buy signals are marked with green upward triangles,
            sell signals are marked with red downward triangles.
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=DEFAULT_PRICE_FIGSIZE,
            gridspec_kw={"height_ratios": [PRICE_HEIGHT_RATIO, 1]}
        )

        # Price chart
        ax1.plot(df.index, df["close"], label="Price", color="#2E86AB", linewidth=1.5)

        # Mark buy and sell points
        buy_trades = [t for t in result.trades if t.action == "buy"]
        sell_trades = [t for t in result.trades if t.action == "sell"]

        if buy_trades:
            buy_times = [t.timestamp for t in buy_trades]
            buy_prices = [t.price for t in buy_trades]
            ax1.scatter(
                buy_times, buy_prices, marker="^", color="green",
                s=100, label="Buy", zorder=5, edgecolors='black', linewidths=0.5
            )

        if sell_trades:
            sell_times = [t.timestamp for t in sell_trades]
            sell_prices = [t.price for t in sell_trades]
            ax1.scatter(
                sell_times, sell_prices, marker="v", color="red",
                s=100, label="Sell", zorder=5, edgecolors='black', linewidths=0.5
            )

        # Show moving averages (if present)
        if "ma_short" in df.columns and "ma_long" in df.columns:
            ax1.plot(
                df.index, df["ma_short"],
                label="MA Short", color="orange", alpha=0.7, linewidth=1
            )
            ax1.plot(
                df.index, df["ma_long"],
                label="MA Long", color="purple", alpha=0.7, linewidth=1
            )

        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Volume chart (using vectorized operations instead of loops)
        if "volume" in df.columns:
            colors = ["green" if c >= o else "red"
                      for c, o in zip(df["close"], df["open"])]
            ax2.bar(df.index, df["volume"], color=colors, alpha=0.6, width=0.8)
            ax2.set_ylabel("Volume", fontsize=12)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig
