"""
Strategy Comparison Visualization

Provides multi-strategy performance comparison charts, including metrics comparison, ranking, and equity curve comparison.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization._constants import (
    DEFAULT_COMPARISON_FIGSIZE,
    DEFAULT_RANKING_FIGSIZE_BASE,
    DEFAULT_EQUITY_COMPARISON_FIGSIZE,
    DEFAULT_TRADE_DETAILS_FIGSIZE,
    DEFAULT_TOP_N_COMPARISON,
    DEFAULT_TOP_N_EQUITY,
)


class ComparisonMixin:
    """Strategy comparison plotting methods"""

    def plot_metrics_comparison(
        self,
        results: Dict[str, object],
        save_path: Optional[str] = None,
        show_plot: bool = True,
        top_n: int = DEFAULT_TOP_N_COMPARISON,
    ) -> plt.Figure:
        """
        Compare multi-strategy backtest metrics - optimized layout version

        Optimized for comparing 11 strategies:
        - 2x3 subplot layout displaying core metrics
        - Grouped bar charts to avoid crowding

        Args:
            results: Dict[str, BacktestResult] - strategy name to result mapping
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)
            top_n: Number of top strategies to display in detail (default 6)

        Returns:
            matplotlib Figure object
        """
        # Define metrics
        metrics_config = [
            ("total_return_pct", "Total Return (%)", "#2E86AB"),
            ("annual_return_pct", "Annual Return (%)", "#A23B72"),
            ("sharpe_ratio", "Sharpe Ratio", "#F18F01"),
            ("max_drawdown_pct", "Max Drawdown (%)", "#C73E1D"),
            ("win_rate_pct", "Win Rate (%)", "#3B1F2B"),
            ("total_trades", "Total Trades", "#6A994E"),
        ]

        strategies = list(results.keys())
        n_strategies = len(strategies)

        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=DEFAULT_COMPARISON_FIGSIZE)
        axes = axes.flatten()

        # Create subplot for each metric
        for idx, (metric, label, color) in enumerate(metrics_config):
            ax = axes[idx]
            values = [results[s].metrics.get(metric, 0) for s in strategies]

            # For all metrics, sort descending so index [0] = best value
            # (max_drawdown_pct is negative; closest to 0 = highest = best)
            sorted_indices = np.argsort(values)[::-1]
            colors = [
                "#27ae60" if i == sorted_indices[0]
                else "#3498db" if i == sorted_indices[1]
                else "#95a5a6"
                for i in range(len(values))
            ]

            bars = ax.bar(
                range(n_strategies), values,
                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5
            )

            ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
            ax.set_xticks(range(n_strategies))
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.01 * max(abs(v) for v in values) if values else 0.1
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (offset if height >= 0 else -offset),
                    f"{val:.1f}",
                    ha="center", va=va, fontsize=8, fontweight='bold',
                )

            # Add zero line (for metrics that may have negative values)
            if metric in ["total_return_pct", "annual_return_pct", "max_drawdown_pct"]:
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.suptitle(
            f"Strategy Performance Comparison (n={n_strategies})",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig

    def plot_strategy_ranking(
        self,
        results: Dict[str, object],
        metric: str = "sharpe_ratio",
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot strategy ranking chart - horizontal bar chart

        Displays strategy ranking by specified metric using a horizontal bar chart,
        better suited for showing multiple strategies.

        Args:
            results: Dict[str, BacktestResult] - strategy results
            metric: Sorting metric (default 'sharpe_ratio')
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)

        Returns:
            matplotlib Figure object
        """
        strategies = list(results.keys())
        values = [results[s].metrics.get(metric, 0) for s in strategies]

        # Sort
        sorted_pairs = sorted(zip(strategies, values), key=lambda x: x[1], reverse=True)
        sorted_strategies, sorted_values = zip(*sorted_pairs)

        # Color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(strategies)))

        fig, ax = plt.subplots(
            figsize=(DEFAULT_RANKING_FIGSIZE_BASE[0],
                     max(DEFAULT_RANKING_FIGSIZE_BASE[1], len(strategies) * 0.5)),
        )

        bars = ax.barh(range(len(strategies)), sorted_values, color=colors, alpha=0.8)

        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(sorted_strategies)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(
            f"Strategy Ranking by {metric.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, sorted_values):
            width = bar.get_width()
            ax.text(
                width + 0.01 * max(abs(v) for v in sorted_values) if sorted_values else 0.1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
                ha='left', va='center', fontsize=10, fontweight='bold',
            )

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig

    def plot_equity_comparison(
        self,
        results: Dict[str, object],
        save_path: Optional[str] = None,
        show_plot: bool = True,
        top_n: int = DEFAULT_TOP_N_EQUITY,
    ) -> plt.Figure:
        """
        Plot multi-strategy equity curve comparison

        Only shows equity curves for the top_n best-performing strategies to avoid chart crowding.

        Args:
            results: Dict[str, BacktestResult] - strategy results
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)
            top_n: Number of top strategies to display (default 5)

        Returns:
            matplotlib Figure object
        """
        # Sort by total return
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].metrics.get("total_return_pct", 0),
            reverse=True,
        )
        top_strategies = sorted_results[:top_n]

        fig, ax = plt.subplots(figsize=DEFAULT_EQUITY_COMPARISON_FIGSIZE)

        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        for i, (name, result) in enumerate(top_strategies):
            # Normalize to initial capital
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            ax.plot(
                normalized.index, normalized.values,
                label=f"{name} ({result.metrics.get('total_return_pct', 0):.1f}%)",
                color=colors[i], linewidth=2, alpha=0.8,
            )

        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Initial (100%)')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
        ax.set_title(
            f"Top {top_n} Strategy Equity Curves Comparison",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig

    def plot_trade_details_comparison(
        self,
        results: Dict[str, object],
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot trade details for all strategies in a subplot grid

        Each subplot is a scatter plot showing buy/sell trades:
        - X-axis: trade time
        - Y-axis: trade price
        - Green triangle-up markers = buy, Red triangle-down markers = sell
        - Marker size proportional to trade value
        - Alpha adapts to trade count for readability

        Args:
            results: Dict[str, BacktestResult] - strategy results
            save_path: Save path (optional)
            show_plot: Whether to display the chart (default True)

        Returns:
            matplotlib Figure object
        """
        strategies = sorted(results.keys())
        n = len(strategies)
        if n == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No strategies to display", ha="center", va="center")
            return fig

        ncols = 3
        nrows = int(np.ceil(n / ncols))

        # Dynamic figure size: wider subplots, taller for more rows
        fig_w = 18
        fig_h = max(20, nrows * 5)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten() if n > 1 else np.array([axes])

        for idx, name in enumerate(strategies):
            ax = axes[idx]
            result = results[name]
            trades = result.trades

            if not trades:
                ax.text(
                    0.5, 0.5, "No trades",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#888",
                )
                ax.set_title(f"{name} (0 trades)", fontsize=11, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            buy_times, buy_prices, buy_vals = [], [], []
            sell_times, sell_prices, sell_vals = [], [], []

            for t in trades:
                if t.action == "buy":
                    buy_times.append(t.timestamp)
                    buy_prices.append(t.price)
                    buy_vals.append(t.value)
                elif t.action == "sell":
                    sell_times.append(t.timestamp)
                    sell_prices.append(t.price)
                    sell_vals.append(t.value)

            n_trades = len(buy_times) + len(sell_times)

            # Adaptive alpha: more trades -> lower alpha for readability
            alpha = max(0.05, 0.8 - (n_trades / 3000))

            # Normalize marker sizes proportional to trade value
            all_vals = buy_vals + sell_vals
            max_val = max(all_vals) if all_vals else 1
            min_size, max_size = 15, 150

            def scale_sizes(vals):
                if not vals or max_val == 0:
                    return [min_size] * len(vals)
                return [min_size + (v / max_val) * (max_size - min_size) for v in vals]

            buy_sizes = scale_sizes(buy_vals)
            sell_sizes = scale_sizes(sell_vals)

            # Plot buy markers (green triangle-up)
            if buy_times:
                ax.scatter(
                    buy_times, buy_prices,
                    s=buy_sizes, c="#00cc66", marker="^",
                    alpha=alpha, edgecolors="#006633", linewidths=0.5,
                    label=f"Buy ({len(buy_times)})", zorder=3,
                )
            # Plot sell markers (red triangle-down)
            if sell_times:
                ax.scatter(
                    sell_times, sell_prices,
                    s=sell_sizes, c="#ff4444", marker="v",
                    alpha=alpha, edgecolors="#990000", linewidths=0.5,
                    label=f"Sell ({len(sell_times)})", zorder=3,
                )

            ax.set_title(
                f"{name} ({len(buy_times)} buys / {len(sell_times)} sells)",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylabel("Price ($)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "Trade Details by Strategy (marker size = trade value)",
            fontsize=16, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.autofmt_xdate()
        self._save_figure(fig, save_path)

        if show_plot:
            plt.show()
            plt.close(fig)

        return fig
