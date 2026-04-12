"""
Report Generation Module

Generates complete backtest reports containing multiple charts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class ReportMixin:
    """Report generation methods"""

    def create_full_report(
        self,
        result: Any,
        df: pd.DataFrame,
        strategy_name: str,
        coin: str,
        output_dir: str = "results",
        days: int = 730,
        interval: str = "1h",
    ) -> List[str]:
        """
        Generate full report (containing all charts)

        Generates equity curve, price signal chart, and monthly returns heatmap in sequence.

        Args:
            result: BacktestResult object
            df: Price data
            strategy_name: Strategy name
            coin: Coin symbol
            output_dir: Output directory
            days: Backtest days
            interval: Time interval

        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # File naming format: {strategy}_{coin}_{days}d_{interval}_{timestamp}_{type}.png
        base_name = f"{output_dir}/{strategy_name}_{coin}_{days}d_{interval}_{timestamp}"
        generated_files = []

        # 1. Equity curve
        fig1 = self.plot_equity_curve(
            result,
            title=f"{strategy_name} - {coin} Backtest",
            save_path=f"{base_name}_equity.png",
            show_plot=False,
        )
        generated_files.append(f"{base_name}_equity.png")
        plt.close(fig1)

        # 2. Price and signals
        fig2 = self.plot_price_with_signals(
            df, result,
            title=f"{coin} Price Chart with Signals",
            save_path=f"{base_name}_signals.png",
            show_plot=False,
        )
        generated_files.append(f"{base_name}_signals.png")
        plt.close(fig2)

        # 3. Monthly returns
        fig3 = self.plot_monthly_returns(
            result,
            save_path=f"{base_name}_monthly.png",
            show_plot=False,
        )
        if fig3:
            generated_files.append(f"{base_name}_monthly.png")
            plt.close(fig3)

        logger.info(
            f"✅ Full report generated: {output_dir}/\n   Files: {', '.join(generated_files)}"
        )

        return generated_files

    def create_comparison_report(
        self,
        results: Dict[str, Any],
        coin: str,
        output_dir: str = "results",
        days: int = 730,
        interval: str = "1h",
    ) -> List[str]:
        """
        Generate strategy comparison report

        Contains multiple charts displaying strategy comparison results:
        - Core metrics comparison (2x3 layout)
        - Strategy ranking (by Sharpe ratio)
        - Equity curve comparison (Top 5)

        Args:
            results: Dict[str, BacktestResult] - strategy results dictionary
            coin: Coin symbol
            output_dir: Output directory
            days: Backtest days
            interval: Time interval

        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # File naming format: comparison_{type}_{coin}_{days}d_{interval}_{timestamp}.png
        generated_files = []

        # 1. Core metrics comparison
        fig1 = self.plot_metrics_comparison(
            results,
            save_path=f"{output_dir}/comparison_metrics_{coin}_{days}d_{interval}_{timestamp}.png",
            show_plot=False,
        )
        generated_files.append(f"{output_dir}/comparison_metrics_{coin}_{days}d_{interval}_{timestamp}.png")
        plt.close(fig1)

        # 2. Strategy ranking
        fig2 = self.plot_strategy_ranking(
            results,
            metric="sharpe_ratio",
            save_path=f"{output_dir}/comparison_ranking_{coin}_{days}d_{interval}_{timestamp}.png",
            show_plot=False,
        )
        generated_files.append(f"{output_dir}/comparison_ranking_{coin}_{days}d_{interval}_{timestamp}.png")
        plt.close(fig2)

        # 3. Equity curve comparison
        fig3 = self.plot_equity_comparison(
            results,
            save_path=f"{output_dir}/comparison_equity_{coin}_{days}d_{interval}_{timestamp}.png",
            show_plot=False,
            top_n=5,
        )
        generated_files.append(f"{output_dir}/comparison_equity_{coin}_{days}d_{interval}_{timestamp}.png")
        plt.close(fig3)

        # 4. Trade details comparison
        fig4 = self.plot_trade_details_comparison(
            results,
            save_path=f"{output_dir}/comparison_trade_details_{coin}_{days}d_{interval}_{timestamp}.png",
            show_plot=False,
        )
        generated_files.append(f"{output_dir}/comparison_trade_details_{coin}_{days}d_{interval}_{timestamp}.png")
        plt.close(fig4)

        logger.info(f"✅ Comparison report generated: {output_dir}/")

        return generated_files
