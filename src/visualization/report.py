"""
报告生成模块

生成完整的回测报告，包含多个图表。
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


class ReportMixin:
    """报告生成方法"""

    def create_full_report(
        self,
        result: object,
        df,
        strategy_name: str,
        coin: str,
        output_dir: str = "results",
    ) -> List[str]:
        """
        生成完整报告（包含所有图表）

        依次生成权益曲线、价格信号图和月度收益热力图。

        Args:
            result: BacktestResult 对象
            df: 价格数据
            strategy_name: 策略名称
            coin: 币种
            output_dir: 输出目录

        Returns:
            生成的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = f"{output_dir}/{strategy_name}_{coin}"
        generated_files = []

        # 1. 权益曲线
        fig1 = self.plot_equity_curve(
            result,
            title=f"{strategy_name} - {coin} Backtest",
            save_path=f"{base_name}_equity.png",
            show_plot=False,
        )
        generated_files.append(f"{base_name}_equity.png")
        plt.close(fig1)

        # 2. 价格与信号
        fig2 = self.plot_price_with_signals(
            df, result,
            title=f"{coin} Price Chart with Signals",
            save_path=f"{base_name}_signals.png",
            show_plot=False,
        )
        generated_files.append(f"{base_name}_signals.png")
        plt.close(fig2)

        # 3. 月度收益
        fig3 = self.plot_monthly_returns(
            result,
            save_path=f"{base_name}_monthly.png",
            show_plot=False,
        )
        if fig3:
            generated_files.append(f"{base_name}_monthly.png")
            plt.close(fig3)

        import logging
        logging.getLogger(__name__).info(
            f"✅ 完整报告已生成: {output_dir}/\n   文件: {', '.join(generated_files)}"
        )

        return generated_files

    def create_comparison_report(
        self,
        results: Dict[str, object],
        coin: str,
        output_dir: str = "results",
    ) -> List[str]:
        """
        生成策略对比报告

        包含多个图表展示策略对比结果:
        - 核心指标对比（2x3 布局）
        - 策略排名（按夏普比率）
        - 权益曲线对比（Top 5）

        Args:
            results: Dict[str, BacktestResult] - 策略结果字典
            coin: 币种
            output_dir: 输出目录

        Returns:
            生成的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 1. 核心指标对比
        fig1 = self.plot_metrics_comparison(
            results,
            save_path=f"{output_dir}/comparison_metrics_{coin}.png",
            show_plot=False,
        )
        generated_files.append(f"{output_dir}/comparison_metrics_{coin}.png")
        plt.close(fig1)

        # 2. 策略排名
        fig2 = self.plot_strategy_ranking(
            results,
            metric="sharpe_ratio",
            save_path=f"{output_dir}/comparison_ranking_{coin}.png",
            show_plot=False,
        )
        generated_files.append(f"{output_dir}/comparison_ranking_{coin}.png")
        plt.close(fig2)

        # 3. 权益曲线对比
        fig3 = self.plot_equity_comparison(
            results,
            save_path=f"{output_dir}/comparison_equity_{coin}.png",
            show_plot=False,
            top_n=5,
        )
        generated_files.append(f"{output_dir}/comparison_equity_{coin}.png")
        plt.close(fig3)

        import logging
        logging.getLogger(__name__).info(f"✅ 对比报告已生成: {output_dir}/")

        return generated_files
