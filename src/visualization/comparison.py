"""
策略对比可视化

提供多策略性能对比的图表，包括指标对比、排名和权益曲线对比。
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from visualization._constants import (
    DEFAULT_COMPARISON_FIGSIZE,
    DEFAULT_RANKING_FIGSIZE_BASE,
    DEFAULT_EQUITY_COMPARISON_FIGSIZE,
    DEFAULT_TOP_N_COMPARISON,
    DEFAULT_TOP_N_EQUITY,
)


class ComparisonMixin:
    """策略对比绘图方法"""

    def plot_metrics_comparison(
        self,
        results: Dict[str, object],
        save_path: Optional[str] = None,
        show_plot: bool = True,
        top_n: int = DEFAULT_TOP_N_COMPARISON,
    ) -> plt.Figure:
        """
        比较多策略的回测指标 - 优化布局版本

        针对 11 个策略的对比进行优化:
        - 2x3 子图布局展示核心指标
        - 分组柱状图避免拥挤

        Args:
            results: Dict[str, BacktestResult] - 策略名称到结果的映射
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）
            top_n: 详细对比显示前 N 个策略（默认 6）

        Returns:
            matplotlib Figure 对象
        """
        # 定义指标
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

        # 创建 2x3 子图布局
        fig, axes = plt.subplots(2, 3, figsize=DEFAULT_COMPARISON_FIGSIZE)
        axes = axes.flatten()

        # 为每个指标创建子图
        for idx, (metric, label, color) in enumerate(metrics_config):
            ax = axes[idx]
            values = [results[s].metrics.get(metric, 0) for s in strategies]

            # 根据数值排序获取颜色映射
            sorted_indices = np.argsort(values)[::-1] if metric != "max_drawdown_pct" else np.argsort(values)
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

            # 添加数值标签
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

            # 添加零线（针对可能有负值的指标）
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
        绘制策略排名图 - 水平条形图

        用水平条形图展示策略按指定指标的排名，更适合显示多个策略。

        Args:
            results: Dict[str, BacktestResult] - 策略结果
            metric: 排序指标（默认 'sharpe_ratio'）
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）

        Returns:
            matplotlib Figure 对象
        """
        strategies = list(results.keys())
        values = [results[s].metrics.get(metric, 0) for s in strategies]

        # 排序
        sorted_pairs = sorted(zip(strategies, values), key=lambda x: x[1], reverse=True)
        sorted_strategies, sorted_values = zip(*sorted_pairs)

        # 颜色渐变
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

        # 添加数值标签
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
        绘制多策略权益曲线对比

        只显示表现最好的 top_n 个策略的权益曲线，避免图表过于拥挤。

        Args:
            results: Dict[str, BacktestResult] - 策略结果
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）
            top_n: 显示前 N 个策略（默认 5）

        Returns:
            matplotlib Figure 对象
        """
        # 按总收益排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].metrics.get("total_return_pct", 0),
            reverse=True,
        )
        top_strategies = sorted_results[:top_n]

        fig, ax = plt.subplots(figsize=DEFAULT_EQUITY_COMPARISON_FIGSIZE)

        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        for i, (name, result) in enumerate(top_strategies):
            # 归一化到初始资金
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
