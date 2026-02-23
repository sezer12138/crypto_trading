"""
Visualization Module
回测结果可视化模块 - 提供专业的收益曲线、交易信号和策略对比图表

Features:
    - 权益曲线与回撤分析
    - 价格图表与交易信号
    - 月度收益热力图
    - 多策略性能对比（优化布局）
    - 综合报告生成

Example:
    >>> from visualization import Visualizer
    >>> viz = Visualizer(style='seaborn-v0_8-darkgrid')
    >>> viz.plot_equity_curve(result, title='My Strategy', save_path='equity.png')
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 设置中文字体和绘图样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """
    回测结果可视化器
    
    提供多种图表类型用于分析策略表现，包括：
    - 权益曲线与回撤分析
    - 价格图表与交易信号
    - 月度收益热力图
    - 多策略性能对比
    
    Args:
        style: Matplotlib 样式名称，默认为 'seaborn-v0_8'
        fig_dpi: 图表保存分辨率，默认 300
        
    Attributes:
        style: 当前使用的 matplotlib 样式
        fig_dpi: 图表分辨率设置
    """
    
    def __init__(self, style: str = "seaborn-v0_8", fig_dpi: int = 300):
        """
        初始化可视化器
        
        Args:
            style: Matplotlib 样式名称
            fig_dpi: 图表保存分辨率
        """
        self.fig_dpi = fig_dpi
        
        # 尝试设置样式，失败则使用默认
        try:
            plt.style.use(style)
            self.style = style
        except OSError:
            plt.style.use("default")
            self.style = "default"
            logger.warning(f"样式 '{style}' 不可用，使用默认样式")
    
    def plot_equity_curve(
        self, 
        result: object, 
        title: str = "Strategy Backtest Result", 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        绘制权益曲线和收益曲线
        
        生成三子图布局：
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
            
        Example:
            >>> viz = Visualizer()
            >>> fig = viz.plot_equity_curve(result, title='BTC Strategy')
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 权益曲线已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        result: object,
        title: str = "Price Chart with Trading Signals",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        绘制价格图和交易信号
        
        展示价格走势并标注买卖信号点，同时显示成交量。
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            result: BacktestResult 对象，包含 trades 记录
            title: 图表标题
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）
            
        Returns:
            matplotlib Figure 对象
            
        Note:
            买入信号用绿色三角形标记，卖出信号用红色倒三角标记
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        
        # 价格图
        ax1.plot(df.index, df["close"], label="Price", color="#2E86AB", linewidth=1.5)
        
        # 标记买卖点
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
        
        # 显示均线（如果存在）
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
        
        # 成交量图
        if "volume" in df.columns:
            colors = [
                "green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red"
                for i in range(len(df))
            ]
            ax2.bar(df.index, df["volume"], color=colors, alpha=0.6, width=0.8)
            ax2.set_ylabel("Volume", fontsize=12)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 价格信号图已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_monthly_returns(
        self, 
        result: object, 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        绘制月度收益热力图
        
        将日收益聚合成月度收益，以热力图形式展示各月表现。
        
        Args:
            result: BacktestResult 对象，包含 daily_returns
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）
            
        Returns:
            matplotlib Figure 对象，如果没有数据则返回 None
        """
        if result.daily_returns is None or len(result.daily_returns) == 0:
            logger.warning("没有日收益数据，无法生成月度热力图")
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
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 使用 diverging colormap，中心为白色
        im = ax.imshow(
            pivot_table.values, 
            cmap="RdYlGn", 
            aspect="auto", 
            vmin=-20, 
            vmax=20
        )
        
        # 设置坐标轴标签
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        # 只使用实际存在的月份标签
        ax.set_yticklabels([month_labels[i-1] for i in pivot_table.index])
        
        # 在每个单元格添加数值
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    color = "white" if abs(value) > 10 else "black"
                    ax.text(
                        j, i, f"{value:.1f}%",
                        ha="center", va="center", 
                        color=color, fontsize=9
                    )
        
        ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Month", fontsize=12)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Return (%)", rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 月度收益图已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_metrics_comparison(
        self, 
        results: Dict[str, object], 
        save_path: Optional[str] = None,
        show_plot: bool = True,
        top_n: int = 6
    ) -> plt.Figure:
        """
        比较多策略的回测指标 - 优化布局版本
        
        针对11个策略的对比进行优化：
        - 2x3 子图布局展示核心指标
        - 单独展示 Top N 策略的详细对比
        - 分组柱状图避免拥挤
        
        Args:
            results: Dict[str, BacktestResult] - 策略名称到结果的映射
            save_path: 保存路径（可选）
            show_plot: 是否显示图表（默认 True）
            top_n: 详细对比显示前 N 个策略（默认 6）
            
        Returns:
            matplotlib Figure 对象
            
        Example:
            >>> results = {
            ...     'ma_cross': result1,
            ...     'rsi': result2,
            ...     'multi_factor': result3
            ... }
            >>> viz.plot_metrics_comparison(results, save_path='comparison.png')
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # 为每个指标创建子图
        for idx, (metric, label, color) in enumerate(metrics_config):
            ax = axes[idx]
            values = [results[s].metrics.get(metric, 0) for s in strategies]
            
            # 根据数值排序获取颜色映射
            sorted_indices = np.argsort(values)[::-1] if metric != "max_drawdown_pct" else np.argsort(values)
            colors = ["#27ae60" if i == sorted_indices[0] else "#3498db" if i == sorted_indices[1] else "#95a5a6" 
                     for i in range(len(values))]
            
            bars = ax.bar(range(n_strategies), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
            ax.set_xticks(range(n_strategies))
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.01 * max(values) if max(values) != 0 else 0.1
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (offset if height >= 0 else -offset),
                    f"{val:.1f}",
                    ha="center", va=va, fontsize=8, fontweight='bold'
                )
            
            # 添加零线（针对可能有负值的指标）
            if metric in ["total_return_pct", "annual_return_pct", "max_drawdown_pct"]:
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.suptitle(
            f"Strategy Performance Comparison (n={n_strategies})", 
            fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 策略对比图已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_strategy_ranking(
        self,
        results: Dict[str, object],
        metric: str = "sharpe_ratio",
        save_path: Optional[str] = None,
        show_plot: bool = True
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
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(strategies) * 0.5)))
        
        bars = ax.barh(range(len(strategies)), sorted_values, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(sorted_strategies)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f"Strategy Ranking by {metric.replace('_', ' ').title()}", 
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            width = bar.get_width()
            ax.text(
                width + 0.01 * max(sorted_values),
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
                ha='left', va='center', fontsize=10, fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 策略排名图已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_equity_comparison(
        self,
        results: Dict[str, object],
        save_path: Optional[str] = None,
        show_plot: bool = True,
        top_n: int = 5
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
            reverse=True
        )
        top_strategies = sorted_results[:top_n]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))
        
        for i, (name, result) in enumerate(top_strategies):
            # 归一化到初始资金
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            ax.plot(
                normalized.index, normalized.values,
                label=f"{name} ({result.metrics.get('total_return_pct', 0):.1f}%)",
                color=colors[i], linewidth=2, alpha=0.8
            )
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Initial (100%)')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
        ax.set_title(f"Top {top_n} Strategy Equity Curves Comparison", 
                    fontsize=14, fontweight="bold")
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.fig_dpi, bbox_inches="tight")
            logger.info(f"💾 权益曲线对比图已保存: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_full_report(
        self, 
        result: object, 
        df: pd.DataFrame, 
        strategy_name: str, 
        coin: str, 
        output_dir: str = "results"
    ) -> List[str]:
        """
        生成完整报告（包含所有图表）
        
        Args:
            result: BacktestResult 对象
            df: 价格数据
            strategy_name: 策略名称
            coin: 币种
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
            
        Example:
            >>> viz = Visualizer()
            >>> files = viz.create_full_report(result, df, 'MultiFactor', 'BTC')
            >>> print(files)
            ['results/MultiFactor_BTC_equity.png', ...]
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
            show_plot=False
        )
        generated_files.append(f"{base_name}_equity.png")
        plt.close(fig1)
        
        # 2. 价格与信号
        fig2 = self.plot_price_with_signals(
            df, result,
            title=f"{coin} Price Chart with Signals",
            save_path=f"{base_name}_signals.png",
            show_plot=False
        )
        generated_files.append(f"{base_name}_signals.png")
        plt.close(fig2)
        
        # 3. 月度收益
        fig3 = self.plot_monthly_returns(
            result, 
            save_path=f"{base_name}_monthly.png",
            show_plot=False
        )
        if fig3:
            generated_files.append(f"{base_name}_monthly.png")
            plt.close(fig3)
        
        logger.info(f"✅ 完整报告已生成: {output_dir}/")
        logger.info(f"   文件: {', '.join(generated_files)}")
        
        return generated_files
    
    def create_comparison_report(
        self,
        results: Dict[str, object],
        coin: str,
        output_dir: str = "results"
    ) -> List[str]:
        """
        生成策略对比报告
        
        包含多个图表展示策略对比结果：
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
            show_plot=False
        )
        generated_files.append(f"{output_dir}/comparison_metrics_{coin}.png")
        plt.close(fig1)
        
        # 2. 策略排名
        fig2 = self.plot_strategy_ranking(
            results,
            metric="sharpe_ratio",
            save_path=f"{output_dir}/comparison_ranking_{coin}.png",
            show_plot=False
        )
        generated_files.append(f"{output_dir}/comparison_ranking_{coin}.png")
        plt.close(fig2)
        
        # 3. 权益曲线对比
        fig3 = self.plot_equity_comparison(
            results,
            save_path=f"{output_dir}/comparison_equity_{coin}.png",
            show_plot=False,
            top_n=5
        )
        generated_files.append(f"{output_dir}/comparison_equity_{coin}.png")
        plt.close(fig3)
        
        logger.info(f"✅ 对比报告已生成: {output_dir}/")
        return generated_files


if __name__ == "__main__":
    # 测试可视化
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    equity = 10000 + np.cumsum(np.random.randn(100) * 100)
    
    class MockResult:
        def __init__(self):
            self.equity_curve = pd.Series(equity, index=dates)
            self.cumulative_returns = (self.equity_curve / self.equity_curve.iloc[0] - 1) * 100
            self.daily_returns = self.equity_curve.pct_change().dropna()
            self.metrics = {"total_return_pct": 15.5, "sharpe_ratio": 1.2}
            self.trades = []
    
    result = MockResult()
    viz = Visualizer()
    viz.plot_equity_curve(result, title="Test Strategy")
    print("✅ 可视化测试完成")
