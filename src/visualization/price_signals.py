"""
价格与交易信号可视化

绘制价格走势图并标注买卖信号，同时显示成交量。
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from visualization._constants import DEFAULT_PRICE_FIGSIZE, PRICE_HEIGHT_RATIO


class PriceSignalsMixin:
    """价格信号绘图方法"""

    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        result: object,
        title: str = "Price Chart with Trading Signals",
        save_path: Optional[str] = None,
        show_plot: bool = True,
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
            2, 1, figsize=DEFAULT_PRICE_FIGSIZE,
            gridspec_kw={"height_ratios": [PRICE_HEIGHT_RATIO, 1]}
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

        # 成交量图（使用向量化操作替代循环）
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

        return fig
