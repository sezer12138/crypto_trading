"""
Visualization Module
收益曲线可视化
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """回测结果可视化器"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """初始化可视化器"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_equity_curve(
        self,
        result,
        title: str = "Strategy Backtest Result",
        save_path: str = None
    ):
        """
        绘制权益曲线和收益曲线
        
        Args:
            result: BacktestResult 对象
            title: 图表标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. 权益曲线
        ax1 = axes[0]
        ax1.plot(result.equity_curve.index, result.equity_curve.values, 
                label='Portfolio Value', color='#2E86AB', linewidth=2)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 添加初始资金线
        ax1.axhline(y=result.equity_curve.iloc[0], color='gray', 
                   linestyle='--', alpha=0.5, label='Initial Capital')
        
        # 2. 累计收益率
        ax2 = axes[1]
        ax2.plot(result.cumulative_returns.index, result.cumulative_returns.values,
                label='Cumulative Return (%)', color='#A23B72', linewidth=2)
        ax2.fill_between(result.cumulative_returns.index, 0, 
                        result.cumulative_returns.values,
                        alpha=0.3, color='#A23B72')
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. 回撤曲线
        ax3 = axes[2]
        cummax = result.equity_curve.cummax()
        drawdown = (result.equity_curve - cummax) / cummax * 100
        
        ax3.fill_between(drawdown.index, drawdown.values, 0,
                        color='#E74C3C', alpha=0.5, label='Drawdown')
        ax3.plot(drawdown.index, drawdown.values, color='#C0392B', linewidth=1)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 图表已保存: {save_path}")
        
        plt.show()
    
    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        result,
        title: str = "Price Chart with Trading Signals",
        save_path: str = None
    ):
        """
        绘制价格图和交易信号
        
        Args:
            df: 包含价格的 DataFrame
            result: BacktestResult 对象
            title: 图表标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 1. 价格图
        ax1.plot(df.index, df['close'], label='Price', color='#2E86AB', linewidth=1.5)
        
        # 标记买卖点
        buy_trades = [t for t in result.trades if t.action == 'buy']
        sell_trades = [t for t in result.trades if t.action == 'sell']
        
        if buy_trades:
            buy_times = [t.timestamp for t in buy_trades]
            buy_prices = [t.price for t in buy_trades]
            ax1.scatter(buy_times, buy_prices, marker='^', color='green', 
                       s=100, label='Buy', zorder=5)
        
        if sell_trades:
            sell_times = [t.timestamp for t in sell_trades]
            sell_prices = [t.price for t in sell_trades]
            ax1.scatter(sell_times, sell_prices, marker='v', color='red', 
                       s=100, label='Sell', zorder=5)
        
        # 如果有均线，也画出来
        if 'ma_short' in df.columns and 'ma_long' in df.columns:
            ax1.plot(df.index, df['ma_short'], label='MA Short', 
                    color='orange', alpha=0.7, linewidth=1)
            ax1.plot(df.index, df['ma_long'], label='MA Long', 
                    color='purple', alpha=0.7, linewidth=1)
        
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 成交量
        if 'volume' in df.columns:
            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] 
                     else 'red' for i in range(len(df))]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 价格图表已保存: {save_path}")
        
        plt.show()
    
    def plot_monthly_returns(
        self,
        result,
        save_path: str = None
    ):
        """绘制月度收益热力图"""
        if result.daily_returns is None or len(result.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return
        
        # 计算月度收益
        monthly_returns = result.daily_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_df.pivot(index='Month', columns='Year', values='Return')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
        
        # 设置标签
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 添加数值
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 月度收益图已保存: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, object],
        save_path: str = None
    ):
        """
        比较多策略的回测指标
        
        Args:
            results: Dict[str, BacktestResult] - 多个策略的结果
            save_path: 保存路径
        """
        metrics_names = ['total_return_pct', 'annual_return_pct', 
                        'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
        metrics_labels = ['Total Return\n(%)', 'Annual Return\n(%)', 
                         'Sharpe\nRatio', 'Max Drawdown\n(%)', 'Win Rate\n(%)']
        
        fig, axes = plt.subplots(1, len(metrics_names), figsize=(15, 4))
        
        strategies = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        for idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
            ax = axes[idx]
            values = [results[s].metrics.get(metric, 0) for s in strategies]
            
            bars = ax.bar(strategies, values, color=colors, alpha=0.8)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Strategy Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 对比图已保存: {save_path}")
        
        plt.show()
    
    def create_full_report(
        self,
        result,
        df: pd.DataFrame,
        strategy_name: str,
        coin: str,
        output_dir: str = "results"
    ):
        """
        生成完整报告（包含所有图表）
        
        Args:
            result: BacktestResult 对象
            df: 价格数据
            strategy_name: 策略名称
            coin: 币种
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"{output_dir}/{strategy_name}_{coin}"
        
        # 1. 权益曲线
        self.plot_equity_curve(
            result,
            title=f"{strategy_name} - {coin} Backtest",
            save_path=f"{base_name}_equity.png"
        )
        
        # 2. 价格与信号
        self.plot_price_with_signals(
            df,
            result,
            title=f"{coin} Price Chart with Signals",
            save_path=f"{base_name}_signals.png"
        )
        
        # 3. 月度收益
        self.plot_monthly_returns(
            result,
            save_path=f"{base_name}_monthly.png"
        )
        
        logger.info(f"✅ 完整报告已生成: {output_dir}/")


if __name__ == "__main__":
    # 测试可视化
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    equity = 10000 + np.cumsum(np.random.randn(100) * 100)
    
    class MockResult:
        def __init__(self):
            self.equity_curve = pd.Series(equity, index=dates)
            self.cumulative_returns = (self.equity_curve / self.equity_curve.iloc[0] - 1) * 100
            self.daily_returns = self.equity_curve.pct_change().dropna()
            self.metrics = {
                'total_return_pct': 15.5,
                'sharpe_ratio': 1.2
            }
            self.trades = []
    
    result = MockResult()
    viz = Visualizer()
    viz.plot_equity_curve(result, title="Test Strategy")
    print("✅ 可视化测试完成")
