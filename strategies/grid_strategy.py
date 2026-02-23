"""
Grid Trading Strategy Example
网格交易策略示例

网格策略在预设的价格区间内，每隔一定间距挂单买入和卖出。
适合震荡行情，在高波动市场可能面临穿网风险。

特点:
    - 自动低买高卖
    - 适合震荡市场
    - 需要设置合理的网格范围
    - 高波动市场可能大幅亏损

参数说明:
    lower_price: 网格下限价格
    upper_price: 网格上限价格
    grid_num: 网格数量
    amount_per_grid: 每格交易数量

使用示例:
    python run_backtest.py --coin btc --strategy grid --days 365

风险提示:
    单边行情中可能快速亏损，务必设置止损。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategies import GridStrategy
from backtest import BacktestEngine


def demo_grid_strategy():
    """网格策略演示"""
    # 创建测试数据 (震荡行情)
    dates = pd.date_range("2024-01-01", periods=500, freq="H")
    np.random.seed(42)
    
    # 生成震荡价格数据
    t = np.arange(500)
    base_price = 50000
    oscillation = 5000 * np.sin(t * 0.05)  # 震荡
    noise = np.random.randn(500) * 200     # 噪声
    prices = base_price + oscillation + noise
    
    df = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)
    
    # 设置网格参数
    lower_price = df['low'].min()  # 约 44000
    upper_price = df['high'].max()  # 约 56000
    
    print("=" * 60)
    print("网格交易策略演示")
    print("=" * 60)
    print(f"价格区间: ${lower_price:,.2f} - ${upper_price:,.2f}")
    print(f"网格数量: 10")
    print(f"每格数量: 0.01 BTC")
    print()
    
    # 创建策略
    strategy = GridStrategy(
        lower_price=lower_price,
        upper_price=upper_price,
        grid_num=10,
        amount_per_grid=0.01,
    )
    
    # 运行回测
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run_backtest(df, strategy, coin="BTC")
    
    # 打印结果
    print("\n回测结果:")
    print(f"总收益率: {result.metrics.get('total_return_pct', 0):.2f}%")
    print(f"年化收益: {result.metrics.get('annual_return_pct', 0):.2f}%")
    print(f"夏普比率: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"最大回撤: {result.metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"交易次数: {result.metrics.get('total_trades', 0)}")
    
    return result


def demo_grid_vs_trend():
    """对比网格策略在震荡vs趋势行情的表现"""
    print("\n" + "=" * 60)
    print("网格策略：震荡行情 vs 趋势行情")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. 震荡行情
    print("\n【震荡行情】")
    dates1 = pd.date_range("2024-01-01", periods=300, freq="H")
    t = np.arange(300)
    prices1 = 50000 + 3000 * np.sin(t * 0.08) + np.random.randn(300) * 150
    
    df1 = pd.DataFrame({
        "open": prices1 * 0.998,
        "high": prices1 * 1.005,
        "low": prices1 * 0.995,
        "close": prices1,
        "volume": np.random.randint(1000, 10000, 300),
    }, index=dates1)
    
    strategy1 = GridStrategy(
        lower_price=df1['low'].min(),
        upper_price=df1['high'].max(),
        grid_num=8,
        amount_per_grid=0.01,
    )
    
    engine1 = BacktestEngine(initial_capital=10000)
    result1 = engine1.run_backtest(df1, strategy1, coin="BTC")
    
    print(f"收益率: {result1.metrics.get('total_return_pct', 0):.2f}%")
    print(f"交易次数: {result1.metrics.get('total_trades', 0)}")
    
    # 2. 单边上涨行情
    print("\n【单边上涨行情】")
    dates2 = pd.date_range("2024-01-01", periods=300, freq="H")
    trend = np.linspace(45000, 65000, 300)
    noise = np.random.randn(300) * 200
    prices2 = trend + noise
    
    df2 = pd.DataFrame({
        "open": prices2 * 0.998,
        "high": prices2 * 1.005,
        "low": prices2 * 0.995,
        "close": prices2,
        "volume": np.random.randint(1000, 10000, 300),
    }, index=dates2)
    
    strategy2 = GridStrategy(
        lower_price=df2['low'].min(),
        upper_price=df2['high'].max(),
        grid_num=8,
        amount_per_grid=0.01,
    )
    
    engine2 = BacktestEngine(initial_capital=10000)
    result2 = engine2.run_backtest(df2, strategy2, coin="BTC")
    
    print(f"收益率: {result2.metrics.get('total_return_pct', 0):.2f}%")
    print(f"交易次数: {result2.metrics.get('total_trades', 0)}")
    
    # 3. 单边下跌行情
    print("\n【单边下跌行情】")
    dates3 = pd.date_range("2024-01-01", periods=300, freq="H")
    trend = np.linspace(55000, 35000, 300)
    noise = np.random.randn(300) * 200
    prices3 = trend + noise
    
    df3 = pd.DataFrame({
        "open": prices3 * 0.998,
        "high": prices3 * 1.005,
        "low": prices3 * 0.995,
        "close": prices3,
        "volume": np.random.randint(1000, 10000, 300),
    }, index=dates3)
    
    strategy3 = GridStrategy(
        lower_price=df3['low'].min(),
        upper_price=df3['high'].max(),
        grid_num=8,
        amount_per_grid=0.01,
    )
    
    engine3 = BacktestEngine(initial_capital=10000)
    result3 = engine3.run_backtest(df3, strategy3, coin="BTC")
    
    print(f"收益率: {result3.metrics.get('total_return_pct', 0):.2f}%")
    print(f"交易次数: {result3.metrics.get('total_trades', 0)}")
    
    print("\n" + "=" * 60)
    print("结论：网格策略适合震荡行情，在单边行情中表现较差")
    print("=" * 60)


if __name__ == "__main__":
    demo_grid_strategy()
    demo_grid_vs_trend()
