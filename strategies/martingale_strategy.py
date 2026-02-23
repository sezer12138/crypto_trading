"""
Martingale Strategy Example
马丁格尔策略示例

马丁格尔策略是一种博弈策略，核心思想：
亏损后加倍下注，直到获利后回到初始下注额。

⚠️ 高风险警告：
    - 需要充足的资金储备
    - 连续亏损会快速消耗资金
    - 必须设置最大加倍次数
    - 不适合作为主力策略

参数说明:
    base_amount: 初始下单数量
    multiplier: 加倍倍数 (默认2倍)
    max_steps: 最大加倍次数
    target_profit: 目标盈利比例
    stop_loss: 止损比例

使用示例:
    python run_backtest.py --coin btc --strategy martingale --days 365

风险提示:
    这是高风险策略，仅适合有经验的专业交易者。
    建议只在极小资金比例上使用。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategies import MartingaleStrategy
from backtest import BacktestEngine


def demo_martingale_strategy():
    """马丁格尔策略演示"""
    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=500, freq="H")
    np.random.seed(42)
    
    # 生成带小波动的价格数据
    returns = np.random.randn(500) * 0.005
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)
    
    print("=" * 60)
    print("马丁格尔策略演示")
    print("=" * 60)
    print("策略参数:")
    print(f"  初始数量: 0.001 BTC")
    print(f"  加倍倍数: 2.0x")
    print(f"  最大加倍: 5次")
    print(f"  目标盈利: 1%")
    print(f"  止损比例: 5%")
    print()
    
    # 创建策略
    strategy = MartingaleStrategy(
        base_amount=0.001,
        multiplier=2.0,
        max_steps=5,
        target_profit=0.01,
        stop_loss=0.05,
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


def analyze_martingale_risk():
    """分析马丁格尔策略的风险"""
    print("\n" + "=" * 60)
    print("马丁格尔策略风险分析")
    print("=" * 60)
    
    # 计算连续亏损的资金需求
    base = 0.001
    multiplier = 2.0
    max_steps = 5
    
    print("\n资金需求分析 (初始: 0.001 BTC):")
    print("-" * 40)
    total = 0
    for i in range(max_steps + 1):
        amount = base * (multiplier ** i)
        total += amount
        print(f"第{i}次加倍: {amount:.4f} BTC | 累计: {total:.4f} BTC")
    
    print(f"\n最大仓位需求: {total:.4f} BTC")
    print(f"若BTC价格$50,000，需要资金: ${total * 50000:,.2f}")
    
    # 风险警告
    print("\n" + "!" * 60)
    print("⚠️  高风险警告")
    print("!" * 60)
    print("""
1. 第5次加倍时，单次仓位已达初始的 32 倍
2. 连续亏损5次后，累计损失巨大
3. 需要极强的风险承受能力和充足资金
4. 加密货币波动大，连续亏损概率高
5. 建议仅用小资金测试，不要用于实盘
    """)


def demo_different_parameters():
    """对比不同参数的马丁格尔策略"""
    print("\n" + "=" * 60)
    print("不同参数的对比测试")
    print("=" * 60)
    
    # 生成测试数据
    dates = pd.date_range("2024-01-01", periods=300, freq="H")
    np.random.seed(42)
    returns = np.random.randn(300) * 0.008
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1000, 10000, 300),
    }, index=dates)
    
    # 测试不同参数组合
    params_list = [
        {"base_amount": 0.001, "multiplier": 2.0, "max_steps": 3, "target_profit": 0.01},
        {"base_amount": 0.001, "multiplier": 2.0, "max_steps": 5, "target_profit": 0.01},
        {"base_amount": 0.0005, "multiplier": 2.0, "max_steps": 5, "target_profit": 0.015},
        {"base_amount": 0.001, "multiplier": 1.5, "max_steps": 5, "target_profit": 0.01},
    ]
    
    print(f"\n{'参数组合':<40} {'收益率%':<12} {'交易次数':<10}")
    print("-" * 70)
    
    for params in params_list:
        strategy = MartingaleStrategy(**params)
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run_backtest(df.copy(), strategy, coin="BTC")
        
        param_str = f"base={params['base_amount']}, mult={params['multiplier']}, max={params['max_steps']}"
        ret = result.metrics.get('total_return_pct', 0)
        trades = result.metrics.get('total_trades', 0)
        
        print(f"{param_str:<40} {ret:>10.2f}% {trades:>10}")


if __name__ == "__main__":
    demo_martingale_strategy()
    analyze_martingale_risk()
    demo_different_parameters()
