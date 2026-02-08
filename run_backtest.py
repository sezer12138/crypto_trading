#!/usr/bin/env python3
"""
Main Backtest Runner
主回测程序 - 运行完整的历史回测

Usage:
    python run_backtest.py
    python run_backtest.py --coin btc --strategy multi_factor --days 730
    python run_backtest.py --coin all --compare
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from historical_data import HistoricalDataFetcher
from strategies import get_strategy
from backtest import BacktestEngine
from visualization import Visualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Crypto Trading Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 回测BTC使用多因子策略（默认）
    python run_backtest.py
    
    # 回测ETH使用均线策略
    python run_backtest.py --coin eth --strategy ma_cross
    
    # 回测所有币种
    python run_backtest.py --coin all
    
    # 对比多个策略
    python run_backtest.py --coin btc --compare
    
    # 使用1小时数据回测2年
    python run_backtest.py --interval 1h --days 730
        """
    )
    
    parser.add_argument(
        '--coin',
        type=str,
        default='btc',
        help='币种 (btc, eth, sol, all) - 默认: btc'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='multi_factor',
        choices=['ma_cross', 'rsi', 'bollinger', 'multi_factor', 'mean_reversion'],
        help='交易策略 - 默认: multi_factor'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='回测天数 - 默认: 730 (2年)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='K线时间粒度 - 默认: 1h'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='初始资金 - 默认: $10,000'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='对比所有策略性能'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='不显示可视化图表'
    )
    
    return parser.parse_args()


def run_single_backtest(
    coin: str,
    strategy_name: str,
    days: int,
    interval: str,
    capital: float,
    fetcher: HistoricalDataFetcher
) -> tuple:
    """运行单次回测"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 开始回测 | 币种: {coin.upper()} | 策略: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # 1. 获取历史数据
    data_file = f"data/historical/{coin}_{interval}_{days}d.csv"
    
    try:
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
        logger.info(f"📊 从文件加载数据: {len(df)} 条")
    except FileNotFoundError:
        logger.info("📥 下载历史数据...")
        df = fetcher.fetch_historical_data(coin, interval, days, data_file)
    
    if df.empty:
        logger.error(f"❌ 无法获取 {coin} 数据")
        return None, None
    
    # 2. 创建策略
    strategy = get_strategy(strategy_name)
    
    # 3. 运行回测
    engine = BacktestEngine(initial_capital=capital)
    result = engine.run_backtest(df, strategy, coin=coin.upper())
    
    # 4. 保存日志
    log_file = f"logs/backtest_{coin}_{strategy_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    result.save_logs(log_file)
    
    # 5. 打印结果
    logger.info(f"\n📈 回测结果:")
    logger.info(f"   策略: {strategy_name}")
    logger.info(f"   总收益率: {result.metrics.get('total_return_pct', 0):.2f}%")
    logger.info(f"   年化收益: {result.metrics.get('annual_return_pct', 0):.2f}%")
    logger.info(f"   夏普比率: {result.metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"   最大回撤: {result.metrics.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"   胜率: {result.metrics.get('win_rate_pct', 0):.2f}%")
    logger.info(f"   交易次数: {result.metrics.get('total_trades', 0)}")
    
    return result, df


def compare_strategies(coin: str, days: int, interval: str, capital: float):
    """对比多个策略"""
    strategies = ['ma_cross', 'rsi', 'bollinger', 'multi_factor', 'mean_reversion']
    results = {}
    data_frames = {}
    
    fetcher = HistoricalDataFetcher()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 策略对比 | 币种: {coin.upper()}")
    logger.info(f"{'='*60}")
    
    for strategy_name in strategies:
        result, df = run_single_backtest(
            coin, strategy_name, days, interval, capital, fetcher
        )
        
        if result:
            results[strategy_name] = result
            data_frames[strategy_name] = df
    
    # 打印对比表格
    logger.info(f"\n{'='*60}")
    logger.info("📊 策略对比结果")
    logger.info(f"{'='*60}")
    
    print(f"\n{'Strategy':<15} {'Return%':<12} {'CAGR%':<12} {'Sharpe':<10} {'MaxDD%':<10} {'Win%':<10} {'Trades':<10}")
    print("-" * 85)
    
    for name, result in results.items():
        m = result.metrics
        print(f"{name:<15} "
              f"{m.get('total_return_pct', 0):>10.2f}% "
              f"{m.get('annual_return_pct', 0):>10.2f}% "
              f"{m.get('sharpe_ratio', 0):>8.2f}  "
              f"{m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('win_rate_pct', 0):>8.2f}% "
              f"{m.get('total_trades', 0):>8}")
    
    # 可视化对比
    if results:
        viz = Visualizer()
        viz.plot_metrics_comparison(
            results,
            save_path=f"results/strategy_comparison_{coin}.png"
        )
    
    return results


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确保目录存在
    Path("data/historical").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # 对比模式
        compare_strategies(args.coin, args.days, args.interval, args.capital)
    elif args.coin == 'all':
        # 回测所有币种
        coins = ['btc', 'eth', 'sol']
        fetcher = HistoricalDataFetcher()
        
        for coin in coins:
            result, df = run_single_backtest(
                coin, args.strategy, args.days, args.interval, 
                args.capital, fetcher
            )
            
            if result and not args.no_viz:
                viz = Visualizer()
                viz.create_full_report(
                    result, df, args.strategy, coin, "results"
                )
    else:
        # 单次回测
        fetcher = HistoricalDataFetcher()
        result, df = run_single_backtest(
            args.coin, args.strategy, args.days, args.interval,
            args.capital, fetcher
        )
        
        if result and not args.no_viz:
            viz = Visualizer()
            viz.create_full_report(
                result, df, args.strategy, args.coin, "results"
            )
    
    logger.info("\n✅ 回测完成！")
    logger.info(f"📁 结果保存在: results/")
    logger.info(f"📝 日志保存在: logs/")


if __name__ == "__main__":
    main()
