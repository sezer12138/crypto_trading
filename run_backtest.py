#!/usr/bin/env python3
"""
Main Backtest Runner
主回测程序 - 运行完整的历史回测和策略对比

Usage:
    # 默认回测 BTC 使用多因子策略
    python run_backtest.py
    
    # 回测 ETH 使用均线策略
    python run_backtest.py --coin eth --strategy ma_cross
    
    # 回测所有币种
    python run_backtest.py --coin all
    
    # 对比所有策略性能
    python run_backtest.py --coin btc --compare
    
    # 使用1小时数据回测2年
    python run_backtest.py --interval 1h --days 730 --capital 50000

Options:
    --coin          币种 (btc, eth, sol, all) [默认: btc]
    --strategy      交易策略 [默认: multi_factor]
    --days          回测天数 [默认: 730 (2年)]
    --interval      K线时间粒度 [默认: 1h]
    --capital       初始资金 [默认: 10000]
    --compare       对比所有策略
    --no-viz        不显示可视化图表
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from historical_data import HistoricalDataFetcher
from strategies import get_strategy
from backtest import BacktestEngine, BacktestResult
from visualization import Visualizer

# 确保日志目录存在
Path("logs").mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/backtest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Strategy Backtest",
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
        """,
    )

    parser.add_argument(
        "--coin",
        type=str,
        default="btc",
        help="币种 (btc, eth, sol, all) - 默认: btc",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="multi_factor",
        choices=[
            "ma_cross",
            "rsi",
            "bollinger",
            "multi_factor",
            "mean_reversion",
            "macd",
            "breakout",
            "vwap",
            "momentum",
            "atr_stop",
            "stochastic",
            "grid",
            "martingale",
        ],
        help="交易策略 - 默认: multi_factor",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="回测天数 - 默认: 730 (2年)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="K线时间粒度 - 默认: 1h",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="初始资金 - 默认: $10,000",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比所有策略性能",
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不显示可视化图表",
    )

    return parser.parse_args()


def run_single_backtest(
    coin: str,
    strategy_name: str,
    days: int,
    interval: str,
    capital: float,
    fetcher: HistoricalDataFetcher,
) -> Tuple[Optional[BacktestResult], Optional[pd.DataFrame]]:
    """
    运行单次回测
    
    Args:
        coin: 币种代码
        strategy_name: 策略名称
        days: 回测天数
        interval: 时间粒度
        capital: 初始资金
        fetcher: 数据获取器
        
    Returns:
        (BacktestResult, DataFrame) 元组，如果失败则返回 (None, None)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🚀 开始回测 | 币种: {coin.upper()} | 策略: {strategy_name}")
    logger.info(f"{'=' * 60}")

    # 1. 获取历史数据
    data_file = f"data/historical/{coin}_{interval}_{days}d.csv"

    try:
        df = pd.read_csv(data_file, index_col="timestamp", parse_dates=True)
        logger.info(f"📊 从文件加载数据: {len(df)} 条")
    except FileNotFoundError:
        logger.info("📥 下载历史数据...")
        df = fetcher.fetch_historical_data(coin, interval, days, data_file)

    if df.empty:
        logger.error(f"❌ 无法获取 {coin} 数据")
        return None, None

    # 1.1 验证数据完整性
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = interval_minutes.get(interval, 60)
    expected_records = int((days * 24 * 60) / minutes)
    actual_records = len(df)
    data_ratio = actual_records / expected_records

    logger.info(f"📊 数据完整性: {actual_records}/{expected_records} ({data_ratio*100:.1f}%)")

    if data_ratio < 0.5:
        logger.error(f"❌ 数据严重不完整 (仅 {data_ratio*100:.1f}%)，停止回测")
        logger.error(f"   建议: 检查网络连接或稍后重试")
        return None, None
    elif data_ratio < 0.8:
        logger.warning(f"⚠️  数据不完整 ({data_ratio*100:.1f}%)，回测结果可能不准确")
        logger.warning(f"   建议: 重新运行以获取完整数据")

    # 2. 创建策略
    if strategy_name == "grid":
        # 网格策略参数：从数据中自动计算网格范围
        lower_price = df['low'].min()
        upper_price = df['high'].max()
        strategy = get_strategy(strategy_name, lower_price=lower_price, upper_price=upper_price)
    elif strategy_name == "martingale":
        # 马丁格尔策略参数
        strategy = get_strategy(strategy_name, base_amount=0.001, multiplier=2.0, max_steps=5)
    else:
        strategy = get_strategy(strategy_name)

    # 3. 运行回测
    engine = BacktestEngine(initial_capital=capital)
    result = engine.run_backtest(df, strategy, coin=coin.upper())

    # 4. 保存日志
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/backtest_{coin}_{strategy_name}_{timestamp}.json"
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


def compare_strategies(
    coin: str, 
    days: int, 
    interval: str, 
    capital: float,
    save_report: bool = True
) -> Dict[str, BacktestResult]:
    """
    对比多个策略的性能
    
    运行所有策略的回测并生成对比报告。
    
    Args:
        coin: 币种代码
        days: 回测天数
        interval: 时间粒度
        capital: 初始资金
        save_report: 是否保存可视化报告
        
    Returns:
        策略名称到 BacktestResult 的字典
    """
    strategies = [
        "ma_cross",
        "rsi",
        "bollinger",
        "multi_factor",
        "mean_reversion",
        "macd",
        "breakout",
        "vwap",
        "momentum",
        "atr_stop",
        "stochastic",
        "grid",
        "martingale",
    ]
    
    results: Dict[str, BacktestResult] = {}
    data_frames: Dict[str, pd.DataFrame] = {}

    fetcher = HistoricalDataFetcher()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 策略对比 | 币种: {coin.upper()}")
    logger.info(f"{'=' * 60}")

    # 运行每个策略的回测
    for strategy_name in strategies:
        result, df = run_single_backtest(
            coin, strategy_name, days, interval, capital, fetcher
        )

        if result:
            results[strategy_name] = result
            data_frames[strategy_name] = df

    if not results:
        logger.error("❌ 没有成功运行的策略")
        return {}

    # 打印对比表格
    print_comparison_table(results)

    # 生成可视化报告
    if save_report and results:
        viz = Visualizer()
        viz.create_comparison_report(results, coin, "results")

    return results


def print_comparison_table(results: Dict[str, BacktestResult]) -> None:
    """
    打印策略对比表格
    
    Args:
        results: 策略结果字典
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 策略对比结果")
    logger.info(f"{'=' * 60}")

    # 表头
    header = (
        f"{'Strategy':<15} {'Return%':<12} {'CAGR%':<12} "
        f"{'Sharpe':<10} {'MaxDD%':<10} {'Win%':<10} {'Trades':<10}"
    )
    print(f"\n{header}")
    print("-" * 85)

    # 按夏普比率排序
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].metrics.get("sharpe_ratio", 0),
        reverse=True,
    )

    for name, result in sorted_results:
        m = result.metrics
        print(
            f"{name:<15} "
            f"{m.get('total_return_pct', 0):>10.2f}% "
            f"{m.get('annual_return_pct', 0):>10.2f}% "
            f"{m.get('sharpe_ratio', 0):>8.2f}  "
            f"{m.get('max_drawdown_pct', 0):>8.2f}% "
            f"{m.get('win_rate_pct', 0):>8.2f}% "
            f"{m.get('total_trades', 0):>8}"
        )

    # 高亮最佳策略
    if sorted_results:
        best_strategy, best_result = sorted_results[0]
        logger.info(f"\n🏆 最佳策略: {best_strategy}")
        logger.info(f"   夏普比率: {best_result.metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   总收益率: {best_result.metrics.get('total_return_pct', 0):.2f}%")


def main() -> None:
    """主函数"""
    args = parse_arguments()

    # 确保目录存在
    Path("data/historical").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    if args.compare:
        # 策略对比模式
        compare_strategies(args.coin, args.days, args.interval, args.capital)
        
    elif args.coin == "all":
        # 回测所有币种
        coins = ["btc", "eth", "sol"]
        fetcher = HistoricalDataFetcher()

        for coin in coins:
            result, df = run_single_backtest(
                coin,
                args.strategy,
                args.days,
                args.interval,
                args.capital,
                fetcher,
            )

            if result and not args.no_viz:
                viz = Visualizer()
                viz.create_full_report(
                    result, df, args.strategy, coin, "results"
                )
    else:
        # 单次回测模式
        fetcher = HistoricalDataFetcher()
        result, df = run_single_backtest(
            args.coin,
            args.strategy,
            args.days,
            args.interval,
            args.capital,
            fetcher,
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
