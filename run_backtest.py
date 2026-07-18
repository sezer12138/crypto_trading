#!/usr/bin/env python3
"""
Main Backtest Runner - Runs full historical backtests and strategy comparisons

Usage:
    # Default: backtest BTC with multi-factor strategy
    python run_backtest.py

    # Backtest ETH with MA cross strategy
    python run_backtest.py --coin eth --strategy ma_cross

    # Backtest all coins
    python run_backtest.py --coin all

    # Compare all strategy performance
    python run_backtest.py --coin btc --compare

    # Backtest 2 years with 1h interval
    python run_backtest.py --interval 1h --days 730 --capital 50000

Options:
    --coin          Coin symbol (btc, eth, sol, all) [default: btc]
    --strategy      Trading strategy [default: multi_factor]
    --days          Number of days to backtest [default: 730 (2 years)]
    --interval      K-line interval [default: 1h]
    --capital       Initial capital [default: 10000]
    --compare       Compare all strategies
    --no-viz        Skip visualization charts
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
from visualization.html_report import HTMLReportGenerator

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Backtest BTC with multi-factor strategy (default)
    python run_backtest.py

    # Backtest ETH with MA cross strategy
    python run_backtest.py --coin eth --strategy ma_cross

    # Backtest all coins
    python run_backtest.py --coin all

    # Compare all strategies
    python run_backtest.py --coin btc --compare

    # Backtest 2 years with 1h interval
    python run_backtest.py --interval 1h --days 730
        """,
    )

    parser.add_argument(
        "--coin",
        type=str,
        default="btc",
        help="Coin symbol (btc, eth, sol, all) - default: btc",
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
        help="Trading strategy - default: multi_factor",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Number of days to backtest - default: 730 (2 years)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="K-line interval - default: 1h",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital - default: $10,000",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all strategy performance",
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization charts",
    )

    parser.add_argument(
        "--disable-drawdown-breaker",
        action="store_true",
        help="Disable forced liquidation and trading halt at the maximum drawdown threshold",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["binance", "okx"],
        help="Data source for historical data (default: binance, use okx if Binance is blocked in your region)",
    )

    return parser.parse_args()


def run_single_backtest(
    coin: str,
    strategy_name: str,
    days: int,
    interval: str,
    capital: float,
    fetcher: HistoricalDataFetcher,
    generate_html: bool = True,
    drawdown_breaker_enabled: bool = True,
) -> Tuple[Optional[BacktestResult], Optional[pd.DataFrame]]:
    """
    Run a single backtest

    Args:
        coin: Coin symbol
        strategy_name: Strategy name
        days: Number of days
        interval: K-line interval
        capital: Initial capital
        fetcher: Data fetcher instance
        generate_html: Whether to generate HTML report
        drawdown_breaker_enabled: Whether to force liquidation and halt trading at max drawdown

    Returns:
        (BacktestResult, DataFrame) tuple, or (None, None) on failure
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting backtest | Coin: {coin.upper()} | Strategy: {strategy_name}")
    logger.info(f"{'=' * 60}")

    # 1. Fetch historical data
    data_file = f"data/historical/{coin}_{interval}_{days}d.csv"

    try:
        df = pd.read_csv(data_file, index_col="timestamp", parse_dates=True)
        logger.info(f"Loaded data from file: {len(df)} records")
    except FileNotFoundError:
        logger.info("Downloading historical data...")
        df = fetcher.fetch_historical_data(coin, interval, days, data_file)

    if df.empty:
        logger.error(f"Failed to fetch {coin} data")
        return None, None

    # 1.1 Validate data completeness
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = interval_minutes.get(interval, 60)
    expected_records = int((days * 24 * 60) / minutes)
    actual_records = len(df)
    data_ratio = actual_records / expected_records

    logger.info(f"Data completeness: {actual_records}/{expected_records} ({data_ratio*100:.1f}%)")

    if data_ratio < 0.5:
        logger.error(f"Data severely incomplete (only {data_ratio*100:.1f}%), stopping backtest")
        logger.error("   Suggestion: check network connection or try again later")
        return None, None
    elif data_ratio < 0.8:
        logger.warning(
            f"Data incomplete ({data_ratio*100:.1f}%), backtest results may be inaccurate"
        )
        logger.warning("   Suggestion: re-run to fetch complete data")

    # 1.2 Warn about sub-hourly intervals
    if interval_minutes.get(interval, 60) < 60:
        logger.warning(
            f"Warning: Sub-hourly interval '{interval}' may produce excessive trades "
            f"and high transaction costs. Consider using 1h or longer intervals for "
            f"more reliable results."
        )

    # 2. Create strategy
    if strategy_name == "grid":
        # Grid strategy: calculate grid range from first N bars to avoid look-ahead bias
        lookback_bars = min(100, len(df))
        lower_price = df["low"].iloc[:lookback_bars].min()
        upper_price = df["high"].iloc[:lookback_bars].max()
        # Add margin to accommodate price movement beyond the initial range
        margin = (upper_price - lower_price) * 0.1
        lower_price -= margin
        upper_price += margin
        strategy = get_strategy(strategy_name, lower_price=lower_price, upper_price=upper_price)
    elif strategy_name == "martingale":
        strategy = get_strategy(strategy_name, base_amount=0.001, multiplier=2.0, max_steps=5)
    else:
        strategy = get_strategy(strategy_name)

    # 3. Run backtest
    engine = BacktestEngine(
        initial_capital=capital,
        drawdown_breaker_enabled=drawdown_breaker_enabled,
    )
    result = engine.run_backtest(df, strategy, coin=coin.upper())

    # 4. Save logs
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/backtest_{coin}_{strategy_name}_{days}d_{interval}_{timestamp}.json"
    result.save_logs(log_file)

    # 5. Print results
    logger.info(f"\nBacktest results:")
    logger.info(f"   Strategy: {strategy_name}")
    logger.info(f"   Total return: {result.metrics.get('total_return_pct', 0):.2f}%")
    logger.info(f"   Annual return: {result.metrics.get('annual_return_pct', 0):.2f}%")
    logger.info(f"   Sharpe ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"   Max drawdown: {result.metrics.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"   Win rate: {result.metrics.get('win_rate_pct', 0):.2f}%")
    logger.info(f"   Total trades: {result.metrics.get('total_trades', 0)}")

    # 6. Generate HTML report
    if generate_html:
        try:
            html_generator = HTMLReportGenerator()
            html_file = f"results/{strategy_name}_{coin}_{days}d_{interval}_{timestamp}.html"
            html_generator.generate_single_report(
                result=result,
                df=df,
                strategy_name=strategy_name,
                coin=coin,
                days=days,
                interval=interval,
                capital=capital,
                output_path=html_file,
            )
        except Exception as e:
            logger.warning(f"HTML report generation failed: {e}")

    return result, df


def compare_strategies(
    coin: str,
    days: int,
    interval: str,
    capital: float,
    save_report: bool = True,
    data_source: str = None,
    drawdown_breaker_enabled: bool = True,
) -> Dict[str, BacktestResult]:
    """
    Compare performance of multiple strategies

    Runs backtests for all strategies and generates a comparison report.

    Args:
        coin: Coin symbol
        days: Number of days
        interval: K-line interval
        capital: Initial capital
        save_report: Whether to save visualization report
        data_source: Data source ("binance" or "okx")
        drawdown_breaker_enabled: Whether to force liquidation and halt trading at max drawdown

    Returns:
        Dict mapping strategy name to BacktestResult
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

    fetcher = HistoricalDataFetcher(data_source=data_source)

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"Strategy comparison | Coin: {coin.upper()} | Days: {days}d | Interval: {interval}"
    )
    logger.info(f"{'=' * 60}")

    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    if interval_minutes.get(interval, 60) < 60:
        logger.warning(
            f"Warning: Using sub-hourly interval '{interval}'. This may produce excessive "
            f"trades and high transaction costs. Consider using 1h or longer intervals."
        )

    # Run backtest for each strategy
    for strategy_name in strategies:
        result, df = run_single_backtest(
            coin,
            strategy_name,
            days,
            interval,
            capital,
            fetcher,
            drawdown_breaker_enabled=drawdown_breaker_enabled,
        )

        if result:
            results[strategy_name] = result
            data_frames[strategy_name] = df

    if not results:
        logger.error("No successful strategy runs")
        return {}

    # Print comparison table
    print_comparison_table(results)

    # Generate visualization report
    if save_report and results:
        viz = Visualizer()
        chart_paths = viz.create_comparison_report(results, coin, "results", days, interval)

        # Generate HTML comparison report
        try:
            html_generator = HTMLReportGenerator()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            html_file = f"results/comparison_{coin}_{days}d_{interval}_{timestamp}.html"

            # Map chart paths to format needed by HTML generator
            charts_base64 = {}
            if chart_paths:
                for path in chart_paths:
                    if "metrics" in path:
                        charts_base64["metrics"] = path
                    elif "ranking" in path:
                        charts_base64["ranking"] = path
                    elif "equity" in path:
                        charts_base64["equity"] = path
                    elif "trade_details" in path:
                        charts_base64["trade_details"] = path

            html_generator.generate_comparison_report(
                results=results,
                coin=coin,
                days=days,
                interval=interval,
                capital=capital,
                output_path=html_file,
                chart_paths=charts_base64 if chart_paths else None,
            )
        except Exception as e:
            logger.warning(f"HTML comparison report generation failed: {e}")

    return results


def print_comparison_table(results: Dict[str, BacktestResult]) -> None:
    """
    Print strategy comparison table

    Args:
        results: Strategy results dictionary
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Strategy comparison results")
    logger.info(f"{'=' * 60}")

    # Table header
    header = (
        f"{'Strategy':<15} {'Return%':<12} {'CAGR%':<12} "
        f"{'Sharpe':<10} {'MaxDD%':<10} {'Win%':<10} {'Trades':<10}"
    )
    print(f"\n{header}")
    print("-" * 85)

    # Sort by Sharpe ratio
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

    # Highlight best strategy
    if sorted_results:
        best_strategy, best_result = sorted_results[0]
        logger.info(f"\nBest strategy: {best_strategy}")
        logger.info(f"   Sharpe ratio: {best_result.metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   Total return: {best_result.metrics.get('total_return_pct', 0):.2f}%")


def main() -> None:
    """Main entry point"""
    args = parse_arguments()
    drawdown_breaker_enabled = not args.disable_drawdown_breaker

    # Ensure directories exist
    Path("data/historical").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Strategy comparison mode
        compare_strategies(
            args.coin,
            args.days,
            args.interval,
            args.capital,
            data_source=args.source,
            drawdown_breaker_enabled=drawdown_breaker_enabled,
        )

    elif args.coin == "all":
        # Backtest all coins
        coins = ["btc", "eth", "sol"]
        fetcher = HistoricalDataFetcher(data_source=args.source)

        for coin in coins:
            result, df = run_single_backtest(
                coin,
                args.strategy,
                args.days,
                args.interval,
                args.capital,
                fetcher,
                drawdown_breaker_enabled=drawdown_breaker_enabled,
            )

            if result and not args.no_viz:
                viz = Visualizer()
                viz.create_full_report(
                    result, df, args.strategy, coin, "results", args.days, args.interval
                )
    else:
        # Single backtest mode
        fetcher = HistoricalDataFetcher(data_source=args.source)
        result, df = run_single_backtest(
            args.coin,
            args.strategy,
            args.days,
            args.interval,
            args.capital,
            fetcher,
            drawdown_breaker_enabled=drawdown_breaker_enabled,
        )

        if result and not args.no_viz:
            viz = Visualizer()
            viz.create_full_report(
                result, df, args.strategy, args.coin, "results", args.days, args.interval
            )

    logger.info("\nBacktest complete!")
    logger.info("Results saved in: results/")
    logger.info("Logs saved in: logs/")


if __name__ == "__main__":
    main()
