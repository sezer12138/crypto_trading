#!/usr/bin/env python3
"""Grid-search utilities for Momentum strategy hyperparameters."""

import argparse
import math
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import BacktestEngine
from strategies.momentum import MomentumStrategy

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
Number = TypeVar("Number", int, float)
METRIC_NAMES = (
    "total_return_pct",
    "annual_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "total_trades",
)
DEFAULT_ROC_PERIODS = [5, 10, 15, 20, 30]
DEFAULT_MOMENTUM_PERIODS = [5, 10, 14, 20, 30]
DEFAULT_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def _parse_positive_list(value: str, converter, label: str) -> List[Number]:
    """Parse a comma-separated list of unique positive numbers."""
    raw_items = value.split(",")
    if not value.strip() or any(not item.strip() for item in raw_items):
        raise ValueError(f"{label} must be a comma-separated list of positive values")

    parsed = []
    try:
        for item in raw_items:
            number = converter(item.strip())
            if number <= 0 or not math.isfinite(number):
                raise ValueError
            if number not in parsed:
                parsed.append(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only positive values") from exc
    return parsed


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated positive integer parameters."""
    return _parse_positive_list(value, int, "Integer parameters")


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated positive floating-point parameters."""
    return _parse_positive_list(value, float, "Floating-point parameters")


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load, validate, and chronologically sort an OHLCV CSV file."""
    data = pd.read_csv(path)
    if "timestamp" not in data.columns:
        raise ValueError("Input data is missing required column: timestamp")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")

    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Input data contains an invalid timestamp") from exc
    if data["timestamp"].duplicated().any():
        raise ValueError("Input data contains duplicate timestamps")

    try:
        for column in REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCV columns must contain numeric values") from exc
    if not np.isfinite(data.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV columns must contain finite numeric values")

    return data.set_index("timestamp").loc[:, list(REQUIRED_COLUMNS)].sort_index()


def chronological_split(
    df: pd.DataFrame, train_ratio: float, max_lookback: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered data into non-overlapping train and validation partitions."""
    if not 0 < train_ratio < 1:
        raise ValueError("The train ratio must be between 0 and 1")
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index].copy()
    validation = df.iloc[split_index:].copy()
    if len(train) <= max_lookback or len(validation) <= max_lookback:
        raise ValueError("Train and validation partitions must exceed the maximum lookback")
    return train, validation


def _prefixed_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Select standard backtest metrics and prefix their names."""
    return {f"{prefix}_{name}": metrics.get(name, 0.0) for name in METRIC_NAMES}


def evaluate_grid(
    train_df: pd.DataFrame,
    roc_periods: Sequence[int],
    momentum_periods: Sequence[int],
    thresholds: Sequence[float],
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
) -> pd.DataFrame:
    """Backtest every Momentum parameter combination on training data."""
    rows = []
    for roc_period, momentum_period, threshold in product(
        roc_periods, momentum_periods, thresholds
    ):
        strategy = MomentumStrategy(
            roc_period=roc_period,
            momentum_period=momentum_period,
            threshold=threshold,
        )
        engine = BacktestEngine(
            initial_capital=capital,
            drawdown_breaker_enabled=drawdown_breaker_enabled,
        )
        result = engine.run_backtest(train_df, strategy, coin=coin)
        row = {
            "roc_period": roc_period,
            "momentum_period": momentum_period,
            "threshold": threshold,
        }
        row.update(_prefixed_metrics(result.metrics, "train"))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank training results by return and deterministic tie breakers."""
    ranked = results.sort_values(
        by=[
            "train_total_return_pct",
            "train_sharpe_ratio",
            "train_max_drawdown_pct",
            "roc_period",
            "momentum_period",
            "threshold",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def validate_winner(
    validation_df: pd.DataFrame,
    ranked_results: pd.DataFrame,
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
) -> Dict[str, float]:
    """Evaluate only the highest-ranked training setting on validation data."""
    winner = ranked_results.iloc[0]
    strategy = MomentumStrategy(
        roc_period=int(winner["roc_period"]),
        momentum_period=int(winner["momentum_period"]),
        threshold=float(winner["threshold"]),
    )
    engine = BacktestEngine(
        initial_capital=capital,
        drawdown_breaker_enabled=drawdown_breaker_enabled,
    )
    result = engine.run_backtest(validation_df, strategy, coin=coin)
    return _prefixed_metrics(result.metrics, "validation")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse Momentum grid-search command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find Momentum strategy parameters using chronological validation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/historical/btc_1h_730d.csv"),
        help="Timestamp-indexed OHLCV CSV file",
    )
    parser.add_argument(
        "--roc-periods",
        type=parse_int_list,
        default=DEFAULT_ROC_PERIODS.copy(),
        help="Comma-separated ROC lookback periods",
    )
    parser.add_argument(
        "--momentum-periods",
        type=parse_int_list,
        default=DEFAULT_MOMENTUM_PERIODS.copy(),
        help="Comma-separated momentum lookback periods",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=DEFAULT_THRESHOLDS.copy(),
        help="Comma-separated positive ROC thresholds",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/momentum_grid_search.csv"),
    )
    parser.add_argument("--coin", default="BTC")
    parser.add_argument(
        "--disable-drawdown-breaker",
        action="store_true",
        help="Disable forced liquidation and halt at the maximum drawdown threshold",
    )
    return parser.parse_args(argv)


def write_results(
    ranked_results: pd.DataFrame,
    validation_metrics: Dict[str, float],
    output: Path,
) -> pd.DataFrame:
    """Attach winner validation metrics and write ranked results to CSV."""
    written = ranked_results.copy()
    for name, value in validation_metrics.items():
        written[name] = float("nan")
        written.loc[written.index[0], name] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    written.to_csv(output, index=False)
    return written


def run_search(args: argparse.Namespace) -> pd.DataFrame:
    """Run training grid search, winner validation, and CSV output."""
    if args.capital <= 0 or not math.isfinite(args.capital):
        raise ValueError("Initial capital must be positive")
    data = load_ohlcv(args.data)
    max_lookback = max(max(args.roc_periods), max(args.momentum_periods))
    train, validation = chronological_split(data, args.train_ratio, max_lookback)
    breaker_enabled = not args.disable_drawdown_breaker

    combination_count = len(args.roc_periods) * len(args.momentum_periods) * len(args.thresholds)
    print(f"Searching {combination_count} parameter combinations")
    print(
        f"Training: {len(train)} rows ({train.index[0]} to {train.index[-1]}); "
        f"validation: {len(validation)} rows ({validation.index[0]} to {validation.index[-1]})"
    )

    training_results = evaluate_grid(
        train,
        args.roc_periods,
        args.momentum_periods,
        args.thresholds,
        args.capital,
        breaker_enabled,
        args.coin.upper(),
    )
    ranked = rank_results(training_results)
    validation_metrics = validate_winner(
        validation,
        ranked,
        args.capital,
        breaker_enabled,
        args.coin.upper(),
    )
    written = write_results(ranked, validation_metrics, args.output)
    winner = written.iloc[0]
    print(
        "Winning parameters: "
        f"roc_period={int(winner['roc_period'])}, "
        f"momentum_period={int(winner['momentum_period'])}, "
        f"threshold={winner['threshold']:.6g}"
    )
    print(f"Training total return: {winner['train_total_return_pct']:.2f}%")
    print(f"Validation total return: {winner['validation_total_return_pct']:.2f}%")
    print(f"Results written to: {args.output}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Momentum grid-search CLI."""
    try:
        args = parse_arguments(argv)
        run_search(args)
        return 0
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
