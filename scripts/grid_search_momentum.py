#!/usr/bin/env python3
"""Grid-search utilities for Momentum strategy hyperparameters."""

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TypeVar

import pandas as pd

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


def _parse_positive_list(value: str, converter, label: str) -> List[Number]:
    """Parse a comma-separated list of unique positive numbers."""
    raw_items = value.split(",")
    if not value.strip() or any(not item.strip() for item in raw_items):
        raise ValueError(f"{label} must be a comma-separated list of positive values")

    parsed = []
    try:
        for item in raw_items:
            number = converter(item.strip())
            if number <= 0:
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
