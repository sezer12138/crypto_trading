"""Backtest execution helpers for Bayesian Momentum optimization."""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest import BacktestEngine
from optimization.data import load_ohlcv, split_holdout_and_folds
from optimization.momentum_objective import FoldMetrics, LossConfig, TrialEvaluation, score_folds
from strategies.momentum import MomentumStrategy

_REQUIRED_METRICS = (
    "total_return_pct",
    "annual_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "total_trades",
)


@dataclass(frozen=True)
class BacktestRunConfig:
    """Backtest settings shared by all optimization partitions."""

    initial_capital: float
    coin: str
    drawdown_breaker_enabled: bool
    loss_cooldown_enabled: bool


@dataclass(frozen=True)
class WorkerContext:
    """Read-only data and settings cached by one optimization worker."""

    folds: tuple[pd.DataFrame, ...]
    loss_config: LossConfig
    run_config: BacktestRunConfig


_WORKER_CONTEXT: WorkerContext | None = None


@contextmanager
def _suppress_backtest_logs() -> Iterator[None]:
    """Suppress expected repeated backtest risk logs while preserving logger state."""
    backtest_logger = logging.getLogger("backtest")
    previous_level = backtest_logger.level
    backtest_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        backtest_logger.setLevel(previous_level)


def _run_backtest(
    df: pd.DataFrame,
    parameters: Mapping[str, int | float],
    run_config: BacktestRunConfig,
) -> dict[str, float]:
    """Run one isolated Momentum backtest and retain optimization metrics."""
    with _suppress_backtest_logs():
        engine = BacktestEngine(
            initial_capital=run_config.initial_capital,
            drawdown_breaker_enabled=run_config.drawdown_breaker_enabled,
            loss_cooldown_enabled=run_config.loss_cooldown_enabled,
        )
        result = engine.run_backtest(
            df,
            MomentumStrategy(**parameters),
            coin=run_config.coin,
        )
    if any(name not in result.metrics for name in _REQUIRED_METRICS):
        raise ValueError("Backtest result is missing required optimization metrics")
    return {name: result.metrics[name] for name in _REQUIRED_METRICS}


def evaluate_parameters(
    folds: Sequence[pd.DataFrame],
    parameters: Mapping[str, int | float],
    loss_config: LossConfig,
    run_config: BacktestRunConfig,
) -> TrialEvaluation:
    """Score Momentum parameters across independent chronological folds."""
    metrics = [_run_backtest(fold, parameters, run_config) for fold in folds]
    fold_metrics = [
        FoldMetrics(
            annual_return_pct=metric["annual_return_pct"],
            max_drawdown_pct=metric["max_drawdown_pct"],
            total_trades=int(metric["total_trades"]),
        )
        for metric in metrics
    ]
    return score_folds(fold_metrics, loss_config)


def run_holdout(
    df: pd.DataFrame,
    parameters: Mapping[str, int | float],
    run_config: BacktestRunConfig,
) -> dict[str, float]:
    """Evaluate selected Momentum parameters on one untouched holdout."""
    return _run_backtest(df, parameters, run_config)


def initialize_worker(
    data_path: str,
    holdout_ratio: float,
    fold_count: int,
    max_lookback: int,
    loss_config: LossConfig,
    run_config: BacktestRunConfig,
) -> None:
    """Load and split optimization data once in a worker process."""
    data = load_ohlcv(Path(data_path))
    folds, _ = split_holdout_and_folds(data, holdout_ratio, fold_count, max_lookback)
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = WorkerContext(tuple(folds), loss_config, run_config)


def evaluate_worker(parameters: Mapping[str, int | float]) -> TrialEvaluation:
    """Score parameters with the folds initialized for this worker."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Momentum optimization worker is not initialized")
    return evaluate_parameters(
        _WORKER_CONTEXT.folds,
        parameters,
        _WORKER_CONTEXT.loss_config,
        _WORKER_CONTEXT.run_config,
    )
