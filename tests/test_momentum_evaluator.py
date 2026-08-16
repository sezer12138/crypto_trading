"""Tests for fold and holdout Momentum backtest evaluation."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optimization.momentum_evaluator as evaluator
from optimization.momentum_evaluator import (
    BacktestRunConfig,
    evaluate_parameters,
    evaluate_worker,
    initialize_worker,
    run_holdout,
)
from optimization.momentum_objective import LossConfig


def _frame() -> pd.DataFrame:
    """Return deterministic hourly OHLCV data for evaluator tests."""
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    prices = [100.0 + value for value in range(48)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1.0 for price in prices],
            "low": [price - 1.0 for price in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        },
        index=index,
    )


def _parameters() -> dict[str, int | float]:
    """Return a valid independent Momentum parameter set."""
    return {
        "buy_roc_period": 4,
        "buy_momentum_period": 3,
        "buy_threshold": 0.01,
        "sell_roc_period": 4,
        "sell_momentum_period": 3,
        "sell_threshold": 0.01,
    }


def test_evaluate_parameters_propagates_disabled_controls(monkeypatch) -> None:
    """Passes both disabled risk controls to every partition engine."""
    created: list[dict[str, object]] = []

    class FakeEngine:
        """Minimal backtest result source that records engine configuration."""

        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

        def run_backtest(self, frame: pd.DataFrame, strategy: object, coin: str) -> SimpleNamespace:
            return SimpleNamespace(
                metrics={
                    "annual_return_pct": 12.0,
                    "max_drawdown_pct": -8.0,
                    "total_trades": 6,
                    "total_return_pct": 10.0,
                    "sharpe_ratio": 0.5,
                    "win_rate_pct": 50.0,
                }
            )

    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)

    result = evaluate_parameters(
        [_frame(), _frame()],
        _parameters(),
        LossConfig(),
        BacktestRunConfig(10000.0, "BTC", False, False),
    )

    assert len(created) == 2
    assert all(item["drawdown_breaker_enabled"] is False for item in created)
    assert all(item["loss_cooldown_enabled"] is False for item in created)
    assert result.robust_annual_return_pct == 12.0


def test_run_holdout_returns_only_required_metrics(monkeypatch) -> None:
    """Excludes non-optimization metrics from a holdout result."""

    class FakeEngine:
        """Provide a complete metrics payload including one unrelated value."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def run_backtest(self, frame: pd.DataFrame, strategy: object, coin: str) -> SimpleNamespace:
            return SimpleNamespace(
                metrics={
                    "total_return_pct": 10.0,
                    "annual_return_pct": 12.0,
                    "sharpe_ratio": 0.5,
                    "max_drawdown_pct": -8.0,
                    "win_rate_pct": 50.0,
                    "total_trades": 6,
                    "cost_drag": 1.25,
                }
            )

    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)

    metrics = run_holdout(
        _frame(),
        _parameters(),
        BacktestRunConfig(10000.0, "BTC", True, True),
    )

    assert metrics == {
        "total_return_pct": 10.0,
        "annual_return_pct": 12.0,
        "sharpe_ratio": 0.5,
        "max_drawdown_pct": -8.0,
        "win_rate_pct": 50.0,
        "total_trades": 6,
    }


def test_run_holdout_rejects_missing_optimization_metrics(monkeypatch) -> None:
    """Rejects a backtest payload that cannot support the optimization result."""

    class FakeEngine:
        """Provide an incomplete metrics payload."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def run_backtest(self, frame: pd.DataFrame, strategy: object, coin: str) -> SimpleNamespace:
            return SimpleNamespace(metrics={"annual_return_pct": 12.0})

    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)

    with pytest.raises(
        ValueError, match="Backtest result is missing required optimization metrics"
    ):
        run_holdout(
            _frame(),
            _parameters(),
            BacktestRunConfig(10000.0, "BTC", True, True),
        )


def test_evaluate_parameters_restores_backtest_logger_level(monkeypatch) -> None:
    """Leaves the backtest logger configuration unchanged after partition runs."""

    class FakeEngine:
        """Provide complete metrics without using external dependencies."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def run_backtest(self, frame: pd.DataFrame, strategy: object, coin: str) -> SimpleNamespace:
            return SimpleNamespace(
                metrics={
                    "total_return_pct": 10.0,
                    "annual_return_pct": 12.0,
                    "sharpe_ratio": 0.5,
                    "max_drawdown_pct": -8.0,
                    "win_rate_pct": 50.0,
                    "total_trades": 6,
                }
            )

    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)
    logger = logging.getLogger("backtest")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with evaluator._suppress_backtest_logs():
            assert logger.level == logging.ERROR
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(previous_level)


def test_worker_initialization_evaluates_only_optimization_folds(monkeypatch) -> None:
    """Caches chronological folds once and does not run the final holdout."""
    data = pd.concat([_frame()] * 4)
    data.index = pd.date_range("2024-01-01", periods=len(data), freq="h")
    observed_lengths: list[int] = []

    class FakeEngine:
        """Record the partition size given to each worker backtest."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def run_backtest(self, frame: pd.DataFrame, strategy: object, coin: str) -> SimpleNamespace:
            observed_lengths.append(len(frame))
            return SimpleNamespace(
                metrics={
                    "total_return_pct": 10.0,
                    "annual_return_pct": 12.0,
                    "sharpe_ratio": 0.5,
                    "max_drawdown_pct": -8.0,
                    "win_rate_pct": 50.0,
                    "total_trades": 6,
                }
            )

    monkeypatch.setattr(evaluator, "load_ohlcv", lambda path: data)
    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)
    monkeypatch.setattr(evaluator, "_WORKER_CONTEXT", None)

    initialize_worker(
        "ignored.csv",
        holdout_ratio=0.25,
        fold_count=2,
        max_lookback=5,
        loss_config=LossConfig(),
        run_config=BacktestRunConfig(10000.0, "BTC", True, True),
    )
    result = evaluate_worker(_parameters())

    assert observed_lengths == [72, 72]
    assert result.robust_annual_return_pct == 12.0


def test_evaluate_worker_requires_initialization(monkeypatch) -> None:
    """Rejects parameter evaluation before worker data has been initialized."""
    monkeypatch.setattr(evaluator, "_WORKER_CONTEXT", None)

    with pytest.raises(RuntimeError, match="Momentum optimization worker is not initialized"):
        evaluate_worker(_parameters())
