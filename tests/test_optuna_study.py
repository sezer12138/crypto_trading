"""Tests for persistent Optuna study safety and trial metadata."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.momentum_evaluator import BacktestRunConfig
from optimization.momentum_objective import FoldMetrics, LossConfig, SearchBounds, TrialEvaluation
from optimization.optuna_study import (
    build_metadata,
    create_or_load_study,
    record_evaluation,
    recover_stale_trials,
    run_process_trials,
    run_sequential_trials,
    terminal_trial_count,
)


def _bounds() -> SearchBounds:
    """Return fixed interval-resolved bounds for persistence tests."""
    return SearchBounds(12, 8064, 6, 4032, 0.005, 0.25)


def _fixed_bounds() -> SearchBounds:
    """Return a deterministic search space for coordinator tests."""
    return SearchBounds(12, 12, 6, 6, 0.01, 0.01)


def _evaluation(loss: float = 3.5) -> TrialEvaluation:
    """Return a complete trial evaluation with two fold records."""
    return TrialEvaluation(
        loss=loss,
        robust_annual_return_pct=12.5,
        worst_drawdown_pct=18.0,
        instability=4.25,
        missing_round_trips=1,
        folds=(FoldMetrics(10.0, -12.0, 4), FoldMetrics(15.0, -18.0, 2)),
    )


def _process_evaluator(parameters: dict[str, int | float]) -> TrialEvaluation:
    """Evaluate parameters in a picklable worker without access to study storage."""
    return _evaluation(float(parameters["buy_roc_period"]))


def test_sequential_target_is_total_trials(tmp_path: Path) -> None:
    """Schedules only the missing terminal trials when resuming a study."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    trial = study.ask()
    trial.suggest_int("buy_roc_period", 12, 12, log=True)
    study.tell(trial, 5.0)
    calls: list[dict[str, int | float]] = []

    summary = run_sequential_trials(
        study,
        3,
        _fixed_bounds(),
        lambda parameters: calls.append(parameters) or _evaluation(1.0),
    )

    assert len(calls) == 2
    assert terminal_trial_count(study) == 3
    assert summary.completed == 3
    assert summary.failed == 0
    assert summary.interrupted is False


def test_sequential_evaluation_failure_still_reaches_target(tmp_path: Path) -> None:
    """Turns a caught evaluator failure terminal and continues to the exact target."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    calls = 0

    def fail_first(parameters: dict[str, int | float]) -> TrialEvaluation:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("invalid evaluation")
        return _evaluation(float(parameters["buy_roc_period"]))

    summary = run_sequential_trials(study, 3, _fixed_bounds(), fail_first)

    assert calls == 3
    assert terminal_trial_count(study) == 3
    assert summary.completed == 2
    assert summary.failed == 1
    assert summary.interrupted is False
    assert study.trials[0].user_attrs["failure"] == "invalid evaluation"


def test_process_runner_keeps_sqlite_in_parent(tmp_path: Path) -> None:
    """Coordinates real process workers while all Optuna persistence stays in the parent."""
    path = tmp_path / "study.db"
    study = create_or_load_study(path, "momentum", {}, 42, 5, True)

    summary = run_process_trials(
        study,
        4,
        2,
        _fixed_bounds(),
        None,
        (),
        _process_evaluator,
    )

    resumed = create_or_load_study(path, "momentum", {}, 42, 5, True)
    assert terminal_trial_count(resumed) == 4
    assert summary.completed == 4
    assert summary.failed == 0
    assert summary.interrupted is False


def test_matching_metadata_resumes_and_mismatch_fails(tmp_path: Path) -> None:
    """Keeps completed trials only when the immutable run identity matches."""
    path = tmp_path / "study.db"
    first = create_or_load_study(path, "momentum", {"schema": 1}, 42, 5, False)
    trial = first.ask()
    trial.suggest_int("x", 1, 2)
    first.tell(trial, 1.0)

    resumed = create_or_load_study(path, "momentum", {"schema": 1}, 42, 5, False)

    assert len(resumed.trials) == 1
    with pytest.raises(ValueError, match="metadata does not match"):
        create_or_load_study(path, "momentum", {"schema": 2}, 42, 5, False)


def test_metadata_free_study_with_trials_cannot_be_adopted(tmp_path: Path) -> None:
    """Rejects legacy trial evidence that cannot be proved compatible with this run."""
    path = tmp_path / "study.db"
    legacy = optuna.create_study(
        study_name="momentum",
        storage=f"sqlite:///{path.resolve()}",
        direction="minimize",
    )
    trial = legacy.ask()
    trial.suggest_int("x", 1, 2)
    legacy.tell(trial, 1.0)

    with pytest.raises(ValueError, match="metadata is missing"):
        create_or_load_study(path, "momentum", {"schema": 1}, 42, 5, False)


def test_recover_stale_trials_marks_running_trials_failed(tmp_path: Path) -> None:
    """Converts interrupted ask-created trials to terminal failures before resuming."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {"schema": 1}, 42, 5, False)
    trial = study.ask()
    trial.suggest_int("x", 1, 2)

    assert recover_stale_trials(study) == 1
    assert study.trials[0].state is optuna.trial.TrialState.FAIL
    assert recover_stale_trials(study) == 0


def test_build_metadata_records_the_full_immutable_run_identity(tmp_path: Path) -> None:
    """Captures data, search, split, loss, and risk settings needed for safe resume."""
    data_path = tmp_path / "ohlcv.csv"
    data_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    metadata = build_metadata(
        data_path,
        data,
        60,
        "BTC",
        10_000.0,
        _bounds(),
        LossConfig(drawdown_weight=3.0),
        0.2,
        4,
        BacktestRunConfig(10_000.0, "BTC", False, True),
    )

    assert metadata == {
        "schema": 1,
        "data": {
            "path": str(data_path.resolve()),
            "sha256": "8c16152e4bd556aa7ba69aead7982b82cc8f295ef97ea18da2ced47a86975d0b",
            "rows": 3,
            "first_timestamp": "2024-01-01T00:00:00",
            "last_timestamp": "2024-01-01T02:00:00",
        },
        "interval_minutes": 60,
        "coin": "BTC",
        "capital": 10_000.0,
        "bounds": {
            "roc_min": 12,
            "roc_max": 8064,
            "momentum_min": 6,
            "momentum_max": 4032,
            "threshold_min": 0.005,
            "threshold_max": 0.25,
        },
        "holdout_ratio": 0.2,
        "fold_count": 4,
        "loss_config": {
            "drawdown_target_pct": 25.0,
            "drawdown_weight": 3.0,
            "stability_weight": 0.5,
            "minimum_round_trips": 2,
            "missing_round_trip_penalty": 25.0,
        },
        "risk_controls": {
            "drawdown_breaker_enabled": False,
            "loss_cooldown_enabled": True,
        },
    }
    json.dumps(metadata, allow_nan=False)


def test_build_metadata_normalizes_coin_identity(tmp_path: Path) -> None:
    """Stores a canonical uppercase coin while preserving a matching run configuration."""
    data_path = tmp_path / "ohlcv.csv"
    data_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    metadata = build_metadata(
        data_path,
        data,
        60,
        "btc",
        10_000.0,
        _bounds(),
        LossConfig(),
        0.2,
        4,
        BacktestRunConfig(10_000.0, "btc", False, True),
    )

    assert metadata["coin"] == "BTC"


@pytest.mark.parametrize(
    ("coin", "capital", "run_config"),
    [
        ("ETH", 10_000.0, BacktestRunConfig(10_000.0, "BTC", False, True)),
        ("BTC", 5_000.0, BacktestRunConfig(10_000.0, "BTC", False, True)),
    ],
)
def test_build_metadata_rejects_identity_mismatch(
    tmp_path: Path,
    coin: str,
    capital: float,
    run_config: BacktestRunConfig,
) -> None:
    """Rejects metadata inputs that would misdescribe the executed backtests."""
    data_path = tmp_path / "ohlcv.csv"
    data_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    with pytest.raises(ValueError, match="must match the backtest run configuration"):
        build_metadata(
            data_path,
            data,
            60,
            coin,
            capital,
            _bounds(),
            LossConfig(),
            0.2,
            4,
            run_config,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_build_metadata_rejects_non_finite_values(tmp_path: Path, invalid: float) -> None:
    """Prevents non-finite capital from entering persistent JSON metadata."""
    data_path = tmp_path / "ohlcv.csv"
    data_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    with pytest.raises(ValueError, match="finite"):
        build_metadata(
            data_path,
            data,
            60,
            "BTC",
            invalid,
            _bounds(),
            LossConfig(),
            0.2,
            4,
            BacktestRunConfig(invalid, "BTC", False, True),
        )


def test_record_evaluation_stores_json_serializable_aggregate_and_fold_metrics(
    tmp_path: Path,
) -> None:
    """Makes optimization evidence available to report generation after a resume."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {"schema": 1}, 42, 5, False)
    trial = study.ask()

    record_evaluation(trial, _evaluation())

    assert trial.user_attrs == {
        "robust_annual_return_pct": 12.5,
        "worst_drawdown_pct": 18.0,
        "instability": 4.25,
        "missing_round_trips": 1,
        "folds": [
            {"annual_return_pct": 10.0, "max_drawdown_pct": -12.0, "total_trades": 4},
            {"annual_return_pct": 15.0, "max_drawdown_pct": -18.0, "total_trades": 2},
        ],
    }
    json.dumps(trial.user_attrs, allow_nan=False)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_record_evaluation_rejects_non_finite_values(tmp_path: Path, invalid: float) -> None:
    """Prevents non-finite aggregate evaluation values from entering trial attributes."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {"schema": 1}, 42, 5, False)
    trial = study.ask()

    with pytest.raises(ValueError, match="finite"):
        record_evaluation(trial, replace(_evaluation(), robust_annual_return_pct=invalid))

    assert trial.user_attrs == {}
