"""Tests for persistent Optuna study safety and trial metadata."""

import json
import sys
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
)


def _bounds() -> SearchBounds:
    """Return fixed interval-resolved bounds for persistence tests."""
    return SearchBounds(12, 8064, 6, 4032, 0.005, 0.25)


def _evaluation() -> TrialEvaluation:
    """Return a complete trial evaluation with two fold records."""
    return TrialEvaluation(
        loss=3.5,
        robust_annual_return_pct=12.5,
        worst_drawdown_pct=18.0,
        instability=4.25,
        missing_round_trips=1,
        folds=(FoldMetrics(10.0, -12.0, 4), FoldMetrics(15.0, -18.0, 2)),
    )


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
    json.dumps(metadata)


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
    json.dumps(trial.user_attrs)
