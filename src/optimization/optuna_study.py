"""Persistent Optuna studies and auditable Bayesian trial metadata."""

from dataclasses import asdict
import math
from pathlib import Path
from typing import TypeAlias

import optuna
import pandas as pd
from optuna.study import Study
from optuna.trial import Trial, TrialState

from optimization.data import fingerprint_file
from optimization.momentum_evaluator import BacktestRunConfig
from optimization.momentum_objective import LossConfig, SearchBounds, TrialEvaluation

JSONSerializable: TypeAlias = (
    str | int | float | bool | None | list["JSONSerializable"] | dict[str, "JSONSerializable"]
)

OPTIMIZER_SCHEMA_VERSION = 1


def build_metadata(
    data_path: Path,
    data: pd.DataFrame,
    interval_minutes: int,
    coin: str,
    capital: float,
    bounds: SearchBounds,
    loss_config: LossConfig,
    holdout_ratio: float,
    fold_count: int,
    run_config: BacktestRunConfig,
) -> dict[str, JSONSerializable]:
    """Build the immutable identity record used to validate resumed studies."""
    if data.empty:
        raise ValueError("Optimization metadata requires at least one data row")
    _validate_finite_number(capital)
    _validate_finite_number(run_config.initial_capital)
    if coin != run_config.coin or capital != run_config.initial_capital:
        raise ValueError("Metadata coin and capital must match the backtest run configuration")

    metadata: dict[str, JSONSerializable] = {
        "schema": OPTIMIZER_SCHEMA_VERSION,
        "data": {
            "path": str(data_path.resolve()),
            "sha256": fingerprint_file(data_path),
            "rows": int(len(data)),
            "first_timestamp": pd.Timestamp(data.index[0]).isoformat(),
            "last_timestamp": pd.Timestamp(data.index[-1]).isoformat(),
        },
        "interval_minutes": int(interval_minutes),
        "coin": str(coin).upper(),
        "capital": float(capital),
        "bounds": _json_mapping(asdict(bounds)),
        "holdout_ratio": float(holdout_ratio),
        "fold_count": int(fold_count),
        "loss_config": _json_mapping(asdict(loss_config)),
        "risk_controls": {
            "drawdown_breaker_enabled": bool(run_config.drawdown_breaker_enabled),
            "loss_cooldown_enabled": bool(run_config.loss_cooldown_enabled),
        },
    }
    _validate_json_safe(metadata)
    return metadata


def create_or_load_study(
    storage_path: str | Path,
    study_name: str,
    metadata: dict[str, JSONSerializable],
    seed: int,
    startup_trials: int,
    parallel: bool,
) -> Study:
    """Open a SQLite-backed minimization study only when its metadata matches exactly."""
    _validate_json_safe(metadata)
    path = Path(storage_path)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=startup_trials,
        seed=seed,
        multivariate=True,
        constant_liar=parallel,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{path.resolve()}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    expected = dict(metadata)
    existing = study.user_attrs.get("optimizer_metadata")
    if existing is None:
        if study.trials:
            raise ValueError("Existing study metadata is missing and cannot be safely resumed")
        study.set_user_attr("optimizer_metadata", expected)
    elif existing != expected:
        raise ValueError("Existing study metadata does not match this optimization run")
    return study


def recover_stale_trials(study: Study) -> int:
    """Mark RUNNING trials from an interrupted coordinator as failed."""
    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for trial in running:
        study.tell(trial.number, state=TrialState.FAIL)
    return len(running)


def record_evaluation(trial: Trial, evaluation: TrialEvaluation) -> None:
    """Persist aggregate and per-fold evaluation evidence on one trial."""
    attributes: dict[str, JSONSerializable] = {
        "robust_annual_return_pct": float(evaluation.robust_annual_return_pct),
        "worst_drawdown_pct": float(evaluation.worst_drawdown_pct),
        "instability": float(evaluation.instability),
        "missing_round_trips": int(evaluation.missing_round_trips),
        "folds": [
            {
                "annual_return_pct": float(fold.annual_return_pct),
                "max_drawdown_pct": float(fold.max_drawdown_pct),
                "total_trades": int(fold.total_trades),
            }
            for fold in evaluation.folds
        ],
    }
    _validate_json_safe(float(evaluation.loss))
    _validate_json_safe(attributes)
    for key, value in attributes.items():
        trial.set_user_attr(key, value)


def _json_mapping(values: dict[str, object]) -> dict[str, JSONSerializable]:
    """Convert dataclass dictionaries to plain JSON-supported scalar values."""
    return {
        key: (
            float(value)
            if isinstance(value, float)
            else int(value) if isinstance(value, int) else value
        )
        for key, value in values.items()
    }


def _validate_finite_number(value: float) -> None:
    """Reject values that cannot be represented in strict JSON."""
    if not math.isfinite(value):
        raise ValueError("Persistent optimization values must be finite")


def _validate_json_safe(value: object) -> None:
    """Reject non-finite floats and values outside the supported JSON data model."""
    if isinstance(value, float):
        _validate_finite_number(value)
    elif isinstance(value, (str, int, bool)) or value is None:
        return
    elif isinstance(value, list):
        for item in value:
            _validate_json_safe(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Persistent optimization metadata keys must be strings")
            _validate_json_safe(item)
    else:
        raise ValueError("Persistent optimization values must be JSON serializable")
