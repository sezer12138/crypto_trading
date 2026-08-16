"""Persistent Optuna studies and auditable Bayesian trial metadata."""

from dataclasses import asdict
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

    return {
        "schema": OPTIMIZER_SCHEMA_VERSION,
        "data": {
            "path": str(data_path.resolve()),
            "sha256": fingerprint_file(data_path),
            "rows": int(len(data)),
            "first_timestamp": pd.Timestamp(data.index[0]).isoformat(),
            "last_timestamp": pd.Timestamp(data.index[-1]).isoformat(),
        },
        "interval_minutes": int(interval_minutes),
        "coin": str(coin),
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


def create_or_load_study(
    storage_path: str | Path,
    study_name: str,
    metadata: dict[str, JSONSerializable],
    seed: int,
    startup_trials: int,
    parallel: bool,
) -> Study:
    """Open a SQLite-backed minimization study only when its metadata matches exactly."""
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
    trial.set_user_attr("robust_annual_return_pct", float(evaluation.robust_annual_return_pct))
    trial.set_user_attr("worst_drawdown_pct", float(evaluation.worst_drawdown_pct))
    trial.set_user_attr("instability", float(evaluation.instability))
    trial.set_user_attr("missing_round_trips", int(evaluation.missing_round_trips))
    trial.set_user_attr(
        "folds",
        [
            {
                "annual_return_pct": float(fold.annual_return_pct),
                "max_drawdown_pct": float(fold.max_drawdown_pct),
                "total_trades": int(fold.total_trades),
            }
            for fold in evaluation.folds
        ],
    )


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
