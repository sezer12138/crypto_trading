"""Persistent Optuna studies and auditable Bayesian trial metadata."""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, TypeAlias

import optuna
import pandas as pd
from optuna.study import Study
from optuna.trial import Trial, TrialState

from optimization.data import fingerprint_file
from optimization.momentum_evaluator import BacktestRunConfig
from optimization.momentum_objective import (
    LossConfig,
    SearchBounds,
    TrialEvaluation,
    suggest_momentum_parameters,
)

JSONSerializable: TypeAlias = (
    str | int | float | bool | None | list["JSONSerializable"] | dict[str, "JSONSerializable"]
)

OPTIMIZER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunSummary:
    """Terminal study counts and whether scheduling stopped on interruption."""

    completed: int
    failed: int
    interrupted: bool


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


def terminal_trial_count(study: Study) -> int:
    """Return the number of complete or failed trials in a study."""
    return len(
        study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.FAIL),
        )
    )


def run_sequential_trials(
    study: Study,
    target_trials: int,
    bounds: SearchBounds,
    evaluator: Callable[[dict[str, int | float]], TrialEvaluation],
) -> RunSummary:
    """Evaluate trials sequentially until the study reaches the terminal target."""
    while terminal_trial_count(study) < target_trials:
        trial = study.ask()
        parameters = suggest_momentum_parameters(trial, bounds)
        try:
            evaluation = evaluator(parameters)
            record_evaluation(trial, evaluation)
            study.tell(trial, evaluation.loss)
        except (ValueError, RuntimeError) as exc:
            trial.set_user_attr("failure", str(exc))
            study.tell(trial, state=TrialState.FAIL)
    return _run_summary(study, interrupted=False)


def run_process_trials(
    study: Study,
    target_trials: int,
    workers: int,
    bounds: SearchBounds,
    initializer: Callable[..., None] | None,
    initargs: tuple[object, ...],
    evaluator: Callable[[dict[str, int | float]], TrialEvaluation],
) -> RunSummary:
    """Evaluate parameters in worker processes while coordinating Optuna in the parent."""
    interrupted = False
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        in_flight: dict[Future[TrialEvaluation], Trial] = {}
        _refill_process_trials(pool, in_flight, study, target_trials, workers, bounds, evaluator)
        while in_flight:
            try:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                interrupted = True
                _cancel_pending_trials(study, in_flight)
                finished, _ = wait(in_flight)
            for future in finished:
                trial = in_flight.pop(future)
                _finish_process_trial(study, trial, future)
            if not interrupted:
                _refill_process_trials(
                    pool,
                    in_flight,
                    study,
                    target_trials,
                    workers,
                    bounds,
                    evaluator,
                )
    return _run_summary(study, interrupted)


def _refill_process_trials(
    pool: ProcessPoolExecutor,
    in_flight: dict[Future[TrialEvaluation], Trial],
    study: Study,
    target_trials: int,
    workers: int,
    bounds: SearchBounds,
    evaluator: Callable[[dict[str, int | float]], TrialEvaluation],
) -> None:
    """Keep bounded worker tasks without asking beyond the terminal target."""
    missing = target_trials - terminal_trial_count(study) - len(in_flight)
    for _ in range(min(workers - len(in_flight), missing)):
        trial = study.ask()
        parameters = suggest_momentum_parameters(trial, bounds)
        in_flight[pool.submit(evaluator, parameters)] = trial


def _cancel_pending_trials(
    study: Study,
    in_flight: dict[Future[TrialEvaluation], Trial],
) -> None:
    """Cancel queued evaluations and make their corresponding trials terminal."""
    for future, trial in list(in_flight.items()):
        if future.cancel():
            trial.set_user_attr("failure", "Cancelled after KeyboardInterrupt")
            study.tell(trial, state=TrialState.FAIL)
            del in_flight[future]


def _finish_process_trial(
    study: Study,
    trial: Trial,
    future: Future[TrialEvaluation],
) -> None:
    """Persist one worker result, converting supported evaluation errors to failures."""
    try:
        evaluation = future.result()
        record_evaluation(trial, evaluation)
        study.tell(trial, evaluation.loss)
    except (ValueError, RuntimeError) as exc:
        trial.set_user_attr("failure", str(exc))
        study.tell(trial, state=TrialState.FAIL)


def _run_summary(study: Study, interrupted: bool) -> RunSummary:
    """Snapshot complete and failed trial counts from persistent storage."""
    completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    failed = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))
    return RunSummary(completed, failed, interrupted)


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
