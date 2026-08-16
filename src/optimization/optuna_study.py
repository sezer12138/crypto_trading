"""Persistent Optuna studies and auditable Bayesian trial metadata."""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import signal
import threading
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
    interrupted = False
    pending_error: BaseException | None = None
    try:
        while terminal_trial_count(study) < target_trials:
            trial = _ask_owned_trial(study)
            try:
                parameters = suggest_momentum_parameters(trial, bounds)
                evaluation = evaluator(parameters)
                record_evaluation(trial, evaluation)
                study.tell(trial, evaluation.loss)
            except BaseException as exc:
                interrupted = _mark_trial_failed(study, trial, exc)
                if isinstance(exc, KeyboardInterrupt) or interrupted:
                    interrupted = True
                    break
                if not isinstance(exc, Exception):
                    raise
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        pending_error = exc
    if pending_error is not None:
        raise pending_error
    return _run_summary(study, interrupted)


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
    pending_error: BaseException | None = None
    pool: ProcessPoolExecutor | None = None
    in_flight: dict[Future[TrialEvaluation], Trial] = {}
    try:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=initializer,
            initargs=initargs,
        )
        while terminal_trial_count(study) < target_trials or in_flight:
            interrupted = _refill_process_trials(
                pool,
                in_flight,
                study,
                target_trials,
                workers,
                bounds,
                evaluator,
            )
            if interrupted:
                break
            if not in_flight:
                continue
            try:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                interrupted = True
                break
            for future in finished:
                trial = in_flight[future]
                try:
                    result_interrupted = _finish_process_trial(study, trial, future)
                except BaseException:
                    in_flight.pop(future, None)
                    raise
                if result_interrupted:
                    interrupted = True
                    in_flight.pop(future, None)
                    break
                in_flight.pop(future, None)
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        pending_error = exc
    finally:
        aborting = interrupted or pending_error is not None
        if aborting:
            cleanup_error = _drain_interrupted_trials(study, in_flight)
            if pending_error is None:
                pending_error = cleanup_error
        if pool is not None:
            interrupted = _shutdown_process_pool(pool, aborting) or interrupted
    if pending_error is not None:
        raise pending_error
    return _run_summary(study, interrupted)


def _refill_process_trials(
    pool: ProcessPoolExecutor,
    in_flight: dict[Future[TrialEvaluation], Trial],
    study: Study,
    target_trials: int,
    workers: int,
    bounds: SearchBounds,
    evaluator: Callable[[dict[str, int | float]], TrialEvaluation],
) -> bool:
    """Keep bounded worker tasks without asking beyond the terminal target."""
    missing = target_trials - terminal_trial_count(study) - len(in_flight)
    for _ in range(min(workers - len(in_flight), missing)):
        trial = _ask_owned_trial(study)
        try:
            parameters = suggest_momentum_parameters(trial, bounds)
            future = pool.submit(evaluator, parameters)
            in_flight[future] = trial
        except BaseException as exc:
            interrupted = _mark_trial_failed(study, trial, exc)
            if isinstance(exc, KeyboardInterrupt) or interrupted:
                return True
            if not isinstance(exc, Exception):
                raise
    return False


def _drain_interrupted_trials(
    study: Study,
    in_flight: dict[Future[TrialEvaluation], Trial],
) -> BaseException | None:
    """Collect finished work and fail queued or running work without blocking."""
    pending_error: BaseException | None = None
    for future, trial in list(in_flight.items()):
        while True:
            try:
                if future.done():
                    _finish_process_trial(study, trial, future)
                elif future.cancel():
                    _mark_trial_failed(study, trial, RuntimeError("Cancelled after interrupt"))
                else:
                    _mark_trial_failed(study, trial, RuntimeError("Interrupted while running"))
            except KeyboardInterrupt:
                continue
            except BaseException as exc:
                if pending_error is None:
                    pending_error = exc
            in_flight.pop(future, None)
            break
    return pending_error


def _shutdown_process_pool(pool: ProcessPoolExecutor, nonblocking: bool) -> bool:
    """Shut down workers without letting repeated interrupts strand trial state."""
    interrupted = False
    while True:
        try:
            pool.shutdown(wait=not nonblocking, cancel_futures=nonblocking)
            return interrupted
        except KeyboardInterrupt:
            interrupted = True
            nonblocking = True


def _ask_owned_trial(study: Study) -> Trial:
    """Return an asked trial while deferring SIGINT until its ownership is locally established."""
    can_mask_sigint = hasattr(signal, "pthread_sigmask") and (
        threading.current_thread() is threading.main_thread()
    )
    if not can_mask_sigint:
        return study.ask()

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        trial = study.ask()
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except BaseException as exc:
        _mark_trial_failed(study, trial, exc)
        raise
    return trial


def _mark_trial_failed(study: Study, trial: Trial, error: BaseException) -> bool:
    """Make a trial terminal, retrying persistence if another interrupt arrives."""
    interrupted = False
    try:
        message = str(error)
    except BaseException:
        message = type(error).__name__
    while True:
        try:
            trial.set_user_attr("failure", message)
            break
        except KeyboardInterrupt:
            interrupted = True
        except Exception:
            break
    while True:
        try:
            study.tell(trial, state=TrialState.FAIL, skip_if_finished=True)
            return interrupted
        except KeyboardInterrupt:
            interrupted = True


def _finish_process_trial(
    study: Study,
    trial: Trial,
    future: Future[TrialEvaluation],
) -> bool:
    """Persist one worker result and terminalize every exceptional result."""
    try:
        evaluation = future.result()
        record_evaluation(trial, evaluation)
        study.tell(trial, evaluation.loss)
        return False
    except BaseException as exc:
        interrupted = _mark_trial_failed(study, trial, exc)
        if not isinstance(exc, Exception):
            if isinstance(exc, KeyboardInterrupt):
                return True
            raise
        if interrupted:
            return True
        return False


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
