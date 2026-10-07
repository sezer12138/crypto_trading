"""Tests for persistent Optuna study safety and trial metadata."""

from concurrent.futures import Future, wait as futures_wait
import json
import signal
import sys
from dataclasses import replace
from pathlib import Path

import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.momentum_evaluator import BacktestRunConfig
from optimization.momentum_objective import FoldMetrics, LossConfig, SearchBounds, TrialEvaluation
import optimization.optuna_study as optuna_study_module
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


def _process_type_error(parameters: dict[str, int | float]) -> TrialEvaluation:
    """Raise an ordinary evaluator exception in a real worker process."""
    raise TypeError(f"bad parameter {parameters['buy_roc_period']}")


def _interrupt_when_sigint_handler_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject Ctrl-C while the temporary SIGINT handler is being restored."""
    calls = 0
    previous_handler = signal.getsignal(signal.SIGINT)

    def set_signal_handler(signal_number: int, handler: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return previous_handler
        if calls == 2:
            raise KeyboardInterrupt
        return previous_handler

    monkeypatch.setattr(signal, "signal", set_signal_handler)


def _interrupt_after_sigint_handler_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject Ctrl-C on the first line after successful handler restoration."""
    signal_function = signal.signal
    previous_trace = sys.gettrace()
    ask_code = optuna_study_module._ask_owned_trial.__code__
    signal_calls = 0
    line_events = 0

    def interrupt_after_restore(frame: object, event: str, arg: object) -> object:
        nonlocal line_events
        if getattr(frame, "f_code", None) is ask_code and event == "line":
            line_events += 1
            if line_events == 2:
                sys.settrace(previous_trace)
                frame.f_trace = previous_trace  # type: ignore[attr-defined]
                raise KeyboardInterrupt
        return interrupt_after_restore

    def set_signal_handler(signal_number: int, handler: object) -> object:
        nonlocal signal_calls
        signal_calls += 1
        previous_handler = signal_function(signal_number, handler)
        if signal_calls == 2:
            caller = sys._getframe(1)
            caller.f_trace = interrupt_after_restore
            sys.settrace(interrupt_after_restore)
        return previous_handler

    monkeypatch.setattr(signal, "signal", set_signal_handler)


def _interrupt_when_sigint_mask_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject Ctrl-C at the legacy POSIX signal-mask restoration boundary."""
    calls = 0

    def pthread_sigmask(how: int, mask: object) -> set[signal.Signals]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return set()
        raise KeyboardInterrupt

    monkeypatch.setattr(signal, "pthread_sigmask", pthread_sigmask, raising=False)


def _deliver_sigint_before_ask_returns(
    study: optuna.study.Study,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliver Ctrl-C after one owned and one unrelated trial have been persisted."""
    active_handler: object = signal.default_int_handler

    def get_signal_handler(signal_number: int) -> object:
        return active_handler

    def set_signal_handler(signal_number: int, handler: object) -> object:
        nonlocal active_handler
        previous_handler = active_handler
        active_handler = handler
        return previous_handler

    ask = study.ask

    def ask_then_sigint() -> optuna.trial.Trial:
        trial = ask()
        ask()
        assert callable(active_handler)
        active_handler(signal.SIGINT, None)
        return trial

    monkeypatch.setattr(signal, "getsignal", get_signal_handler)
    monkeypatch.setattr(signal, "signal", set_signal_handler)
    monkeypatch.setattr(study, "ask", ask_then_sigint)


def _interrupt_first_trial_failure_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject one nested Ctrl-C while a trial failure is being recorded."""
    set_user_attr = optuna.trial.Trial.set_user_attr
    writes = 0

    def interrupt_first_write(trial: optuna.trial.Trial, key: str, value: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise KeyboardInterrupt
        set_user_attr(trial, key, value)

    monkeypatch.setattr(optuna.trial.Trial, "set_user_attr", interrupt_first_write)


class _ExecutorDouble:
    """Provide controlled submit outcomes while keeping real Optuna storage in tests."""

    def __init__(self, outcomes: list[Future[TrialEvaluation] | BaseException]) -> None:
        self.outcomes = outcomes

    def __enter__(self) -> "_ExecutorDouble":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(
        self,
        evaluator: object,
        parameters: dict[str, int | float],
    ) -> Future[TrialEvaluation]:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


class _InterruptingResultFuture(Future[TrialEvaluation]):
    """Simulate a second Ctrl-C while the interrupted coordinator drains results."""

    def result(self, timeout: float | None = None) -> TrialEvaluation:
        raise KeyboardInterrupt


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


def test_sequential_suggestion_failure_terminalizes_each_trial(tmp_path: Path) -> None:
    """Prevents invalid search bounds from leaving ask-created trials running."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    invalid_bounds = SearchBounds(12, 12, 6, 6, 0.02, 0.01)

    summary = run_sequential_trials(study, 2, invalid_bounds, _process_evaluator)

    assert terminal_trial_count(study) == 2
    assert summary.completed == 0
    assert summary.failed == 2


def test_sequential_system_exit_survives_interrupt_during_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves evaluator SystemExit when failure persistence also sees Ctrl-C."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    _interrupt_first_trial_failure_write(monkeypatch)

    def stop_evaluation(parameters: dict[str, int | float]) -> TrialEvaluation:
        raise SystemExit("evaluation stopped")

    with pytest.raises(SystemExit, match="evaluation stopped"):
        run_sequential_trials(study, 1, _fixed_bounds(), stop_evaluation)

    assert terminal_trial_count(study) == 1
    assert study.trials[0].state is optuna.trial.TrialState.FAIL


def test_sequential_interrupt_after_ask_terminalizes_the_created_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovers a trial persisted just before Ctrl-C prevents ask from returning to the runner."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    monkeypatch.delattr(signal, "pthread_sigmask", raising=False)
    _deliver_sigint_before_ask_returns(study, monkeypatch)

    summary = run_sequential_trials(study, 2, _fixed_bounds(), _process_evaluator)

    assert len(study.trials) == 2
    assert terminal_trial_count(study) == 1
    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.interrupted is True
    assert study.trials[0].state is optuna.trial.TrialState.FAIL
    assert study.trials[1].state is optuna.trial.TrialState.RUNNING


def test_sequential_interrupt_after_handler_restore_terminalizes_handed_off_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeps exact ownership when Ctrl-C lands after restoration but before return."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    study.ask()
    previous_trace = sys.gettrace()
    _interrupt_after_sigint_handler_is_restored(monkeypatch)

    try:
        summary = run_sequential_trials(study, 2, _fixed_bounds(), _process_evaluator)
    finally:
        sys.settrace(previous_trace)

    assert len(study.trials) == 2
    assert terminal_trial_count(study) == 1
    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.interrupted is True
    assert study.trials[0].state is optuna.trial.TrialState.RUNNING
    assert study.trials[1].state is optuna.trial.TrialState.FAIL


def test_sequential_ask_error_survives_interrupt_during_signal_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeps an ask SystemExit primary when SIGINT arrives during handler restoration."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, False)
    _interrupt_when_sigint_mask_is_restored(monkeypatch)
    _interrupt_when_sigint_handler_is_restored(monkeypatch)

    def stop_asking() -> optuna.trial.Trial:
        raise SystemExit("ask stopped")

    monkeypatch.setattr(study, "ask", stop_asking)
    raised: BaseException | None = None

    try:
        run_sequential_trials(study, 1, _fixed_bounds(), _process_evaluator)
    except BaseException as exc:
        raised = exc

    assert isinstance(raised, SystemExit)
    assert str(raised) == "ask stopped"
    assert study.trials == []


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


def test_process_evaluator_exception_terminalizes_every_trial(tmp_path: Path) -> None:
    """Converts arbitrary exceptions returned by worker futures into terminal failures."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)

    summary = run_process_trials(
        study,
        2,
        2,
        _fixed_bounds(),
        None,
        (),
        _process_type_error,
    )

    assert terminal_trial_count(study) == 2
    assert summary.completed == 0
    assert summary.failed == 2
    assert all(trial.user_attrs["failure"].startswith("bad parameter") for trial in study.trials)


def test_process_submit_failure_terminalizes_each_asked_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeps every trial terminal when the process executor rejects submissions."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    executor = _ExecutorDouble([OSError("submit failed"), OSError("submit failed")])
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)

    summary = run_process_trials(
        study,
        2,
        1,
        _fixed_bounds(),
        None,
        (),
        _process_evaluator,
    )

    assert terminal_trial_count(study) == 2
    assert summary.completed == 0
    assert summary.failed == 2
    assert all(trial.user_attrs["failure"] == "submit failed" for trial in study.trials)


@pytest.mark.parametrize("failure_stage", ["suggestion", "submission"])
def test_process_setup_system_exit_survives_interrupt_during_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Preserves setup SystemExit when failure persistence also sees Ctrl-C."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    outcomes = [] if failure_stage == "suggestion" else [SystemExit("setup stopped")]
    executor = _ExecutorDouble(outcomes)
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)
    if failure_stage == "suggestion":

        def stop_suggestion(trial: optuna.trial.Trial, bounds: SearchBounds) -> None:
            raise SystemExit("setup stopped")

        monkeypatch.setattr(
            optuna_study_module,
            "suggest_momentum_parameters",
            stop_suggestion,
        )
    _interrupt_first_trial_failure_write(monkeypatch)

    with pytest.raises(SystemExit, match="setup stopped"):
        run_process_trials(
            study,
            1,
            1,
            _fixed_bounds(),
            None,
            (),
            _process_evaluator,
        )

    assert terminal_trial_count(study) == 1
    assert study.trials[0].state is optuna.trial.TrialState.FAIL


def test_repeated_process_interrupt_stops_refill_and_terminalizes_all_asked_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Survives repeated Ctrl-C while collecting finished and cancelling queued work."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    completed: Future[TrialEvaluation] = Future()
    completed.set_result(_evaluation(1.0))
    interrupted = _InterruptingResultFuture()
    interrupted.set_result(_evaluation(2.0))
    queued: Future[TrialEvaluation] = Future()
    executor = _ExecutorDouble([completed, interrupted, queued])
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)
    waits = 0

    def interrupt_once(*args: object, **kwargs: object) -> object:
        nonlocal waits
        waits += 1
        if waits == 1:
            raise KeyboardInterrupt
        return futures_wait(*args, **kwargs)

    monkeypatch.setattr(optuna_study_module, "wait", interrupt_once)

    summary = run_process_trials(
        study,
        5,
        3,
        _fixed_bounds(),
        None,
        (),
        _process_evaluator,
    )

    assert len(study.trials) == 3
    assert terminal_trial_count(study) == 3
    assert summary.completed == 1
    assert summary.failed == 2
    assert summary.interrupted is True


def test_process_system_exit_drains_owned_peer_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminalizes peer trials before preserving a worker SystemExit."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    fatal: Future[TrialEvaluation] = Future()
    fatal.set_exception(SystemExit("worker stopped"))
    peer: Future[TrialEvaluation] = Future()
    executor = _ExecutorDouble([fatal, peer])
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)
    monkeypatch.setattr(optuna_study_module, "wait", lambda *args, **kwargs: ({fatal}, {peer}))

    with pytest.raises(SystemExit, match="worker stopped"):
        run_process_trials(
            study,
            2,
            2,
            _fixed_bounds(),
            None,
            (),
            _process_evaluator,
        )

    assert len(study.trials) == 2
    assert terminal_trial_count(study) == 2
    assert all(trial.state is optuna.trial.TrialState.FAIL for trial in study.trials)


def test_process_system_exit_survives_interrupt_during_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves worker SystemExit after Ctrl-C retries its failure persistence."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    fatal: Future[TrialEvaluation] = Future()
    fatal.set_exception(SystemExit("worker stopped"))
    executor = _ExecutorDouble([fatal])
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)
    monkeypatch.setattr(optuna_study_module, "wait", lambda *args, **kwargs: ({fatal}, set()))
    _interrupt_first_trial_failure_write(monkeypatch)

    with pytest.raises(SystemExit, match="worker stopped"):
        run_process_trials(
            study,
            1,
            1,
            _fixed_bounds(),
            None,
            (),
            _process_evaluator,
        )

    assert terminal_trial_count(study) == 1
    assert study.trials[0].state is optuna.trial.TrialState.FAIL


def test_process_interrupt_after_ask_terminalizes_the_created_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovers a persisted trial when Ctrl-C lands before process submission owns it."""
    study = create_or_load_study(tmp_path / "study.db", "momentum", {}, 42, 5, True)
    monkeypatch.delattr(signal, "pthread_sigmask", raising=False)
    _deliver_sigint_before_ask_returns(study, monkeypatch)
    executor = _ExecutorDouble([])
    monkeypatch.setattr(optuna_study_module, "ProcessPoolExecutor", lambda **kwargs: executor)

    summary = run_process_trials(
        study,
        2,
        1,
        _fixed_bounds(),
        None,
        (),
        _process_evaluator,
    )

    assert len(study.trials) == 2
    assert terminal_trial_count(study) == 1
    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.interrupted is True
    assert study.trials[0].state is optuna.trial.TrialState.FAIL
    assert study.trials[1].state is optuna.trial.TrialState.RUNNING


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
        "execution_model": "next_open_intrabar_v2",
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
