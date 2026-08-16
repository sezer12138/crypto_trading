"""Tests for the Bayesian Momentum command-line orchestration."""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from scripts import bayesian_search_momentum as cli
from scripts.bayesian_search_momentum import main, parse_arguments, run_search
from optimization.momentum_objective import SearchBounds
from optimization.optuna_study import RunSummary, terminal_trial_count


def _write_synthetic_ohlcv(path: Path, rows: int = 600) -> None:
    """Write deterministic five-minute OHLCV rows with oscillating prices."""
    offsets = pd.to_timedelta(np.arange(rows) * 5, unit="min")
    regime_gaps = pd.to_timedelta(np.arange(rows) // 100, unit="D")
    timestamp = pd.Timestamp("2025-01-01") + offsets + regime_gaps
    phase = np.linspace(0.0, 24.0 * np.pi, rows)
    close = 100.0 + np.linspace(0.0, 8.0, rows) + 4.0 * np.sin(phase)
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )
    frame.to_csv(path, index=False)


def test_defaults_match_design() -> None:
    """Omitted tuning flags must retain the reproducible design defaults."""
    args = parse_arguments([])

    assert (args.trials, args.workers, args.startup_trials, args.seed) == (300, 1, 30, 42)
    assert (args.holdout_ratio, args.folds) == (0.20, 4)
    assert (args.drawdown_target, args.drawdown_weight, args.stability_weight) == (
        25.0,
        2.0,
        0.5,
    )
    assert (args.minimum_round_trips, args.missing_round_trip_penalty) == (2, 25.0)
    assert args.disable_drawdown_breaker is False
    assert args.disable_loss_cooldown is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--workers", "0"], "workers must be positive"),
        (["--trials", "0"], "trials must be positive"),
        (["--startup-trials", "0"], "startup trials must be positive"),
        (["--folds", "1"], "folds must be at least two"),
        (["--holdout-ratio", "1"], "holdout ratio must be between zero and one"),
        (["--capital", "nan"], "capital must be positive and finite"),
        (["--drawdown-weight", "inf"], "drawdown weight must be positive and finite"),
    ],
)
def test_main_rejects_invalid_configuration(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid numeric configuration must return two with one concise English error."""
    assert main(arguments) == 2

    error = capsys.readouterr().err.strip()
    assert message in error.lower()
    assert len(error.splitlines()) == 1


def test_main_rejects_missing_data(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A missing input file must fail before any study is created."""
    missing = tmp_path / "missing.csv"

    assert main(["--data", str(missing)]) == 2
    assert "data path does not exist" in capsys.readouterr().err.lower()


def test_real_sequential_resume_reaches_exact_total_and_writes_holdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resumed real SQLite run must add only the one missing terminal trial."""
    data_path = tmp_path / "synthetic.csv"
    storage = tmp_path / "nested" / "study.db"
    output_prefix = tmp_path / "reports" / "momentum"
    _write_synthetic_ohlcv(data_path)
    short_bounds = SearchBounds(2, 5, 2, 5, 0.005, 0.02)
    monkeypatch.setattr(cli, "resolve_search_bounds", lambda interval: short_bounds)
    common = [
        "--data",
        str(data_path),
        "--storage",
        str(storage),
        "--output-prefix",
        str(output_prefix),
        "--study-name",
        "resume-test",
        "--startup-trials",
        "2",
        "--folds",
        "2",
        "--workers",
        "1",
    ]

    first_paths = run_search(parse_arguments([*common, "--trials", "2"]))
    second_paths = run_search(parse_arguments([*common, "--trials", "3"]))

    study = optuna.load_study(study_name="resume-test", storage=f"sqlite:///{storage}")
    assert terminal_trial_count(study) == 3
    assert len(study.trials) == 3
    assert set(first_paths) == {"trials", "pareto", "importance", "summary", "report"}
    assert all(path.exists() and path.stat().st_size > 0 for path in second_paths.values())
    summary = json.loads(second_paths["summary"].read_text(encoding="utf-8"))
    assert summary["comparison"]["candidate_metrics"]
    assert summary["comparison"]["baseline_metrics"]


def test_interrupted_run_exports_partial_outputs_without_holdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl-C state must write auditable partial artifacts without touching holdout."""
    data_path = tmp_path / "synthetic.csv"
    _write_synthetic_ohlcv(data_path)
    monkeypatch.setattr(
        cli,
        "resolve_search_bounds",
        lambda interval: SearchBounds(2, 5, 2, 5, 0.005, 0.02),
    )

    def interrupt_runner(*args: object, **kwargs: object) -> RunSummary:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_sequential_trials", interrupt_runner)

    def reject_holdout(*args: object, **kwargs: object) -> object:
        raise AssertionError("holdout must not run after interruption")

    monkeypatch.setattr(cli, "compare_holdout", reject_holdout)
    arguments = [
        "--data",
        str(data_path),
        "--storage",
        str(tmp_path / "study.db"),
        "--output-prefix",
        str(tmp_path / "partial"),
        "--folds",
        "2",
        "--trials",
        "1",
    ]

    assert main(arguments) == 130
    summary = json.loads((tmp_path / "partial_summary.json").read_text(encoding="utf-8"))
    assert summary["comparison"] == {}
