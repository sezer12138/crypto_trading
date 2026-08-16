"""Tests for Bayesian Momentum holdout comparison and report artifacts."""

import csv
import json
import sys
from pathlib import Path

import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optimization.momentum_report as momentum_report
from optimization.momentum_evaluator import BacktestRunConfig
from optimization.momentum_report import (
    adoption_checks,
    compare_holdout,
    write_optimization_outputs,
)

PARAMETERS = {
    "buy_roc_period": 12,
    "buy_momentum_period": 6,
    "buy_threshold": 0.01,
    "sell_roc_period": 24,
    "sell_momentum_period": 12,
    "sell_threshold": 0.02,
}


def _record_trial(
    study: optuna.Study,
    value: float,
    robust_return: float,
    drawdown: float,
    missing_round_trips: int,
    offset: int = 0,
) -> None:
    """Record one realistic completed Optuna trial with fold evidence."""
    study.enqueue_trial(PARAMETERS)
    trial = study.ask()
    trial.suggest_int("buy_roc_period", 12, 120)
    trial.suggest_int("buy_momentum_period", 6, 60)
    trial.suggest_float("buy_threshold", 0.005, 0.25, log=True)
    trial.suggest_int("sell_roc_period", 12, 120)
    trial.suggest_int("sell_momentum_period", 6, 60)
    trial.suggest_float("sell_threshold", 0.005, 0.25, log=True)
    trial.set_user_attr("robust_annual_return_pct", robust_return)
    trial.set_user_attr("worst_drawdown_pct", drawdown)
    trial.set_user_attr("instability", 2.5 + offset)
    trial.set_user_attr("missing_round_trips", missing_round_trips)
    trial.set_user_attr(
        "folds",
        [
            {
                "annual_return_pct": robust_return - 1.0,
                "max_drawdown_pct": -drawdown,
                "total_trades": 4,
            },
            {
                "annual_return_pct": robust_return + 1.0,
                "max_drawdown_pct": -(drawdown - 1.0),
                "total_trades": 6,
            },
        ],
    )
    study.tell(trial, value)


def _study() -> optuna.Study:
    """Return completed, inactive, dominated, and failed trials for artifact tests."""
    study = optuna.create_study(direction="minimize", study_name="momentum<&>")
    _record_trial(study, 4.0, 12.0, 10.0, 0)
    _record_trial(study, 3.0, 18.0, 15.0, 0, offset=1)
    _record_trial(study, 2.0, 11.0, 20.0, 0, offset=2)
    _record_trial(study, 1.0, 30.0, 8.0, 1, offset=3)
    failed = study.ask()
    failed.set_user_attr("failure", "bad <trial>")
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    return study


def _metadata() -> dict[str, object]:
    """Return complete report configuration with boundary-resolved search bounds."""
    return {
        "schema": 1,
        "coin": "BTC<&>",
        "interval_minutes": 5,
        "capital": 10_000.0,
        "fold_count": 2,
        "holdout_ratio": 0.2,
        "bounds": {
            "roc_min": 12,
            "roc_max": 8064,
            "momentum_min": 6,
            "momentum_max": 4032,
            "threshold_min": 0.005,
            "threshold_max": 0.25,
        },
        "loss_config": {
            "drawdown_target_pct": 25.0,
            "drawdown_weight": 2.0,
            "stability_weight": 0.5,
            "minimum_round_trips": 2,
            "missing_round_trip_penalty": 25.0,
        },
        "risk_controls": {
            "drawdown_breaker_enabled": False,
            "loss_cooldown_enabled": False,
        },
    }


def _comparison() -> dict[str, object]:
    """Return selected parameters and complete two-sided holdout evidence."""
    return {
        "candidate_parameters": PARAMETERS,
        "baseline_parameters": {**PARAMETERS, "buy_roc_period": 16},
        "candidate_metrics": {
            "total_return_pct": 20.0,
            "annual_return_pct": 22.0,
            "sharpe_ratio": 1.1,
            "max_drawdown_pct": -12.0,
            "win_rate_pct": 55.0,
            "total_trades": 6,
        },
        "baseline_metrics": {
            "total_return_pct": 10.0,
            "annual_return_pct": 11.0,
            "sharpe_ratio": 0.7,
            "max_drawdown_pct": -14.0,
            "win_rate_pct": 50.0,
            "total_trades": 4,
        },
        "adoption_checks": {
            "beats_baseline_return": True,
            "minimum_round_trips": True,
            "maximum_drawdown": True,
            "passed": True,
        },
    }


def test_adoption_checks_names_each_gate() -> None:
    """A failed drawdown gate must prevent an otherwise adoptable candidate."""
    assert adoption_checks(
        {"total_return_pct": 20.0, "max_drawdown_pct": -31.0, "total_trades": 8},
        {"total_return_pct": 10.0},
    ) == {
        "beats_baseline_return": True,
        "minimum_round_trips": True,
        "maximum_drawdown": False,
        "passed": False,
    }


def test_adoption_checks_include_exact_drawdown_and_activity_boundaries() -> None:
    """Exactly two round trips and thirty percent absolute drawdown pass the guard."""
    checks = adoption_checks(
        {"total_return_pct": 10.1, "max_drawdown_pct": -30.0, "total_trades": 4},
        {"total_return_pct": 10.0},
    )

    assert checks == {
        "beats_baseline_return": True,
        "minimum_round_trips": True,
        "maximum_drawdown": True,
        "passed": True,
    }


def test_compare_holdout_runs_candidate_then_baseline_exactly_once(monkeypatch) -> None:
    """Holdout evidence must consist of exactly the selected and baseline runs."""
    calls: list[dict[str, int | float]] = []

    def fake_run_holdout(
        frame: pd.DataFrame,
        parameters: dict[str, int | float],
        config: BacktestRunConfig,
    ) -> dict[str, float]:
        calls.append(dict(parameters))
        return {
            "total_return_pct": float(len(calls) * 10),
            "max_drawdown_pct": -10.0,
            "total_trades": 4.0,
        }

    monkeypatch.setattr(momentum_report, "run_holdout", fake_run_holdout)
    baseline = {**PARAMETERS, "buy_roc_period": 16}

    comparison = compare_holdout(
        pd.DataFrame({"close": [100.0]}),
        PARAMETERS,
        baseline,
        BacktestRunConfig(10_000.0, "BTC", False, False),
    )

    assert calls == [PARAMETERS, baseline]
    assert comparison["candidate_metrics"]["total_return_pct"] == 10.0
    assert comparison["baseline_metrics"]["total_return_pct"] == 20.0
    assert comparison["adoption_checks"]["passed"] is False


def test_compare_holdout_serializes_general_defaults_for_empty_baseline(monkeypatch) -> None:
    """An absent interval profile must use all six general Momentum defaults."""
    calls: list[dict[str, int | float]] = []

    def fake_run_holdout(*args: object) -> dict[str, float]:
        calls.append(dict(args[1]))
        return {"total_return_pct": 1.0, "max_drawdown_pct": -1.0, "total_trades": 4.0}

    monkeypatch.setattr(momentum_report, "run_holdout", fake_run_holdout)

    comparison = compare_holdout(
        pd.DataFrame({"close": [100.0]}),
        PARAMETERS,
        {},
        BacktestRunConfig(10_000.0, "BTC", False, False),
    )

    assert set(calls[1]) == set(PARAMETERS)
    assert comparison["baseline_parameters"] == calls[1]


def test_outputs_are_complete_auditable_and_standalone(tmp_path: Path) -> None:
    """All five outputs preserve terminal trials, selection, and escaped report evidence."""
    paths = write_optimization_outputs(
        _study(), tmp_path / "nested" / "momentum", _metadata(), _comparison()
    )

    assert set(paths) == {"trials", "pareto", "importance", "summary", "report"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())

    with paths["trials"].open(encoding="utf-8", newline="") as handle:
        trial_rows = list(csv.DictReader(handle))
    assert [row["state"] for row in trial_rows] == [
        "COMPLETE",
        "COMPLETE",
        "COMPLETE",
        "COMPLETE",
        "FAIL",
    ]
    assert trial_rows[0]["param_buy_roc_period"]
    assert trial_rows[0]["fold_1_annual_return_pct"] == "11.0"
    assert trial_rows[-1]["failure"] == "bad <trial>"

    with paths["pareto"].open(encoding="utf-8", newline="") as handle:
        pareto_rows = list(csv.DictReader(handle))
    assert [row["trial"] for row in pareto_rows] == ["0", "1"]
    assert all(row["missing_round_trips"] == "0" for row in pareto_rows)

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["metadata"] == _metadata()
    assert summary["selected_trial"] == 3
    assert summary["selected_parameters"]
    assert summary["comparison"] == _comparison()
    assert summary["trial_counts"] == {"COMPLETE": 4, "FAIL": 1}
    assert summary["boundary_warnings"]

    html = paths["report"].read_text(encoding="utf-8")
    assert "<svg" in html and "https://" not in html and "http://" not in html
    assert "Composite Loss" in html
    assert "robust_return = median(A_i)" in html
    assert "instability = population_stddev(A_i)" in html
    assert "T_i = floor(total_trades_i / 2)" in html
    assert "Top 25 Trials" in html
    assert "Pareto Front" in html
    assert "Fold Metrics" in html
    assert "Holdout Comparison" in html
    assert "Adoption Gates" in html
    assert "Boundary Warnings" in html
    assert "momentum&lt;&amp;&gt;" in html
    assert "BTC&lt;&amp;&gt;" in html
    assert "bad &lt;trial&gt;" in html
    assert "momentum<&>" not in html
    assert "BTC<&>" not in html
    assert "bad <trial>" not in html


def test_importance_failure_writes_headers_warning_and_report(monkeypatch, tmp_path: Path) -> None:
    """Importance analysis failure must not prevent auditable artifact generation."""

    def fail_importance(study: optuna.Study) -> dict[str, float]:
        raise RuntimeError("importance <unavailable>")

    monkeypatch.setattr(momentum_report, "get_param_importances", fail_importance)

    paths = write_optimization_outputs(_study(), tmp_path / "momentum", _metadata(), _comparison())

    with paths["importance"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert handle.seek(0) == 0
        header = handle.readline().strip()
    assert header == "parameter,importance,warning"
    assert rows == [
        {
            "parameter": "",
            "importance": "",
            "warning": "Parameter importance unavailable: importance <unavailable>",
        }
    ]
    html = paths["report"].read_text(encoding="utf-8")
    assert "importance &lt;unavailable&gt;" in html
    assert "importance <unavailable>" not in html


def test_partial_study_still_writes_all_five_artifacts(tmp_path: Path) -> None:
    """Interrupted studies without a completed trial remain exportable and explicit."""
    study = optuna.create_study(direction="minimize")
    failed = study.ask()
    failed.set_user_attr("failure", "worker stopped")
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    paths = write_optimization_outputs(study, tmp_path / "partial", _metadata(), {})

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["selected_trial"] is None
    assert summary["selected_parameters"] is None
    assert summary["comparison"] == {}
    with paths["pareto"].open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
