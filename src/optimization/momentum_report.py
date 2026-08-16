"""Auditable holdout comparison and reports for Bayesian Momentum optimization."""

import csv
from datetime import date, datetime
from enum import Enum
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from optuna.importance import get_param_importances
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from optimization.momentum_evaluator import BacktestRunConfig, run_holdout
from optimization.momentum_objective import pareto_front
from strategies.momentum import MomentumStrategy

PARAMETER_NAMES = (
    "buy_roc_period",
    "buy_momentum_period",
    "buy_threshold",
    "sell_roc_period",
    "sell_momentum_period",
    "sell_threshold",
)
AGGREGATE_NAMES = (
    "robust_annual_return_pct",
    "worst_drawdown_pct",
    "instability",
    "missing_round_trips",
)
FOLD_METRIC_NAMES = ("annual_return_pct", "max_drawdown_pct", "total_trades")
IMPORTANCE_FIELDS = ("parameter", "importance", "warning")
LOSS_FORMULA = (
    "loss = -robust_return + drawdown_weight * max(0, worst_drawdown - "
    "drawdown_target) + stability_weight * instability + "
    "missing_round_trip_penalty * missing_round_trips"
)
LOSS_DEFINITIONS = "\n".join(
    (
        "T_i = floor(total_trades_i / 2)",
        "robust_return = median(A_i)",
        "worst_drawdown = max(abs(M_i))",
        "instability = population_stddev(A_i)",
        "missing_round_trips = sum(max(0, minimum_round_trips - T_i))",
    )
)


def adoption_checks(
    candidate_metrics: Mapping[str, float], baseline_metrics: Mapping[str, float]
) -> dict[str, bool]:
    """Return each holdout adoption gate and their conjunction."""
    checks = {
        "beats_baseline_return": (
            float(candidate_metrics["total_return_pct"])
            > float(baseline_metrics["total_return_pct"])
        ),
        "minimum_round_trips": int(candidate_metrics["total_trades"]) // 2 >= 2,
        "maximum_drawdown": abs(float(candidate_metrics["max_drawdown_pct"])) <= 30.0,
    }
    return {**checks, "passed": all(checks.values())}


def compare_holdout(
    holdout: pd.DataFrame,
    candidate: Mapping[str, int | float],
    baseline: Mapping[str, int | float],
    config: BacktestRunConfig,
) -> dict[str, Any]:
    """Run the selected candidate and baseline once each on the untouched holdout."""
    candidate_parameters = dict(candidate)
    baseline_parameters = dict(baseline) or _general_momentum_parameters()
    candidate_metrics = run_holdout(holdout, candidate_parameters, config)
    baseline_metrics = run_holdout(holdout, baseline_parameters, config)
    return {
        "candidate_parameters": candidate_parameters,
        "baseline_parameters": baseline_parameters,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "adoption_checks": adoption_checks(candidate_metrics, baseline_metrics),
    }


def write_optimization_outputs(
    study: Study,
    prefix: str | Path,
    metadata: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
) -> dict[str, Path]:
    """Write trials, Pareto, importance, summary, and standalone HTML artifacts."""
    output_prefix = Path(prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "trials": Path(f"{output_prefix}_trials.csv"),
        "pareto": Path(f"{output_prefix}_pareto.csv"),
        "importance": Path(f"{output_prefix}_importance.csv"),
        "summary": Path(f"{output_prefix}_summary.json"),
        "report": Path(f"{output_prefix}_report.html"),
    }

    fold_count = _fold_count(study, metadata)
    trial_fields = _trial_fields(fold_count)
    terminal_trials = _terminal_trials(study)
    trial_rows = [_flatten_trial(trial, fold_count) for trial in terminal_trials]
    pareto_rows = _pareto_rows(trial_rows)
    importance_rows, importance_warning = _importance_rows(study)
    selected = _selected_trial(terminal_trials)
    boundary_warnings = _boundary_warnings(selected.params if selected else {}, metadata)
    comparison_data = dict(comparison or {})
    summary = _summary(
        study,
        metadata,
        comparison_data,
        terminal_trials,
        selected,
        boundary_warnings,
        importance_warning,
    )

    _write_csv(paths["trials"], trial_fields, trial_rows)
    _write_csv(paths["pareto"], trial_fields, pareto_rows)
    _write_csv(paths["importance"], IMPORTANCE_FIELDS, importance_rows)
    paths["summary"].write_text(
        json.dumps(_json_compatible(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        _render_report(
            study,
            metadata,
            comparison_data,
            trial_rows,
            pareto_rows,
            importance_rows,
            importance_warning,
            selected,
            boundary_warnings,
        ),
        encoding="utf-8",
    )
    return paths


def _general_momentum_parameters() -> dict[str, int | float]:
    """Serialize all constructor defaults from the general Momentum strategy."""
    strategy = MomentumStrategy()
    return {name: getattr(strategy, name) for name in PARAMETER_NAMES}


def _terminal_trials(study: Study) -> list[FrozenTrial]:
    """Return complete and failed trials in stable trial-number order."""
    trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE, TrialState.FAIL),
    )
    return sorted(trials, key=lambda trial: trial.number)


def _fold_count(study: Study, metadata: Mapping[str, Any]) -> int:
    """Resolve a stable fold schema from configuration and recorded evidence."""
    configured = int(metadata.get("fold_count", 0))
    recorded = max(
        (
            len(trial.user_attrs.get("folds", []))
            for trial in study.get_trials(deepcopy=False)
            if isinstance(trial.user_attrs.get("folds", []), Sequence)
        ),
        default=0,
    )
    return max(configured, recorded)


def _trial_fields(fold_count: int) -> tuple[str, ...]:
    """Return the deterministic flattened trial CSV schema."""
    base = (
        "trial",
        "state",
        "loss",
        "duration_seconds",
        "datetime_start",
        "datetime_complete",
        *(f"param_{name}" for name in PARAMETER_NAMES),
        *AGGREGATE_NAMES,
        "failure",
    )
    folds = tuple(
        f"fold_{fold_number}_{metric}"
        for fold_number in range(1, fold_count + 1)
        for metric in FOLD_METRIC_NAMES
    )
    return (*base, *folds)


def _flatten_trial(trial: FrozenTrial, fold_count: int) -> dict[str, Any]:
    """Flatten one Optuna trial and its fold user attributes into stable columns."""
    row: dict[str, Any] = {
        "trial": trial.number,
        "state": trial.state.name,
        "loss": trial.value if trial.state is TrialState.COMPLETE else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "datetime_start": _isoformat(trial.datetime_start),
        "datetime_complete": _isoformat(trial.datetime_complete),
        **{f"param_{name}": trial.params.get(name) for name in PARAMETER_NAMES},
        **{name: trial.user_attrs.get(name) for name in AGGREGATE_NAMES},
        "failure": trial.user_attrs.get("failure"),
    }
    folds = trial.user_attrs.get("folds", [])
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        folds = []
    for fold_index in range(fold_count):
        fold = folds[fold_index] if fold_index < len(folds) else {}
        if not isinstance(fold, Mapping):
            fold = {}
        for metric in FOLD_METRIC_NAMES:
            row[f"fold_{fold_index + 1}_{metric}"] = fold.get(metric)
    return row


def _pareto_rows(trial_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return nondominated completed rows that satisfy the fold activity requirement."""
    active: list[dict[str, Any]] = []
    for row in trial_rows:
        if (
            row["state"] != TrialState.COMPLETE.name
            or row["missing_round_trips"] != 0
            or not _is_finite(row["robust_annual_return_pct"])
            or not _is_finite(row["worst_drawdown_pct"])
        ):
            continue
        active.append(
            {
                **row,
                "robust_return": float(row["robust_annual_return_pct"]),
                "worst_drawdown": abs(float(row["worst_drawdown_pct"])),
            }
        )
    front = pareto_front(active)
    return [
        {key: value for key, value in row.items() if key not in {"robust_return", "worst_drawdown"}}
        for row in front
    ]


def _importance_rows(study: Study) -> tuple[list[dict[str, Any]], str | None]:
    """Calculate importance or retain an explanatory warning row on failure."""
    try:
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not complete:
            raise ValueError("no completed trials")
        importance = get_param_importances(study)
        if not importance:
            raise ValueError("no parameter importance values were produced")
        return (
            [
                {"parameter": parameter, "importance": value, "warning": None}
                for parameter, value in importance.items()
            ],
            None,
        )
    except Exception as exc:  # Optuna evaluators may reject sparse or degenerate studies.
        warning = f"Parameter importance unavailable: {exc}"
        return ([{"parameter": None, "importance": None, "warning": warning}], warning)


def _selected_trial(trials: Sequence[FrozenTrial]) -> FrozenTrial | None:
    """Return the lowest-loss completed trial with trial number as the tie-breaker."""
    complete = [
        trial for trial in trials if trial.state is TrialState.COMPLETE and _is_finite(trial.value)
    ]
    return min(complete, key=lambda trial: (float(trial.value), trial.number), default=None)


def _boundary_warnings(parameters: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    """Describe selected values that equal a configured search-space boundary."""
    bounds = metadata.get("bounds", {})
    if not isinstance(bounds, Mapping):
        return []
    warnings: list[str] = []
    for name in PARAMETER_NAMES:
        value = parameters.get(name)
        family = (
            "roc"
            if "roc_period" in name
            else "momentum" if "momentum_period" in name else "threshold"
        )
        for side in ("min", "max"):
            boundary = bounds.get(f"{family}_{side}")
            if _numbers_equal(value, boundary):
                warnings.append(f"{name} equals the {side} search boundary ({boundary}).")
    return warnings


def _summary(
    study: Study,
    metadata: Mapping[str, Any],
    comparison: Mapping[str, Any],
    terminal_trials: Sequence[FrozenTrial],
    selected: FrozenTrial | None,
    boundary_warnings: Sequence[str],
    importance_warning: str | None,
) -> dict[str, Any]:
    """Build complete machine-readable configuration, selection, and evidence."""
    counts: dict[str, int] = {}
    for trial in terminal_trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    selected_evaluation = None
    if selected is not None:
        selected_evaluation = {
            "loss": selected.value,
            **{name: selected.user_attrs.get(name) for name in AGGREGATE_NAMES},
            "folds": selected.user_attrs.get("folds", []),
        }
    return {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "metadata": dict(metadata),
        "loss_definitions": LOSS_DEFINITIONS,
        "loss_formula": LOSS_FORMULA,
        "trial_counts": counts,
        "selected_trial": selected.number if selected else None,
        "selected_parameters": dict(selected.params) if selected else None,
        "selected_evaluation": selected_evaluation,
        "boundary_warnings": list(boundary_warnings),
        "importance_warning": importance_warning,
        "comparison": dict(comparison),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write mappings with one stable header, including for empty results."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    study: Study,
    metadata: Mapping[str, Any],
    comparison: Mapping[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
    pareto_rows: Sequence[Mapping[str, Any]],
    importance_rows: Sequence[Mapping[str, Any]],
    importance_warning: str | None,
    selected: FrozenTrial | None,
    boundary_warnings: Sequence[str],
) -> str:
    """Render a self-contained HTML report with inline SVG and escaped evidence."""
    loss_config = metadata.get("loss_config", {})
    selected_parameters = dict(selected.params) if selected else {}
    top_rows = sorted(
        trial_rows,
        key=lambda row: (
            row["state"] != TrialState.COMPLETE.name,
            float(row["loss"]) if _is_finite(row["loss"]) else math.inf,
            int(row["trial"]),
        ),
    )[:25]
    folds = selected.user_attrs.get("folds", []) if selected else []

    metadata_table = _key_value_table(_flatten_mapping(metadata))
    parameter_table = _key_value_table(list(selected_parameters.items()))
    trial_table = _table(
        (
            "Trial",
            "State",
            "Loss",
            "Robust return",
            "Worst drawdown",
            "Missing round trips",
            "Failure",
        ),
        [
            (
                row["trial"],
                row["state"],
                row["loss"],
                row["robust_annual_return_pct"],
                row["worst_drawdown_pct"],
                row["missing_round_trips"],
                row["failure"],
            )
            for row in top_rows
        ],
    )
    pareto_table = _table(
        ("Trial", "Loss", "Robust return", "Worst drawdown"),
        [
            (
                row["trial"],
                row["loss"],
                row["robust_annual_return_pct"],
                row["worst_drawdown_pct"],
            )
            for row in pareto_rows
        ],
    )
    fold_table = _table(
        ("Fold", "Annual return", "Maximum drawdown", "Total trades"),
        [
            (
                index,
                fold.get("annual_return_pct"),
                fold.get("max_drawdown_pct"),
                fold.get("total_trades"),
            )
            for index, fold in enumerate(folds, start=1)
            if isinstance(fold, Mapping)
        ],
    )
    holdout_table, candidate_table, baseline_table, gate_table = _comparison_tables(comparison)
    warning_list = _warning_list(boundary_warnings)
    importance_message = (
        f'<p class="warning">{_escape(importance_warning)}</p>' if importance_warning else ""
    )
    resolved_loss = _resolved_loss(loss_config)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bayesian Momentum Optimization — {_escape(study.study_name)}</title>
<style>
body {{ background:#0b1020; color:#e6edf7; font-family:Arial,sans-serif; margin:0; }}
main {{ margin:auto; max-width:1180px; padding:32px 20px 64px; }}
h1,h2 {{ color:#f8fafc; }} h2 {{ border-bottom:1px solid #334155; padding-bottom:8px; }}
.card {{ background:#111a2e; border:1px solid #26334d; border-radius:10px; margin:18px 0;
padding:18px; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid #2b3956;
padding:8px 10px; text-align:left; }} th {{ color:#93c5fd; }}
code,pre {{ background:#09101d; border-radius:6px; color:#c4f1be; padding:10px; white-space:pre-wrap; }}
.warning {{ color:#fbbf24; }} .pass {{ color:#86efac; }} .fail {{ color:#fca5a5; }}
svg {{ background:#09101d; border-radius:8px; height:auto; max-width:100%; }}
</style>
</head>
<body><main>
<h1>Bayesian Momentum Optimization</h1>
<p>Study: {_escape(study.study_name)}</p>
<section class="card"><h2>Metadata and Configuration</h2>{metadata_table}</section>
<section class="card"><h2>Composite Loss</h2>
<pre>{_escape(LOSS_DEFINITIONS)}
{_escape(LOSS_FORMULA)}</pre><p>{_escape(resolved_loss)}</p></section>
<section class="card"><h2>Selected Parameters</h2>{parameter_table}</section>
<section class="card"><h2>Boundary Warnings</h2>{warning_list}</section>
<section class="card"><h2>Optimization History</h2>{_history_svg(trial_rows)}</section>
<section class="card"><h2>Parameter Importance</h2>{importance_message}
{_importance_svg(importance_rows, importance_warning)}</section>
<section class="card"><h2>Top 25 Trials</h2>{trial_table}</section>
<section class="card"><h2>Pareto Front</h2>
<p>Active nondominated trials maximize robust return and minimize absolute drawdown.</p>{pareto_table}</section>
<section class="card"><h2>Fold Metrics</h2>{fold_table}</section>
<section class="card"><h2>Holdout Comparison</h2>{holdout_table}
<h3>Candidate Parameters</h3>{candidate_table}<h3>Baseline Parameters</h3>{baseline_table}</section>
<section class="card"><h2>Adoption Gates</h2>{gate_table}
<p>Holdout results are adoption evidence only and never feed back into optimization. No runtime
profile is edited automatically.</p></section>
</main></body></html>
"""


def _comparison_tables(
    comparison: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Render holdout metrics, parameters, and named adoption gates."""
    candidate_metrics = comparison.get("candidate_metrics", {})
    baseline_metrics = comparison.get("baseline_metrics", {})
    if not isinstance(candidate_metrics, Mapping):
        candidate_metrics = {}
    if not isinstance(baseline_metrics, Mapping):
        baseline_metrics = {}
    metric_names = sorted(set(candidate_metrics) | set(baseline_metrics))
    holdout = _table(
        ("Metric", "Candidate", "Baseline"),
        [(name, candidate_metrics.get(name), baseline_metrics.get(name)) for name in metric_names],
    )
    candidate = comparison.get("candidate_parameters", {})
    baseline = comparison.get("baseline_parameters", {})
    candidate_table = _key_value_table(
        list(candidate.items()) if isinstance(candidate, Mapping) else []
    )
    baseline_table = _key_value_table(
        list(baseline.items()) if isinstance(baseline, Mapping) else []
    )
    checks = comparison.get("adoption_checks", {})
    if not isinstance(checks, Mapping):
        checks = {}
    gates = _table(
        ("Gate", "Result"),
        [(name, "PASS" if value else "FAIL") for name, value in checks.items()],
    )
    return holdout, candidate_table, baseline_table, gates


def _history_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render loss by completed-trial number as an inline SVG."""
    points = [
        (int(row["trial"]), float(row["loss"]))
        for row in rows
        if row["state"] == TrialState.COMPLETE.name and _is_finite(row["loss"])
    ]
    if not points:
        return '<svg viewBox="0 0 800 120" role="img"><text x="20" y="65" fill="#94a3b8">No completed trial history.</text></svg>'
    trial_numbers = [point[0] for point in points]
    losses = [point[1] for point in points]
    x_min, x_max = min(trial_numbers), max(trial_numbers)
    y_min, y_max = min(losses), max(losses)
    coordinates = [
        (
            _scale(trial_number, x_min, x_max, 30.0, 770.0),
            _scale(loss, y_min, y_max, 170.0, 20.0),
        )
        for trial_number, loss in points
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#60a5fa" />' for x, y in coordinates
    )
    return (
        '<svg viewBox="0 0 800 200" role="img" aria-label="Optimization loss history">'
        '<line x1="30" y1="180" x2="770" y2="180" stroke="#475569" />'
        f'<polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="2" />'
        f"{circles}</svg>"
    )


def _importance_svg(rows: Sequence[Mapping[str, Any]], warning: str | None) -> str:
    """Render parameter-importance bars or the escaped reason they are unavailable."""
    values = [row for row in rows if _is_finite(row.get("importance"))]
    if not values:
        label = warning or "No parameter importance values."
        return (
            '<svg viewBox="0 0 800 120" role="img">'
            f'<text x="20" y="65" fill="#fbbf24">{_escape(label)}</text></svg>'
        )
    height = max(120, 42 * len(values) + 20)
    maximum = max(float(row["importance"]) for row in values) or 1.0
    bars = []
    for index, row in enumerate(values):
        y = 18 + 42 * index
        width = 520 * float(row["importance"]) / maximum
        bars.append(
            f'<text x="10" y="{y + 15}" fill="#cbd5e1">{_escape(row["parameter"])}</text>'
            f'<rect x="220" y="{y}" width="{width:.2f}" height="20" fill="#34d399" />'
            f'<text x="750" y="{y + 15}" text-anchor="end" fill="#cbd5e1">'
            f'{_escape(row["importance"])}</text>'
        )
    return f'<svg viewBox="0 0 800 {height}" role="img">{"".join(bars)}</svg>'


def _resolved_loss(config: Any) -> str:
    """Describe the resolved loss weights without replacing the exact formula."""
    if not isinstance(config, Mapping):
        return "Resolved loss configuration is unavailable."
    return (
        f"Resolved values: drawdown_target={config.get('drawdown_target_pct')}, "
        f"drawdown_weight={config.get('drawdown_weight')}, "
        f"stability_weight={config.get('stability_weight')}, "
        f"minimum_round_trips={config.get('minimum_round_trips')}, "
        f"missing_round_trip_penalty={config.get('missing_round_trip_penalty')}."
    )


def _warning_list(warnings: Sequence[str]) -> str:
    """Render boundary messages without allowing markup from dynamic values."""
    if not warnings:
        return "<p>No selected value equals a search boundary.</p>"
    return (
        '<ul class="warning">'
        + "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
        + "</ul>"
    )


def _key_value_table(rows: Sequence[tuple[Any, Any]]) -> str:
    """Render a two-column table."""
    return _table(("Field", "Value"), rows)


def _table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """Render an escaped HTML table and retain a header for empty evidence."""
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    if not body:
        body = f'<tr><td colspan="{len(headers)}">No evidence available.</td></tr>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested configuration mappings into stable dotted labels."""
    rows: list[tuple[str, Any]] = []
    for key in sorted(value, key=str):
        label = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            rows.extend(_flatten_mapping(item, label))
        else:
            rows.append((label, item))
    return rows


def _escape(value: Any) -> str:
    """Convert any dynamic report value to escaped text."""
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        text = json.dumps(_json_compatible(value), sort_keys=True, allow_nan=False)
    else:
        text = str(value)
    return html.escape(text, quote=True)


def _json_compatible(value: Any) -> Any:
    """Return recursively JSON-compatible evidence with finite numeric values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    return str(value)


def _isoformat(value: datetime | None) -> str | None:
    """Serialize an optional Optuna timestamp."""
    return value.isoformat() if value else None


def _is_finite(value: Any) -> bool:
    """Return whether a value is a finite real number, excluding booleans."""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _numbers_equal(left: Any, right: Any) -> bool:
    """Compare numeric boundary values while rejecting missing and non-finite inputs."""
    return (
        _is_finite(left)
        and _is_finite(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _scale(
    value: float, source_min: float, source_max: float, target_min: float, target_max: float
) -> float:
    """Scale a chart coordinate, centering a degenerate one-point domain."""
    if source_min == source_max:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)
