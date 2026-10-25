"""Predeclared capacity/history studies with absolute dates and a frozen final tail."""

from dataclasses import asdict, dataclass, replace
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from learning.config import LearningConfig
from learning.data import prepare_data
from learning.execution import make_engine
from learning.report import export_trajectory, research_metrics
from learning.strategy import load_checkpoint
from learning.train import BenchmarkStrategy, run_experiment
from optimization.data import fingerprint_file, load_ohlcv
from strategies import get_strategy


@dataclass(frozen=True)
class StudyCase:
    """An immutable model recipe and training-history start."""

    name: str
    training_start: str
    config: LearningConfig


def default_cases(pretrain_device: str = "cpu") -> list[StudyCase]:
    """Freeze two capacities, two history lengths and short-history budget controls."""
    base = LearningConfig(batch_size=32, cpu_threads=2, pretrain_device=pretrain_device)
    extended = replace(
        base,
        pretrain_epochs=50,
        policy_epochs=30,
        early_stopping_patience=8,
        min_pretrain_epochs=10,
        min_policy_epochs=10,
        lr_patience=3,
        restore_best_pretrain=True,
    )
    return [
        StudyCase(f"short_h{hidden}_baseline", "2023-01-01", replace(base, hidden=hidden))
        for hidden in (32, 128)
    ] + [
        StudyCase(f"{history}_h{hidden}_extended", start, replace(extended, hidden=hidden))
        for history, start in (("short", "2023-01-01"), ("long", "2018-01-01"))
        for hidden in (32, 128)
    ]


def choose_case(cases: dict[str, Any], limit: float) -> str:
    """Rank recipes by validation-only feasibility and median return/cost over all seeds."""

    def score(name: str) -> tuple[bool, float, float]:
        metrics = [run["validation"] for run in cases[name]["seeds"]]
        return (
            all(m["max_drawdown_pct"] <= limit * 100 for m in metrics),
            float(np.median([m["net_return_pct"] for m in metrics])),
            -float(np.median([m["total_cost"] for m in metrics])),
        )

    return max(cases, key=score)


def _timestamp(value: str) -> pd.Timestamp:
    """Normalize a protocol boundary to naive UTC, matching the stored CSV interface."""
    result = pd.Timestamp(value)
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def _boundary(frame: pd.DataFrame, value: str) -> int:
    stamp = _timestamp(value)
    if stamp not in frame.index:
        raise ValueError(f"Requested boundary is not an observed hourly candle: {stamp}")
    return int(frame.index.get_loc(stamp))


def _write_json(path: Path, value: Any) -> None:
    """Replace a completed JSON artifact atomically so interrupted runs can resume."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _source_hashes(repository: Path | None = None) -> dict[str, str]:
    """Prevent resuming completed cells against different modeling code."""
    root = repository if repository is not None else Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "learning").glob("*.py"))
    paths += list((root / "src" / "strategies").rglob("*.py"))
    paths += [root / "src" / "backtest.py", root / "src" / "optimization" / "data.py"]
    return {
        str(path.relative_to(root)): fingerprint_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def verify_completion(directory: Path) -> None:
    """Reject modified completed-cell evidence before a study resumes."""
    marker = directory / "completion.json"
    if not marker.exists():
        raise ValueError("Completed cell has no fingerprint manifest")
    for name, expected in json.loads(marker.read_text()).items():
        path = directory / name
        if not path.exists() or fingerprint_file(path) != expected:
            raise ValueError(f"Completed cell fingerprint changed: {name}")


def _tail_evaluation(
    frame: pd.DataFrame, start: int, output: Path, frozen: dict[str, Any], cases: list[StudyCase]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Open recent performance only after all expected checkpoint fingerprints are frozen."""
    for entry in frozen["checkpoints"]:
        if fingerprint_file(output / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen checkpoint changed before final-tail evaluation")
    result: dict[str, list[dict[str, Any]]] = {case.name: [] for case in cases}
    for entry in frozen["checkpoints"]:
        case = next(case for case in cases if case.name == entry["case"])
        # Only the causal warmup context is needed; it also avoids old training gaps.
        context = frame.iloc[max(0, start - 168 - case.config.lookback) : start]
        strategy = load_checkpoint(output / entry["path"], context=context)
        recent = make_engine(case.config).run_backtest(frame.iloc[start:], strategy)
        stress = make_engine(case.config, cost_multiplier=2).run_backtest(
            frame.iloc[start:], load_checkpoint(output / entry["path"], context=context)
        )
        name = f"{entry['case']}_seed_{entry['seed']}_recent"
        export_trajectory(recent, output, name)
        export_trajectory(stress, output, name + "_costs_doubled")
        result[entry["case"]].append(
            {
                "seed": entry["seed"],
                "metrics": research_metrics(recent),
                "costs_doubled": research_metrics(stress),
                "range": [str(frame.index[start]), str(frame.index[-1])],
            }
        )
    config = cases[0].config
    baselines = {}
    for name, strategy in {
        "cash": BenchmarkStrategy(False),
        "buy_hold_capped": BenchmarkStrategy(True),
        "ma_cross": get_strategy("ma_cross"),
        "momentum_generic": get_strategy("momentum"),
    }.items():
        account = make_engine(config).run_backtest(frame.iloc[start:], strategy)
        baselines[name] = research_metrics(account)
        export_trajectory(account, output, f"baseline_{name}_recent")
    return result, baselines


def _write_summary(report: dict[str, Any], output: Path) -> None:
    """Export all outcomes, including cash/rejected models, with no tail-based ranking."""
    rows = []
    for name, case in report["cases"].items():
        for seed in case["seeds"]:
            tail = next(t for t in case["recent_tail"] if t["seed"] == seed["seed"])
            summary = seed.get("training_summary", {})
            rows.append(
                [
                    name,
                    str(seed["seed"]),
                    str(seed["selected_epoch"]),
                    f"{seed['validation']['net_return_pct']:.2f}",
                    f"{seed['holdout']['net_return_pct']:.2f}",
                    f"{seed['holdout']['max_drawdown_pct']:.2f}",
                    f"{tail['metrics']['net_return_pct']:.2f}",
                    f"{tail['metrics']['max_drawdown_pct']:.2f}",
                    str(tail["metrics"]["total_trades"]),
                    f"{tail['costs_doubled']['net_return_pct']:.2f}",
                    str(summary.get("pretrain_updates", "See audit")),
                    str(summary.get("policy_updates", "See audit")),
                ]
            )
    columns = [
        "Case",
        "Seed",
        "Selected policy epoch",
        "Validation return %",
        "Historical return %",
        "Historical MDD %",
        "Recent return %",
        "Recent MDD %",
        "Recent fills",
        "Recent doubled-cost return %",
        "Pretrain updates",
        "Policy updates",
    ]
    introduction = (
        f"Selected recipe: **{report['selected_case']}**, ranked on validation only; "
        f"predeclared primary seed: **{report['primary_seed']}**.\n\n"
        "Historical evaluation reuses previously researched dates and is not blind. The recent tail "
        "was opened after all checkpoints were frozen; it is short retrospective evidence, not a "
        "prospective test. Models can remain in cash or hit permanent drawdown breakers. "
        "A zero loss/return or low drawdown does not establish a useful trading policy.\n\n"
        "Expanded-history runs receive more samples/updates per epoch; this measures the practical "
        "effect of adding history, not a compute-matched data-scaling law. Extended-budget runs also "
        "add LR decay, early stopping and best-pretraining restoration, so that comparison tests the "
        "complete optimization recipe rather than epoch count alone.\n\n"
        "All return and drawdown values below are percentages. Baselines use the same protective "
        "execution/risk rules; indicators warm up inside each supplied evaluation frame.\n\n"
    )
    markdown = "# BTC TCN Capacity and History Study\n\n" + introduction
    markdown += "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
    markdown += "\n## Recent-tail baselines\n\n| Baseline | Return % | MDD % | Fills |\n|---|---:|---:|---:|\n"
    for name, metrics in report["recent_baselines"].items():
        markdown += f"| {name} | {metrics['net_return_pct']:.2f} | {metrics['max_drawdown_pct']:.2f} | {metrics['total_trades']} |\n"
    (output / "report.md").write_text(markdown)
    table = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    table += "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BTC TCN Study</title>'
        "<style>body{font:15px system-ui;margin:32px;background:#101827;color:#e5e7eb}"
        "table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #374151;text-align:right}"
        "td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap}</style>"
        "<h1>BTC TCN capacity and history study</h1><pre>"
        + html.escape(introduction)
        + "</pre><table>"
        + table
        + "</table><h2>Protocol and evidence</h2>"
        "<p>See protocol.json, frozen.json, report.json and per-case audit files. "
        "TensorBoard: tensorboard --logdir this-study-directory</p></html>"
    )
    (output / "report.html").write_text(page)


def run_study(
    data_path: Path,
    output: Path,
    cases: list[StudyCase],
    seeds: list[int],
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    historical_end: str = "2026-10-07 04:00:00",
    tensorboard: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Freeze recipes/dates, train all cases, then evaluate an untouched recent tail.

    Resume reuses only complete, fingerprinted cells. A partially written cell must
    be preserved under another name before resuming; it is never overwritten.
    """
    if (
        not cases
        or len({c.name for c in cases}) != len(cases)
        or not seeds
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Unique nonempty study cases and seeds are required")
    if any(not c.name.replace("_", "").isalnum() for c in cases):
        raise ValueError("Case names must contain only letters, numbers and underscores")
    if (
        len(
            {
                (
                    c.config.commission,
                    c.config.slippage,
                    c.config.drawdown_limit,
                    c.config.max_weight,
                )
                for c in cases
            }
        )
        != 1
    ):
        raise ValueError("Study cases must share execution costs and risk limits")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a new empty study directory or explicit resume")
    frame = load_ohlcv(data_path)
    train_index = _boundary(frame, train_end)
    val_index = _boundary(frame, validation_end)
    history_index = _boundary(frame, historical_end)
    if not train_index < val_index < history_index < len(frame):
        raise ValueError(
            "Training, validation, historical and recent-tail boundaries must increase"
        )
    protocol = {
        "version": 1,
        "source": str(data_path.resolve()),
        "sha256": fingerprint_file(data_path),
        "source_code_sha256": _source_hashes(),
        "runtime_versions": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seeds": seeds,
        "primary_seed": seeds[0],
        "train_end": train_end,
        "validation_end": validation_end,
        "historical_end": historical_end,
        "recent_end": str(frame.index[-1]),
        "tensorboard": tensorboard,
        "cases": [
            {"name": c.name, "training_start": c.training_start, "config": asdict(c.config)}
            for c in cases
        ],
        "selection_rule": "all-seed feasible validation MDD; median validation return; lower median validation cost",
        "historical_status": "previously_researched_retrospective",
        "recent_status": "opened_only_after_all_recipes_and_checkpoints_frozen; short_retrospective_tail",
    }
    # JSON normalizes configuration tuples for exact comparisons on resume.
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume protocol differs in data, code, configuration or dates")
    else:
        if resume:
            raise ValueError("Cannot resume without an existing protocol")
        _write_json(protocol_path, protocol)
    reports = {}
    for case in cases:
        start = _boundary(frame, case.training_start)
        if start >= train_index:
            raise ValueError("Training history must start before the shared validation boundary")
        historical = frame.iloc[start:history_index]
        dataset_path = output / f"{case.name}_development.csv"
        cell_path = output / case.name
        if (cell_path / "report.json").exists():
            verify_completion(cell_path)
            cell = json.loads((cell_path / "report.json").read_text())
            if cell["data"]["sha256"] != fingerprint_file(dataset_path):
                raise ValueError("Completed cell dataset fingerprint changed")
        else:
            historical.to_csv(dataset_path, index_label="timestamp")
            prepared = prepare_data(
                historical,
                case.config,
                split_indices=(train_index - start, val_index - start),
                allow_training_gaps=True,
            )
            print(
                f"Study {case.name}: {len(prepared.train_indices)} safe training closes; "
                f"{case.config.hidden} hidden units",
                flush=True,
            )
            cell = run_experiment(
                dataset_path,
                cell_path,
                case.config,
                seeds,
                tensorboard=tensorboard,
                split_indices=(train_index - start, val_index - start),
                allow_training_gaps=True,
            )
            _write_json(
                cell_path / "completion.json",
                {
                    name: fingerprint_file(cell_path / name)
                    for name in ["report.json", *[f"seed_{seed}.pt" for seed in seeds]]
                },
            )
        reports[case.name] = cell
        _write_json(
            output / "progress.json", {"completed_cases": list(reports), "total_cases": len(cases)}
        )
    frozen = {
        "selection_partition": "validation",
        "selected_case": choose_case(reports, cases[0].config.drawdown_limit),
        "primary_seed": seeds[0],
        "checkpoints": [
            {
                "case": case.name,
                "seed": seed,
                "path": f"{case.name}/seed_{seed}.pt",
                "sha256": fingerprint_file(output / case.name / f"seed_{seed}.pt"),
            }
            for case in cases
            for seed in seeds
        ],
    }
    frozen_path = output / "frozen.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Frozen selection/checkpoints changed")
    _write_json(frozen_path, frozen)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    tails, baselines = _tail_evaluation(frame, history_index, output, frozen, cases)
    for name, values in tails.items():
        reports[name]["recent_tail"] = values
    report = {
        "protocol": protocol,
        "primary_seed": seeds[0],
        "selected_case": frozen["selected_case"],
        "cases": reports,
        "recent_baselines": baselines,
    }
    _write_json(output / "report.json", report)
    _write_summary(report, output)
    return report
