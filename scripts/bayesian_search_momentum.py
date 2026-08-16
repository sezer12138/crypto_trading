#!/usr/bin/env python3
"""Run resumable Bayesian optimization for the asymmetric Momentum strategy."""

import argparse
import functools
import math
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optuna.trial import TrialState

from optimization.data import infer_interval_minutes, load_ohlcv, split_holdout_and_folds
from optimization.momentum_evaluator import (
    BacktestRunConfig,
    evaluate_parameters,
    evaluate_worker,
    initialize_worker,
)
from optimization.momentum_objective import LossConfig, resolve_search_bounds
from optimization.momentum_report import compare_holdout, write_optimization_outputs
from optimization.optuna_study import (
    build_metadata,
    create_or_load_study,
    recover_stale_trials,
    run_process_trials,
    run_sequential_trials,
)
from strategies.momentum import MomentumStrategy
from strategies.momentum_profiles import get_momentum_profile

DEFAULT_DATA = Path("data/historical/btc_5m_1800d.csv")
DEFAULT_STORAGE = Path("results/momentum_bayesian.db")
DEFAULT_OUTPUT_PREFIX = Path("results/momentum_bayesian")
PARAMETER_NAMES = (
    "buy_roc_period",
    "buy_momentum_period",
    "buy_threshold",
    "sell_roc_period",
    "sell_momentum_period",
    "sell_threshold",
)


class _ArgumentParser(argparse.ArgumentParser):
    """Turn parse failures into the same one-line configuration error contract."""

    def error(self, message: str) -> None:
        raise ValueError(message)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Bayesian Momentum optimizer command-line arguments."""
    parser = _ArgumentParser(
        description="Optimize asymmetric Momentum parameters with resumable Optuna TPE"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--startup-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-ratio", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--drawdown-target", type=float, default=25.0)
    parser.add_argument("--drawdown-weight", type=float, default=2.0)
    parser.add_argument("--stability-weight", type=float, default=0.5)
    parser.add_argument("--minimum-round-trips", type=int, default=2)
    parser.add_argument("--missing-round-trip-penalty", type=float, default=25.0)
    parser.add_argument("--study-name", default="momentum_bayesian")
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--disable-drawdown-breaker",
        action="store_true",
        help="Disable forced liquidation and halt at the maximum drawdown threshold",
    )
    parser.add_argument(
        "--disable-loss-cooldown",
        action="store_true",
        help="Disable pausing new entries after consecutive losing trades",
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid configuration before loading data or creating a study."""
    if args.trials <= 0:
        raise ValueError("Trials must be positive")
    if args.workers <= 0:
        raise ValueError("Workers must be positive")
    if args.startup_trials <= 0:
        raise ValueError("Startup trials must be positive")
    if args.folds < 2:
        raise ValueError("Folds must be at least two")
    if args.minimum_round_trips <= 0:
        raise ValueError("Minimum round trips must be positive")
    if not math.isfinite(args.holdout_ratio) or not 0 < args.holdout_ratio < 1:
        raise ValueError("Holdout ratio must be between zero and one")
    _validate_positive_finite(args.capital, "Capital")
    _validate_positive_finite(args.drawdown_target, "Drawdown target")
    _validate_positive_finite(args.drawdown_weight, "Drawdown weight")
    _validate_positive_finite(args.stability_weight, "Stability weight")
    _validate_positive_finite(
        args.missing_round_trip_penalty,
        "Missing round trip penalty",
    )
    if not args.coin.strip():
        raise ValueError("Coin must not be empty")
    if not args.study_name.strip():
        raise ValueError("Study name must not be empty")
    if not args.data.exists() or not args.data.is_file():
        raise ValueError("Data path does not exist or is not a file")


def _validate_positive_finite(value: float, label: str) -> None:
    """Require one finite number strictly greater than zero."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def _interval_label(interval_minutes: int) -> str:
    """Format an inferred whole-minute cadence for runtime profile lookup."""
    if interval_minutes % (24 * 60) == 0:
        return f"{interval_minutes // (24 * 60)}d"
    if interval_minutes % 60 == 0:
        return f"{interval_minutes // 60}h"
    return f"{interval_minutes}m"


def _general_momentum_parameters() -> dict[str, int | float]:
    """Serialize the complete general Momentum constructor defaults."""
    strategy = MomentumStrategy()
    return {name: getattr(strategy, name) for name in PARAMETER_NAMES}


def _resolve_baseline(coin: str, interval_minutes: int) -> dict[str, int | float]:
    """Resolve the matching runtime profile or all general strategy defaults."""
    profile = get_momentum_profile(coin, _interval_label(interval_minutes))
    return dict(profile) if profile else _general_momentum_parameters()


def _execute_search(args: argparse.Namespace) -> tuple[dict[str, Path], bool]:
    """Compose one optimization run and report whether it was interrupted."""
    _validate_arguments(args)
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = load_ohlcv(args.data)
    interval_minutes = infer_interval_minutes(data.index)
    bounds = resolve_search_bounds(interval_minutes)
    max_lookback = max(bounds.roc_max, bounds.momentum_max)
    folds, holdout = split_holdout_and_folds(
        data,
        args.holdout_ratio,
        args.folds,
        max_lookback,
    )
    coin = args.coin.strip().upper()
    run_config = BacktestRunConfig(
        args.capital,
        coin,
        not args.disable_drawdown_breaker,
        not args.disable_loss_cooldown,
    )
    loss_config = LossConfig(
        args.drawdown_target,
        args.drawdown_weight,
        args.stability_weight,
        args.minimum_round_trips,
        args.missing_round_trip_penalty,
    )
    metadata = build_metadata(
        args.data,
        data,
        interval_minutes,
        coin,
        args.capital,
        bounds,
        loss_config,
        args.holdout_ratio,
        args.folds,
        run_config,
    )
    study = create_or_load_study(
        args.storage,
        args.study_name,
        metadata,
        args.seed,
        args.startup_trials,
        args.workers > 1,
    )
    recover_stale_trials(study)

    interrupted = False
    try:
        if args.workers == 1:
            evaluator = functools.partial(
                evaluate_parameters,
                folds,
                loss_config=loss_config,
                run_config=run_config,
            )
            run_summary = run_sequential_trials(study, args.trials, bounds, evaluator)
        else:
            initargs = (
                str(args.data),
                args.holdout_ratio,
                args.folds,
                max_lookback,
                loss_config,
                run_config,
            )
            run_summary = run_process_trials(
                study,
                args.trials,
                args.workers,
                bounds,
                initialize_worker,
                initargs,
                evaluate_worker,
            )
        interrupted = run_summary.interrupted
    except KeyboardInterrupt:
        interrupted = True

    comparison = None
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not interrupted and complete:
        candidate = dict(study.best_trial.params)
        baseline = _resolve_baseline(coin, interval_minutes)
        comparison = compare_holdout(holdout, candidate, baseline, run_config)
    paths = write_optimization_outputs(study, args.output_prefix, metadata, comparison)
    return paths, interrupted


def run_search(args: argparse.Namespace) -> dict[str, Path]:
    """Run optimization and return all generated artifact paths."""
    paths, _ = _execute_search(args)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a conventional process exit status."""
    try:
        args = parse_arguments(argv)
        paths, interrupted = _execute_search(args)
    except ValueError as exc:
        message = " ".join(str(exc).splitlines())
        print(f"Configuration error: {message}", file=sys.stderr)
        return 2
    except OSError as exc:
        message = " ".join(str(exc).splitlines())
        print(f"Configuration error: {message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    for name, path in paths.items():
        print(f"{name}: {path}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
