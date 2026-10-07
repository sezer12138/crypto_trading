"""Optimize all non-Momentum strategies with development-only selection."""

import argparse
import logging
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna
import pandas as pd

from optimization.data import (
    fingerprint_file,
    infer_interval_minutes,
    load_ohlcv,
    split_holdout_and_folds,
)
from optimization.strategy_report import write_results
from optimization.strategy_search import (
    STRATEGIES,
    SearchConfig,
    evaluate_partition,
    optimize_strategy,
)


def main() -> None:
    """Run reproducible searches and checkpoint evidence after every strategy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, help="OHLCV CSV; defaults to the coin/interval 730-day cache"
    )
    parser.add_argument("--coin", default="btc")
    parser.add_argument("--interval", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enable-drawdown-breaker", action="store_true")
    parser.add_argument("--enable-loss-cooldown", action="store_true")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("At least two trials are required (default baseline plus search)")
    if args.data is None:
        args.data = Path(f"data/historical/{args.coin.lower()}_{args.interval}_730d.csv")
    cached = re.match(r"^(btc|eth|sol)_(1m|5m|15m|1h|4h|1d)_\d+d\.csv$", args.data.name.lower())
    if cached and (cached.group(1) != args.coin.lower() or cached.group(2) != args.interval):
        parser.error("Cached data filename does not match the requested coin and interval")
    config = SearchConfig(
        initial_capital=args.capital,
        coin=args.coin.lower(),
        interval=args.interval,
        drawdown_breaker_enabled=args.enable_drawdown_breaker,
        loss_cooldown_enabled=args.enable_loss_cooldown,
    )
    data = load_ohlcv(args.data)
    interval = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[args.interval]
    if infer_interval_minutes(data.index) != interval:
        parser.error("Requested interval does not match the input data cadence")
    folds, holdout = split_holdout_and_folds(data, 0.2, 3, max_lookback=128)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("backtest").setLevel(logging.ERROR)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(
        f"results/strategy_optimization_{args.coin}_{args.interval}_{stamp}"
    )
    evidence = {
        "schema": 1,
        "identity": config.identity(),
        "data_path": str(args.data.resolve()),
        "data_sha256": fingerprint_file(args.data),
        "seed": args.seed,
        "trials_per_strategy": args.trials,
        "holdout": {
            "start": str(holdout.index[0]),
            "end": str(holdout.index[-1]),
            "rows": len(holdout),
        },
        "development_partitions": [
            {"start": str(f.index[0]), "end": str(f.index[-1]), "rows": len(f)} for f in folds
        ],
        "strategies": {},
    }
    write_results(output, evidence)
    for index, name in enumerate(args.strategies):
        print(
            f"[{index+1}/{len(args.strategies)}] Searching {name}: {args.trials} trials", flush=True
        )
        result = optimize_strategy(
            name, folds, holdout, config, args.trials, args.seed + STRATEGIES.index(name)
        )
        # Full-sample results are descriptive only and cannot affect adoption.
        result["baseline_full_sample"] = evaluate_partition(data, name, {}, config)
        result["candidate_full_sample"] = evaluate_partition(
            data, name, result["parameters"], config
        )
        evidence["strategies"][name] = result
        write_results(output, evidence)
        print(
            f"  Holdout default={result['baseline_holdout']['total_return_pct']:+.2f}% candidate={result['candidate_holdout']['total_return_pct']:+.2f}% adopted={result['adopted']}",
            flush=True,
        )
    print(f"Report: {(output / 'report.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
