#!/usr/bin/env python3
"""Train and validate the optional BTC/cash TCN on an hourly historical CSV."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Parse explicit experiment parameters and preserve immutable output folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="Disable TensorBoard event logging"
    )
    args = parser.parse_args()
    try:
        from learning.config import LearningConfig
        from learning.train import run_experiment
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            parser.error("Install optional dependencies: pip install -r requirements-learning.txt")
        raise
    config = replace(
        LearningConfig(),
        pretrain_epochs=args.pretrain_epochs,
        policy_epochs=args.policy_epochs,
        hidden=args.hidden,
    )
    report = run_experiment(
        args.data, args.output, config, args.seeds, tensorboard=not args.no_tensorboard
    )
    print(report["interpretation"])
    print(f'Report: {(args.output/"report.html").resolve()}')
    if not args.no_tensorboard:
        print(f'TensorBoard logs: {(args.output/"tensorboard").resolve()}')


if __name__ == "__main__":
    main()
