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
    parser.add_argument("--pretrain-stride", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--restore-best-pretrain", action="store_true")
    parser.add_argument("--min-pretrain-epochs", type=int, default=0)
    parser.add_argument("--min-policy-epochs", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--pretrain-device", choices=["cpu", "mps"], default="cpu")
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
        pretrain_stride=args.pretrain_stride,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        restore_best_pretrain=args.restore_best_pretrain,
        min_pretrain_epochs=args.min_pretrain_epochs,
        min_policy_epochs=args.min_policy_epochs,
        cpu_threads=args.cpu_threads,
        pretrain_device=args.pretrain_device,
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
