#!/usr/bin/env python3
"""Run the predeclared BTC hourly capacity/training/history study."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Freeze all study cells before any model sees the recent evaluation tail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    parser.add_argument("--pretrain-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from learning.study import default_cases, run_study

    report = run_study(
        args.data,
        args.output,
        default_cases(args.pretrain_device),
        args.seeds,
        tensorboard=not args.no_tensorboard,
        resume=args.resume,
    )
    print(f"Validation-selected recipe: {report['selected_case']}")
    print(f"Report: {(args.output / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
