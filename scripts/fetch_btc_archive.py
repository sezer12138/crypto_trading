#!/usr/bin/env python3
"""Acquire verified long-history Binance BTCUSDT spot hourly candles."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning.archive import acquire_btc_history


def main() -> None:
    """Parse explicit UTC bounds and preserve raw archives and dataset provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", required=True, help="Exclusive closed UTC hour, e.g. 2026-10-24")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/.binance_archive"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = acquire_btc_history(args.start, args.end, args.output, args.cache, args.workers)
    print(
        json.dumps(
            {key: result[key] for key in ("rows", "first", "last", "gaps", "outputs")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
