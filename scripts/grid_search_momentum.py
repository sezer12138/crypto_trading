#!/usr/bin/env python3
"""Grid-search utilities for Momentum strategy hyperparameters."""

import argparse
import math
import logging
import sys
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import BacktestEngine
from strategies.momentum import MomentumStrategy
from strategies.momentum_profiles import get_momentum_profile
from strategies.constants import (
    DEFAULT_MOMENTUM_SELL_PERIOD,
    DEFAULT_MOMENTUM_SELL_ROC_PERIOD,
    DEFAULT_MOMENTUM_SELL_THRESHOLD,
)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
Number = TypeVar("Number", int, float)
METRIC_NAMES = (
    "total_return_pct",
    "annual_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "win_rate_pct",
    "total_trades",
)
DEFAULT_ROC_PERIODS = list(range(2, 31, 2))
DEFAULT_MOMENTUM_PERIODS = list(range(2, 41, 2))
DEFAULT_THRESHOLDS = [value / 1000 for value in range(5, 81, 5)]
SUB_HOURLY_ROC_HOURS = [2, 4, 8, 12, 16, 24, 36]
SUB_HOURLY_MOMENTUM_HOURS = [1, 2, 4, 6, 12, 24]
SUB_HOURLY_THRESHOLDS = [0.015, 0.025, 0.035, 0.045, 0.055]
TOP_BUY_CANDIDATES = 5
STABILITY_SLICE_COUNT = 3
STABILITY_SHORTLIST_SIZE = 20
PARAMETER_COLUMNS = (
    "buy_roc_period",
    "buy_momentum_period",
    "buy_threshold",
    "sell_roc_period",
    "sell_momentum_period",
    "sell_threshold",
)


@contextmanager
def quiet_backtest_logs():
    """Suppress expected per-combination risk warnings and restore logger state."""
    logger = logging.getLogger("backtest")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def _parse_positive_list(value: str, converter, label: str) -> List[Number]:
    """Parse a comma-separated list of unique positive numbers."""
    raw_items = value.split(",")
    if not value.strip() or any(not item.strip() for item in raw_items):
        raise ValueError(f"{label} must be a comma-separated list of positive values")

    parsed = []
    try:
        for item in raw_items:
            number = converter(item.strip())
            if number <= 0 or not math.isfinite(number):
                raise ValueError
            if number not in parsed:
                parsed.append(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only positive values") from exc
    return parsed


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated positive integer parameters."""
    return _parse_positive_list(value, int, "Integer parameters")


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated positive floating-point parameters."""
    return _parse_positive_list(value, float, "Floating-point parameters")


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load, validate, and chronologically sort an OHLCV CSV file."""
    data = pd.read_csv(path)
    if "timestamp" not in data.columns:
        raise ValueError("Input data is missing required column: timestamp")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")

    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Input data contains an invalid timestamp") from exc
    if data["timestamp"].duplicated().any():
        raise ValueError("Input data contains duplicate timestamps")

    try:
        for column in REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCV columns must contain numeric values") from exc
    if not np.isfinite(data.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV columns must contain finite numeric values")

    return data.set_index("timestamp").loc[:, list(REQUIRED_COLUMNS)].sort_index()


def chronological_split(
    df: pd.DataFrame, train_ratio: float, max_lookback: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered data into non-overlapping train and validation partitions."""
    if not 0 < train_ratio < 1:
        raise ValueError("The train ratio must be between 0 and 1")
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index].copy()
    validation = df.iloc[split_index:].copy()
    if len(train) <= max_lookback or len(validation) <= max_lookback:
        raise ValueError("Train and validation partitions must exceed the maximum lookback")
    return train, validation


def infer_interval_minutes(index: pd.DatetimeIndex) -> int:
    """Infer the dominant positive whole-minute candle cadence."""
    if len(index) < 3:
        raise ValueError("At least three timestamps are required to infer candle cadence")
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60
    valid = deltas[(deltas > 0) & (deltas % 1 == 0)].astype(int)
    if len(valid) != len(deltas):
        raise ValueError("Timestamps must have a positive whole-minute cadence")
    counts = valid.value_counts()
    if counts.empty or int(counts.iloc[0]) <= len(valid) / 2:
        raise ValueError("Timestamps do not have a dominant candle cadence")
    return int(counts.index[0])


def _hours_to_bars(hours: Sequence[int], interval_minutes: int) -> List[int]:
    """Convert duration candidates to unique candle counts."""
    return list(dict.fromkeys(max(1, round(hour * 60 / interval_minutes)) for hour in hours))


def resolve_search_ranges(
    args: argparse.Namespace, data: pd.DataFrame
) -> Tuple[List[int], List[int], List[float]]:
    """Resolve explicit raw-bar ranges or interval-aware defaults."""
    if (
        args.roc_periods is not None
        and args.momentum_periods is not None
        and args.thresholds is not None
    ):
        return args.roc_periods, args.momentum_periods, args.thresholds

    interval_minutes = infer_interval_minutes(data.index)
    if interval_minutes < 60:
        default_roc = _hours_to_bars(SUB_HOURLY_ROC_HOURS, interval_minutes)
        default_momentum = _hours_to_bars(SUB_HOURLY_MOMENTUM_HOURS, interval_minutes)
        default_thresholds = SUB_HOURLY_THRESHOLDS.copy()
    else:
        default_roc = DEFAULT_ROC_PERIODS.copy()
        default_momentum = DEFAULT_MOMENTUM_PERIODS.copy()
        default_thresholds = DEFAULT_THRESHOLDS.copy()

    return (
        args.roc_periods if args.roc_periods is not None else default_roc,
        args.momentum_periods if args.momentum_periods is not None else default_momentum,
        args.thresholds if args.thresholds is not None else default_thresholds,
    )


def _prefixed_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Select standard backtest metrics and prefix their names."""
    return {f"{prefix}_{name}": metrics.get(name, 0.0) for name in METRIC_NAMES}


def _run_combination(
    train_df: pd.DataFrame,
    parameters: Dict[str, float],
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    loss_cooldown_enabled: bool = True,
) -> Dict[str, float]:
    """Run one Momentum combination and return parameters plus training metrics."""
    strategy = MomentumStrategy(**parameters)
    engine = BacktestEngine(
        initial_capital=capital,
        drawdown_breaker_enabled=drawdown_breaker_enabled,
        loss_cooldown_enabled=loss_cooldown_enabled,
    )
    result = engine.run_backtest(train_df, strategy, coin=coin)
    row = parameters.copy()
    row.update(_prefixed_metrics(result.metrics, "train"))
    return row


def evaluate_buy_grid(
    train_df: pd.DataFrame,
    roc_periods: Sequence[int],
    momentum_periods: Sequence[int],
    thresholds: Sequence[float],
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    loss_cooldown_enabled: bool = True,
) -> pd.DataFrame:
    """Backtest every buy combination with fixed default sell parameters."""
    rows = []
    for roc_period, momentum_period, threshold in product(
        roc_periods, momentum_periods, thresholds
    ):
        parameters = {
            "buy_roc_period": roc_period,
            "buy_momentum_period": momentum_period,
            "buy_threshold": threshold,
            "sell_roc_period": DEFAULT_MOMENTUM_SELL_ROC_PERIOD,
            "sell_momentum_period": DEFAULT_MOMENTUM_SELL_PERIOD,
            "sell_threshold": DEFAULT_MOMENTUM_SELL_THRESHOLD,
        }
        rows.append(
            _run_combination(
                train_df,
                parameters,
                capital,
                drawdown_breaker_enabled,
                coin,
                loss_cooldown_enabled,
            )
        )
    return pd.DataFrame(rows)


def select_top_buy_candidates(
    results: pd.DataFrame, count: int = TOP_BUY_CANDIDATES
) -> pd.DataFrame:
    """Return the highest-ranked unique buy triples."""
    buy_columns = ["buy_roc_period", "buy_momentum_period", "buy_threshold"]
    ranked = results.sort_values(
        by=[
            "train_total_return_pct",
            "train_sharpe_ratio",
            "train_max_drawdown_pct",
            *buy_columns,
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    )
    return ranked.loc[:, buy_columns].drop_duplicates().head(count).reset_index(drop=True)


def evaluate_sell_grid(
    train_df: pd.DataFrame,
    buy_candidates: pd.DataFrame,
    roc_periods: Sequence[int],
    momentum_periods: Sequence[int],
    thresholds: Sequence[float],
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    loss_cooldown_enabled: bool = True,
) -> pd.DataFrame:
    """Backtest each retained buy triple against every sell combination."""
    rows = []
    for buy in buy_candidates.itertuples(index=False):
        for roc_period, momentum_period, threshold in product(
            roc_periods, momentum_periods, thresholds
        ):
            parameters = {
                "buy_roc_period": int(buy.buy_roc_period),
                "buy_momentum_period": int(buy.buy_momentum_period),
                "buy_threshold": float(buy.buy_threshold),
                "sell_roc_period": roc_period,
                "sell_momentum_period": momentum_period,
                "sell_threshold": threshold,
            }
            rows.append(
                _run_combination(
                    train_df,
                    parameters,
                    capital,
                    drawdown_breaker_enabled,
                    coin,
                    loss_cooldown_enabled,
                )
            )
    return pd.DataFrame(rows)


def split_stability_slices(
    train_df: pd.DataFrame, slice_count: int, max_lookback: int
) -> List[pd.DataFrame]:
    """Split training data into equal, ordered, non-overlapping stability slices."""
    if slice_count < 1:
        raise ValueError("The stability slice count must be positive")
    base_size, remainder = divmod(len(train_df), slice_count)
    sizes = [base_size + (1 if index < remainder else 0) for index in range(slice_count)]
    if min(sizes) <= max_lookback:
        raise ValueError("Each stability slice must exceed the maximum lookback")
    slices = []
    start = 0
    for size in sizes:
        slices.append(train_df.iloc[start : start + size].copy())
        start += size
    return slices


def evaluate_stability(
    candidates: pd.DataFrame,
    slices: Sequence[pd.DataFrame],
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    loss_cooldown_enabled: bool = True,
) -> pd.DataFrame:
    """Evaluate shortlisted candidates across chronological stability slices."""
    rows = []
    for _, candidate in candidates.iterrows():
        parameters = {
            name: (int(candidate[name]) if "period" in name else float(candidate[name]))
            for name in PARAMETER_COLUMNS
        }
        row = candidate.to_dict()
        returns = []
        sharpes = []
        drawdowns = []
        round_trips = []
        for number, data_slice in enumerate(slices, start=1):
            metrics = _run_combination(
                data_slice,
                parameters,
                capital,
                drawdown_breaker_enabled,
                coin,
                loss_cooldown_enabled,
            )
            total_return = float(metrics["train_total_return_pct"])
            trades = int(metrics["train_total_trades"])
            completed_round_trips = trades // 2
            returns.append(total_return)
            sharpes.append(float(metrics["train_sharpe_ratio"]))
            drawdowns.append(float(metrics["train_max_drawdown_pct"]))
            round_trips.append(completed_round_trips)
            row[f"stability_slice_{number}_return_pct"] = total_return
            row[f"stability_slice_{number}_round_trips"] = completed_round_trips

        compounded = (math.prod(1 + value / 100 for value in returns) - 1) * 100
        row["stability_total_return_pct"] = compounded
        row["stability_worst_return_pct"] = min(returns)
        row["stability_mean_sharpe_ratio"] = sum(sharpes) / len(sharpes)
        row["stability_max_drawdown_pct"] = min(drawdowns)
        row["stability_total_round_trips"] = sum(round_trips)
        row["stability_eligible"] = all(value >= 1 for value in round_trips) and sum(
            round_trips
        ) >= len(slices)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_stable_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Rank sufficiently active candidates by stability Total Return."""
    eligible = results.loc[results["stability_eligible"].astype(bool)].copy()
    if eligible.empty:
        eligible.insert(0, "stability_rank", pd.Series(dtype=int))
        return eligible
    ranked = eligible.sort_values(
        by=[
            "stability_total_return_pct",
            "stability_worst_return_pct",
            "stability_mean_sharpe_ratio",
            "stability_max_drawdown_pct",
            *PARAMETER_COLUMNS,
        ],
        ascending=[False, False, False, False, True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "stability_rank", range(1, len(ranked) + 1))
    return ranked


def profile_adoption_passes(
    candidate_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
) -> bool:
    """Return whether validation evidence is sufficient for runtime profile adoption."""
    return (
        candidate_metrics["validation_total_return_pct"]
        > baseline_metrics["validation_total_return_pct"]
        and int(candidate_metrics["validation_total_trades"]) // 2 >= 2
        and candidate_metrics["validation_max_drawdown_pct"] >= -30.0
    )


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank training results by return and deterministic tie breakers."""
    ranked = results.sort_values(
        by=[
            "train_total_return_pct",
            "train_sharpe_ratio",
            "train_max_drawdown_pct",
            *PARAMETER_COLUMNS,
        ],
        ascending=[False, False, False, True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def validate_winner(
    validation_df: pd.DataFrame,
    ranked_results: pd.DataFrame,
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    loss_cooldown_enabled: bool = True,
) -> Dict[str, float]:
    """Evaluate only the highest-ranked training setting on validation data."""
    winner = ranked_results.iloc[0]
    strategy = MomentumStrategy(
        buy_roc_period=int(winner["buy_roc_period"]),
        buy_momentum_period=int(winner["buy_momentum_period"]),
        buy_threshold=float(winner["buy_threshold"]),
        sell_roc_period=int(winner["sell_roc_period"]),
        sell_momentum_period=int(winner["sell_momentum_period"]),
        sell_threshold=float(winner["sell_threshold"]),
    )
    engine = BacktestEngine(
        initial_capital=capital,
        drawdown_breaker_enabled=drawdown_breaker_enabled,
        loss_cooldown_enabled=loss_cooldown_enabled,
    )
    result = engine.run_backtest(validation_df, strategy, coin=coin)
    return _prefixed_metrics(result.metrics, "validation")


def _interval_label(interval_minutes: int) -> str:
    """Return the canonical repository interval label for a candle cadence."""
    known = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d"}
    return known.get(interval_minutes, f"{interval_minutes}m")


def validate_current_profile(
    validation_df: pd.DataFrame,
    capital: float,
    drawdown_breaker_enabled: bool,
    coin: str,
    interval: str,
    loss_cooldown_enabled: bool = True,
) -> Dict[str, float]:
    """Evaluate the currently resolved runtime profile on validation data."""
    strategy = MomentumStrategy(**get_momentum_profile(coin, interval))
    engine = BacktestEngine(
        initial_capital=capital,
        drawdown_breaker_enabled=drawdown_breaker_enabled,
        loss_cooldown_enabled=loss_cooldown_enabled,
    )
    result = engine.run_backtest(validation_df, strategy, coin=coin)
    return _prefixed_metrics(result.metrics, "validation")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse Momentum grid-search command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find Momentum strategy parameters using chronological validation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/historical/btc_1h_730d.csv"),
        help="Timestamp-indexed OHLCV CSV file",
    )
    parser.add_argument(
        "--roc-periods",
        type=parse_int_list,
        default=None,
        help="Comma-separated ROC lookback periods in candles",
    )
    parser.add_argument(
        "--momentum-periods",
        type=parse_int_list,
        default=None,
        help="Comma-separated momentum lookback periods in candles",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=None,
        help="Comma-separated positive ROC thresholds",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/momentum_grid_search.csv"),
    )
    parser.add_argument("--coin", default="BTC")
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


def write_results(
    ranked_results: pd.DataFrame,
    validation_metrics: Dict[str, float],
    output: Path,
) -> pd.DataFrame:
    """Attach winner validation metrics and write ranked results to CSV."""
    written = ranked_results.copy()
    for name, value in validation_metrics.items():
        written[name] = float("nan")
        written.loc[written.index[0], name] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    written.to_csv(output, index=False)
    return written


def run_search(args: argparse.Namespace) -> pd.DataFrame:
    """Run interval-aware search, stability selection, and final validation."""
    if args.capital <= 0 or not math.isfinite(args.capital):
        raise ValueError("Initial capital must be positive")
    data = load_ohlcv(args.data)
    interval_minutes = infer_interval_minutes(data.index)
    roc_periods, momentum_periods, thresholds = resolve_search_ranges(args, data)
    max_lookback = max(max(roc_periods), max(momentum_periods))
    train, validation = chronological_split(data, args.train_ratio, max_lookback)
    breaker_enabled = not args.disable_drawdown_breaker
    loss_cooldown_enabled = not args.disable_loss_cooldown

    grid_size = len(roc_periods) * len(momentum_periods) * len(thresholds)
    retained_count = min(TOP_BUY_CANDIDATES, grid_size)
    evaluation_count = grid_size + retained_count * grid_size
    print(
        f"Searching {evaluation_count} parameter combinations "
        f"({grid_size} buy + {retained_count * grid_size} sell)"
    )
    print(
        f"Training: {len(train)} rows ({train.index[0]} to {train.index[-1]}); "
        f"validation: {len(validation)} rows ({validation.index[0]} to {validation.index[-1]})"
    )

    with quiet_backtest_logs():
        buy_results = evaluate_buy_grid(
            train,
            roc_periods,
            momentum_periods,
            thresholds,
            args.capital,
            breaker_enabled,
            args.coin.upper(),
            loss_cooldown_enabled,
        )
        buy_candidates = select_top_buy_candidates(buy_results)
        training_results = evaluate_sell_grid(
            train,
            buy_candidates,
            roc_periods,
            momentum_periods,
            thresholds,
            args.capital,
            breaker_enabled,
            args.coin.upper(),
            loss_cooldown_enabled,
        )
        ranked = rank_results(training_results)
        shortlist = ranked.head(STABILITY_SHORTLIST_SIZE).copy()
        slices = split_stability_slices(train, STABILITY_SLICE_COUNT, max_lookback)
        stability_results = evaluate_stability(
            shortlist,
            slices,
            args.capital,
            breaker_enabled,
            args.coin.upper(),
            loss_cooldown_enabled,
        )
        stable_ranked = rank_stable_candidates(stability_results)

        if stable_ranked.empty:
            diagnostics = stability_results.copy()
            diagnostics.insert(0, "stability_rank", float("nan"))
            written = write_results(diagnostics, {}, args.output)
            print(
                "No deployable winner: no shortlisted candidate completed at least "
                "one round trip in every stability slice"
            )
            print(f"Results written to: {args.output}")
            return written

        validation_metrics = validate_winner(
            validation,
            stable_ranked,
            args.capital,
            breaker_enabled,
            args.coin.upper(),
            loss_cooldown_enabled,
        )
        baseline_metrics = validate_current_profile(
            validation,
            args.capital,
            breaker_enabled,
            args.coin.upper(),
            _interval_label(interval_minutes),
            loss_cooldown_enabled,
        )

    inactive = stability_results.loc[~stability_results["stability_eligible"].astype(bool)].copy()
    inactive.insert(0, "stability_rank", float("nan"))
    ordered = pd.concat([stable_ranked, inactive], ignore_index=True, sort=False)
    adoption_passed = profile_adoption_passes(validation_metrics, baseline_metrics)
    output_metrics = validation_metrics.copy()
    output_metrics.update(
        {
            name.replace("validation_", "baseline_validation_"): value
            for name, value in baseline_metrics.items()
        }
    )
    output_metrics["profile_adoption_passed"] = float(adoption_passed)
    written = write_results(ordered, output_metrics, args.output)
    winner = written.iloc[0]
    print(
        "Winning parameters: "
        f"buy_roc_period={int(winner['buy_roc_period'])}, "
        f"buy_momentum_period={int(winner['buy_momentum_period'])}, "
        f"buy_threshold={winner['buy_threshold']:.6g}, "
        f"sell_roc_period={int(winner['sell_roc_period'])}, "
        f"sell_momentum_period={int(winner['sell_momentum_period'])}, "
        f"sell_threshold={winner['sell_threshold']:.6g}"
    )
    print(f"Stability total return: {winner['stability_total_return_pct']:.2f}%")
    print(f"Validation total return: {winner['validation_total_return_pct']:.2f}%")
    print(
        f"Current-profile validation return: "
        f"{winner['baseline_validation_total_return_pct']:.2f}%"
    )
    print(f"Profile adoption guard: {'PASS' if adoption_passed else 'FAIL'}")
    print(f"Results written to: {args.output}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Momentum grid-search CLI."""
    try:
        args = parse_arguments(argv)
        run_search(args)
        return 0
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
