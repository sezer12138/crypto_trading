"""Data loading, validation, partitioning, and fingerprinting utilities."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


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


def split_holdout_and_folds(
    df: pd.DataFrame, holdout_ratio: float, fold_count: int, max_lookback: int
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Split chronological data into optimization folds and a final holdout."""
    if not 0 < holdout_ratio < 1:
        raise ValueError("Holdout ratio must be between zero and one")
    if fold_count < 2:
        raise ValueError("Fold count must be at least two")
    split = int(len(df) * (1 - holdout_ratio))
    pool, holdout = df.iloc[:split], df.iloc[split:]
    base, remainder = divmod(len(pool), fold_count)
    sizes = [base + (index < remainder) for index in range(fold_count)]
    if min(sizes) <= max_lookback or len(holdout) <= max_lookback:
        raise ValueError("Every optimization partition must exceed the maximum lookback")
    folds, start = [], 0
    for size in sizes:
        folds.append(pool.iloc[start : start + size].copy())
        start += size
    return folds, holdout.copy()


def fingerprint_file(path: Path) -> str:
    """Return the streaming SHA-256 fingerprint for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
