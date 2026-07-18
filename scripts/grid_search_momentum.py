#!/usr/bin/env python3
"""Grid-search utilities for Momentum strategy hyperparameters."""

from pathlib import Path
from typing import List, Tuple, TypeVar

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
Number = TypeVar("Number", int, float)


def _parse_positive_list(value: str, converter, label: str) -> List[Number]:
    """Parse a comma-separated list of unique positive numbers."""
    raw_items = value.split(",")
    if not value.strip() or any(not item.strip() for item in raw_items):
        raise ValueError(f"{label} must be a comma-separated list of positive values")

    parsed = []
    try:
        for item in raw_items:
            number = converter(item.strip())
            if number <= 0:
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
