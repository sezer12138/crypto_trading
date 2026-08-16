"""Tests for shared optimization data utilities."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.data import fingerprint_file, split_holdout_and_folds


def _ohlcv_frame(periods: int) -> pd.DataFrame:
    """Return deterministic hourly OHLCV data for partitioning tests."""
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    prices = [100.0 + value for value in range(periods)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [1000.0] * periods,
        },
        index=index,
    )


def test_split_holdout_and_folds_is_chronological_and_non_overlapping() -> None:
    """Partitions input into ordered equal folds followed by a holdout."""
    data = _ohlcv_frame(100)

    folds, holdout = split_holdout_and_folds(data, 0.20, 4, max_lookback=5)

    assert [len(fold) for fold in folds] == [20, 20, 20, 20]
    assert len(holdout) == 20
    assert all(folds[i].index.max() < folds[i + 1].index.min() for i in range(3))
    assert folds[-1].index.max() < holdout.index.min()


def test_fingerprint_file_changes_when_content_changes(tmp_path) -> None:
    """Returns a different fingerprint after a file's contents are replaced."""
    path = tmp_path / "data.csv"
    path.write_text("first", encoding="utf-8")
    first = fingerprint_file(path)

    path.write_text("second", encoding="utf-8")

    assert fingerprint_file(path) != first
