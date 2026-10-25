"""Tests for fixed-date comparisons and genuine training-history gaps."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning.config import LearningConfig
from learning.data import prepare_data


def candles(n: int = 2400) -> pd.DataFrame:
    prices = 100 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.ones(n) * 100,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


def test_explicit_partitions_do_not_move_with_added_training_history():
    """Catch ratio-based date drift when comparing training-history lengths."""
    frame = candles()
    config = LearningConfig()
    long = prepare_data(frame, config, split_indices=(1600, 2000))
    short = prepare_data(frame.iloc[400:], config, split_indices=(1200, 1600))
    assert long.frame.index[long.train_end] == short.frame.index[short.train_end]
    assert long.frame.index[long.test_start] == short.frame.index[short.test_start]
    assert long.train_indices[-1] == 1575
    assert long.validation_indices[-1] == 1975


@pytest.mark.parametrize(
    "boundaries", [(0, 2000), (1600, 1600), (2000, 1600), (1600, 2400), (1600.5, 2000)]
)
def test_bad_explicit_partitions_are_rejected(boundaries):
    with pytest.raises(ValueError):
        prepare_data(candles(), LearningConfig(), split_indices=boundaries)


def test_training_gaps_reset_features_and_purge_windows_and_labels():
    """Catch windows, labels or return features spanning an actual exchange outage."""
    frame = candles().drop(candles().index[700:705])
    data = prepare_data(
        frame, LearningConfig(), split_indices=(1700, 2000), allow_training_gaps=True
    )
    assert 675 in data.train_indices
    assert 676 not in data.train_indices
    assert 700 not in data.train_indices
    assert 1034 not in data.train_indices
    assert 1035 in data.train_indices
    assert np.isnan(data.labels[699]).all()
    return_channel = data.feature_names.index("return_168_missing")
    assert data.features[700, return_channel] == 1
    assert data.features[868, return_channel] == 0
    assert len(data.frame) == 2395


def test_validation_or_test_gaps_are_rejected_even_when_training_gaps_allowed():
    frame = candles().drop(candles().index[1800:1805])
    with pytest.raises(ValueError, match="validation|evaluation"):
        prepare_data(frame, LearningConfig(), split_indices=(1600, 2000), allow_training_gaps=True)
