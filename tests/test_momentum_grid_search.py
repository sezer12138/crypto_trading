"""Tests for the Momentum strategy grid-search tool."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.grid_search_momentum import (
    chronological_split,
    load_ohlcv,
    parse_float_list,
    parse_int_list,
)


def _ohlcv_frame(periods: int = 10) -> pd.DataFrame:
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


def test_parse_parameter_lists_preserves_order_and_removes_duplicates():
    assert parse_int_list(" 5,10,5,20 ") == [5, 10, 20]
    assert parse_float_list(" 0.005,0.02,0.005 ") == [0.005, 0.02]


@pytest.mark.parametrize("value", ["", "1,,2", "0,2", "-1,2"])
def test_parse_int_list_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_int_list(value)


@pytest.mark.parametrize("value", ["", "0.01,,0.02", "0,0.02", "-0.01,0.02"])
def test_parse_float_list_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_float_list(value)


def test_chronological_split_is_ordered_and_does_not_mutate_input():
    data = _ohlcv_frame()
    original = data.copy(deep=True)

    train, validation = chronological_split(data, train_ratio=0.7, max_lookback=2)

    assert len(train) == 7
    assert len(validation) == 3
    assert train.index.max() < validation.index.min()
    assert train.index.is_monotonic_increasing
    assert validation.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_chronological_split_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="train ratio"):
        chronological_split(_ohlcv_frame(), ratio, max_lookback=2)


def test_chronological_split_rejects_insufficient_history():
    with pytest.raises(ValueError, match="lookback"):
        chronological_split(_ohlcv_frame(10), 0.7, max_lookback=3)


def _write_csv(path: Path, data: pd.DataFrame) -> None:
    data.rename_axis("timestamp").reset_index().to_csv(path, index=False)


def test_load_ohlcv_sorts_timestamps(tmp_path):
    data = _ohlcv_frame().iloc[::-1]
    path = tmp_path / "prices.csv"
    _write_csv(path, data)

    loaded = load_ohlcv(path)

    assert loaded.index.is_monotonic_increasing
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_load_ohlcv_rejects_missing_column(tmp_path):
    path = tmp_path / "prices.csv"
    _write_csv(path, _ohlcv_frame().drop(columns="volume"))
    with pytest.raises(ValueError, match="volume"):
        load_ohlcv(path)


def test_load_ohlcv_rejects_duplicate_timestamps(tmp_path):
    data = pd.concat([_ohlcv_frame(), _ohlcv_frame().iloc[[0]]])
    path = tmp_path / "prices.csv"
    _write_csv(path, data)
    with pytest.raises(ValueError, match="duplicate"):
        load_ohlcv(path)


@pytest.mark.parametrize("column,value,error", [("timestamp", "bad", "timestamp"), ("close", "bad", "numeric")])
def test_load_ohlcv_rejects_invalid_values(tmp_path, column, value, error):
    data = _ohlcv_frame().rename_axis("timestamp").reset_index()
    data.loc[0, column] = value
    path = tmp_path / "prices.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_ohlcv(path)
