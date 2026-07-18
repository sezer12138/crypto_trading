"""Tests for the Momentum strategy grid-search tool."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.grid_search_momentum import (
    chronological_split,
    evaluate_grid,
    load_ohlcv,
    main,
    parse_arguments,
    parse_float_list,
    parse_int_list,
    rank_results,
    validate_winner,
    write_results,
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


@pytest.mark.parametrize(
    "value",
    ["", "0.01,,0.02", "0,0.02", "-0.01,0.02", "nan,0.02", "inf,0.02"],
)
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


@pytest.mark.parametrize(
    "column,value,error", [("timestamp", "bad", "timestamp"), ("close", "bad", "numeric")]
)
def test_load_ohlcv_rejects_invalid_values(tmp_path, column, value, error):
    data = _ohlcv_frame().rename_axis("timestamp").reset_index()
    data[column] = data[column].astype(object)
    data.loc[0, column] = value
    path = tmp_path / "prices.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_ohlcv(path)


def test_load_ohlcv_rejects_non_finite_values(tmp_path):
    data = _ohlcv_frame()
    data.loc[data.index[0], "close"] = float("nan")
    path = tmp_path / "prices.csv"
    _write_csv(path, data)
    with pytest.raises(ValueError, match="finite"):
        load_ohlcv(path)


class _FakeResult:
    metrics = {
        "total_return_pct": 12.0,
        "annual_return_pct": 6.0,
        "sharpe_ratio": 0.5,
        "max_drawdown_pct": -8.0,
        "win_rate_pct": 55.0,
        "total_trades": 10,
    }


def test_evaluate_grid_runs_every_combination_with_fresh_engine(monkeypatch):
    import scripts.grid_search_momentum as search

    engines = []

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            engines.append(self)

        def run_backtest(self, df, strategy, coin):
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)

    results = evaluate_grid(
        _ohlcv_frame(40),
        roc_periods=[5, 10],
        momentum_periods=[7, 14],
        thresholds=[0.01, 0.02],
        capital=12345.0,
        drawdown_breaker_enabled=False,
        coin="BTC",
    )

    assert len(results) == 8
    assert len(results[["roc_period", "momentum_period", "threshold"]].drop_duplicates()) == 8
    assert len(engines) == 8
    assert all(engine.kwargs["initial_capital"] == 12345.0 for engine in engines)
    assert all(engine.kwargs["drawdown_breaker_enabled"] is False for engine in engines)
    assert set(results.columns) == {
        "roc_period",
        "momentum_period",
        "threshold",
        "train_total_return_pct",
        "train_annual_return_pct",
        "train_sharpe_ratio",
        "train_max_drawdown_pct",
        "train_win_rate_pct",
        "train_total_trades",
    }


def test_rank_results_uses_deterministic_total_return_order():
    results = pd.DataFrame(
        [
            {
                "roc_period": 10,
                "momentum_period": 14,
                "threshold": 0.02,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 1,
                "train_max_drawdown_pct": -20,
            },
            {
                "roc_period": 5,
                "momentum_period": 14,
                "threshold": 0.02,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 1,
                "train_max_drawdown_pct": -10,
            },
            {
                "roc_period": 5,
                "momentum_period": 10,
                "threshold": 0.01,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 2,
                "train_max_drawdown_pct": -20,
            },
            {
                "roc_period": 5,
                "momentum_period": 10,
                "threshold": 0.02,
                "train_total_return_pct": 10,
                "train_sharpe_ratio": 5,
                "train_max_drawdown_pct": -5,
            },
        ]
    )

    ranked = rank_results(results)

    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert ranked[["roc_period", "momentum_period", "threshold"]].values.tolist() == [
        [5.0, 10.0, 0.01],
        [5.0, 14.0, 0.02],
        [10.0, 14.0, 0.02],
        [5.0, 10.0, 0.02],
    ]


def test_validate_winner_runs_only_best_setting(monkeypatch):
    import scripts.grid_search_momentum as search

    calls = []

    class FakeEngine:
        def __init__(self, **kwargs):
            calls.append(("engine", kwargs))

        def run_backtest(self, df, strategy, coin):
            calls.append(("run", strategy.roc_period, strategy.momentum_period, strategy.threshold))
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)
    ranked = pd.DataFrame(
        [
            {"rank": 1, "roc_period": 5, "momentum_period": 10, "threshold": 0.01},
            {"rank": 2, "roc_period": 10, "momentum_period": 14, "threshold": 0.02},
        ]
    )

    metrics = validate_winner(
        _ohlcv_frame(40), ranked, capital=10000.0, drawdown_breaker_enabled=True, coin="BTC"
    )

    assert [call[0] for call in calls] == ["engine", "run"]
    assert calls[1][1:] == (5, 10, 0.01)
    assert metrics["validation_total_return_pct"] == 12.0
    assert set(metrics) == {
        "validation_total_return_pct",
        "validation_annual_return_pct",
        "validation_sharpe_ratio",
        "validation_max_drawdown_pct",
        "validation_win_rate_pct",
        "validation_total_trades",
    }


def test_parse_arguments_has_expected_defaults():
    args = parse_arguments([])

    assert args.roc_periods == [5, 10, 15, 20, 30]
    assert args.momentum_periods == [5, 10, 14, 20, 30]
    assert args.thresholds == [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
    assert args.train_ratio == 0.7
    assert args.capital == 10000.0
    assert args.disable_drawdown_breaker is False


def test_parse_arguments_accepts_overrides(tmp_path):
    args = parse_arguments(
        [
            "--data",
            str(tmp_path / "input.csv"),
            "--roc-periods",
            "3,6",
            "--momentum-periods",
            "4,8",
            "--thresholds",
            "0.01,0.03",
            "--train-ratio",
            "0.8",
            "--capital",
            "5000",
            "--output",
            str(tmp_path / "output.csv"),
            "--coin",
            "ETH",
            "--disable-drawdown-breaker",
        ]
    )

    assert args.roc_periods == [3, 6]
    assert args.momentum_periods == [4, 8]
    assert args.thresholds == [0.01, 0.03]
    assert args.train_ratio == 0.8
    assert args.capital == 5000.0
    assert args.coin == "ETH"
    assert args.disable_drawdown_breaker is True


def test_write_results_populates_validation_only_for_winner(tmp_path):
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "roc_period": 5,
                "momentum_period": 10,
                "threshold": 0.01,
                "train_total_return_pct": 20.0,
            },
            {
                "rank": 2,
                "roc_period": 10,
                "momentum_period": 14,
                "threshold": 0.02,
                "train_total_return_pct": 10.0,
            },
        ]
    )
    validation = {
        "validation_total_return_pct": 5.0,
        "validation_sharpe_ratio": 0.4,
    }
    output = tmp_path / "nested" / "results.csv"

    written = write_results(ranked, validation, output)
    loaded = pd.read_csv(output)

    assert output.exists()
    assert written["rank"].tolist() == [1, 2]
    assert loaded.loc[0, "validation_total_return_pct"] == 5.0
    assert pd.isna(loaded.loc[1, "validation_total_return_pct"])
    assert "validation_sharpe_ratio" in loaded.columns


def test_main_runs_one_combination_end_to_end(tmp_path, capsys):
    input_path = tmp_path / "prices.csv"
    output_path = tmp_path / "results.csv"
    _write_csv(input_path, _ohlcv_frame(120))

    exit_code = main(
        [
            "--data",
            str(input_path),
            "--roc-periods",
            "5",
            "--momentum-periods",
            "7",
            "--thresholds",
            "0.01",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert len(pd.read_csv(output_path)) == 1
    assert "Winning parameters" in captured.out
    assert "Training total return" in captured.out
    assert "Validation total return" in captured.out


def test_main_reports_invalid_input(tmp_path, capsys):
    input_path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"], "close": [100]}).to_csv(input_path, index=False)

    exit_code = main(["--data", str(input_path), "--output", str(tmp_path / "out.csv")])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err


def test_main_rejects_non_finite_capital(tmp_path, capsys):
    input_path = tmp_path / "prices.csv"
    _write_csv(input_path, _ohlcv_frame(120))

    exit_code = main(
        [
            "--data",
            str(input_path),
            "--capital",
            "nan",
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )

    assert exit_code == 1
    assert "capital" in capsys.readouterr().err.lower()
