"""Tests for the Momentum strategy grid-search tool."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.grid_search_momentum import (
    chronological_split,
    evaluate_buy_grid,
    evaluate_sell_grid,
    evaluate_stability,
    infer_interval_minutes,
    load_ohlcv,
    main,
    parse_arguments,
    parse_float_list,
    parse_int_list,
    quiet_backtest_logs,
    rank_results,
    rank_stable_candidates,
    profile_adoption_passes,
    resolve_search_ranges,
    select_top_buy_candidates,
    split_stability_slices,
    validate_current_profile,
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


def test_infer_interval_minutes_uses_dominant_positive_cadence():
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:10:00",
            "2024-01-01 00:20:00",
            "2024-01-01 00:25:00",
        ]
    )

    assert infer_interval_minutes(index) == 5


def test_infer_interval_minutes_rejects_ambiguous_cadence():
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:30:00",
        ]
    )

    with pytest.raises(ValueError, match="cadence"):
        infer_interval_minutes(index)


def test_resolve_search_ranges_converts_sub_hourly_durations_to_bars():
    args = parse_arguments([])
    data = _ohlcv_frame(500)
    data.index = pd.date_range("2024-01-01", periods=len(data), freq="5min")

    roc_periods, momentum_periods, thresholds = resolve_search_ranges(args, data)

    assert roc_periods == [24, 48, 96, 144, 192, 288, 432]
    assert momentum_periods == [12, 24, 48, 72, 144, 288]
    assert thresholds == [0.015, 0.025, 0.035, 0.045, 0.055]


def test_resolve_search_ranges_preserves_explicit_raw_bar_overrides():
    args = parse_arguments(
        [
            "--roc-periods",
            "3,6",
            "--momentum-periods",
            "4,8",
            "--thresholds",
            "0.01,0.03",
        ]
    )
    data = _ohlcv_frame(100)
    data.index = pd.date_range("2024-01-01", periods=len(data), freq="5min")

    assert resolve_search_ranges(args, data) == ([3, 6], [4, 8], [0.01, 0.03])


class _FakeResult:
    metrics = {
        "total_return_pct": 12.0,
        "annual_return_pct": 6.0,
        "sharpe_ratio": 0.5,
        "max_drawdown_pct": -8.0,
        "win_rate_pct": 55.0,
        "total_trades": 10,
    }


def test_evaluate_buy_grid_runs_every_combination_with_fixed_sell_defaults(monkeypatch):
    import scripts.grid_search_momentum as search

    engines = []

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            engines.append(self)

        def run_backtest(self, df, strategy, coin):
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)

    results = evaluate_buy_grid(
        _ohlcv_frame(40),
        roc_periods=[5, 10],
        momentum_periods=[7, 14],
        thresholds=[0.01, 0.02],
        capital=12345.0,
        drawdown_breaker_enabled=False,
        coin="BTC",
        loss_cooldown_enabled=False,
    )

    assert len(results) == 8
    assert (
        len(results[["buy_roc_period", "buy_momentum_period", "buy_threshold"]].drop_duplicates())
        == 8
    )
    assert len(engines) == 8
    assert all(engine.kwargs["initial_capital"] == 12345.0 for engine in engines)
    assert all(engine.kwargs["drawdown_breaker_enabled"] is False for engine in engines)
    assert all(engine.kwargs["loss_cooldown_enabled"] is False for engine in engines)
    assert set(results.columns) == {
        "buy_roc_period",
        "buy_momentum_period",
        "buy_threshold",
        "sell_roc_period",
        "sell_momentum_period",
        "sell_threshold",
        "train_total_return_pct",
        "train_annual_return_pct",
        "train_sharpe_ratio",
        "train_max_drawdown_pct",
        "train_win_rate_pct",
        "train_total_trades",
    }


def test_select_top_buy_candidates_keeps_five_best_unique_settings():
    results = pd.DataFrame(
        [
            {
                "buy_roc_period": value,
                "buy_momentum_period": 2,
                "buy_threshold": 0.01,
                "train_total_return_pct": float(value),
                "train_sharpe_ratio": 0.0,
                "train_max_drawdown_pct": -10.0,
            }
            for value in range(1, 8)
        ]
    )

    selected = select_top_buy_candidates(results)

    assert selected["buy_roc_period"].tolist() == [7, 6, 5, 4, 3]


def test_evaluate_sell_grid_pairs_each_buy_candidate_with_every_sell_setting(monkeypatch):
    import scripts.grid_search_momentum as search

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def run_backtest(self, df, strategy, coin):
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)
    buy_candidates = pd.DataFrame(
        [
            {"buy_roc_period": 3, "buy_momentum_period": 4, "buy_threshold": 0.01},
            {"buy_roc_period": 5, "buy_momentum_period": 6, "buy_threshold": 0.02},
        ]
    )

    results = evaluate_sell_grid(
        _ohlcv_frame(40),
        buy_candidates,
        roc_periods=[7, 9],
        momentum_periods=[8],
        thresholds=[0.03, 0.04],
        capital=10000.0,
        drawdown_breaker_enabled=True,
        coin="BTC",
    )

    assert len(results) == 8
    assert (
        len(
            results[
                [
                    "buy_roc_period",
                    "buy_momentum_period",
                    "buy_threshold",
                    "sell_roc_period",
                    "sell_momentum_period",
                    "sell_threshold",
                ]
            ].drop_duplicates()
        )
        == 8
    )


def test_rank_results_uses_deterministic_total_return_order():
    results = pd.DataFrame(
        [
            {
                "buy_roc_period": 10,
                "buy_momentum_period": 14,
                "buy_threshold": 0.02,
                "sell_roc_period": 8,
                "sell_momentum_period": 12,
                "sell_threshold": 0.03,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 1,
                "train_max_drawdown_pct": -20,
            },
            {
                "buy_roc_period": 5,
                "buy_momentum_period": 14,
                "buy_threshold": 0.02,
                "sell_roc_period": 8,
                "sell_momentum_period": 12,
                "sell_threshold": 0.03,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 1,
                "train_max_drawdown_pct": -10,
            },
            {
                "buy_roc_period": 5,
                "buy_momentum_period": 10,
                "buy_threshold": 0.01,
                "sell_roc_period": 9,
                "sell_momentum_period": 11,
                "sell_threshold": 0.04,
                "train_total_return_pct": 20,
                "train_sharpe_ratio": 2,
                "train_max_drawdown_pct": -20,
            },
            {
                "buy_roc_period": 5,
                "buy_momentum_period": 10,
                "buy_threshold": 0.02,
                "sell_roc_period": 7,
                "sell_momentum_period": 10,
                "sell_threshold": 0.02,
                "train_total_return_pct": 10,
                "train_sharpe_ratio": 5,
                "train_max_drawdown_pct": -5,
            },
        ]
    )

    ranked = rank_results(results)

    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert ranked[["buy_roc_period", "buy_momentum_period", "buy_threshold"]].values.tolist() == [
        [5.0, 10.0, 0.01],
        [5.0, 14.0, 0.02],
        [10.0, 14.0, 0.02],
        [5.0, 10.0, 0.02],
    ]


def test_split_stability_slices_returns_ordered_non_overlapping_partitions():
    data = _ohlcv_frame(30)

    slices = split_stability_slices(data, slice_count=3, max_lookback=2)

    assert [len(part) for part in slices] == [10, 10, 10]
    assert slices[0].index.max() < slices[1].index.min()
    assert slices[1].index.max() < slices[2].index.min()


def test_evaluate_stability_compounds_returns_and_counts_round_trips(monkeypatch):
    import scripts.grid_search_momentum as search

    candidate = pd.DataFrame(
        [
            {
                "buy_roc_period": 5,
                "buy_momentum_period": 7,
                "buy_threshold": 0.01,
                "sell_roc_period": 6,
                "sell_momentum_period": 8,
                "sell_threshold": 0.02,
                "train_total_return_pct": 20.0,
            }
        ]
    )
    slices = [_ohlcv_frame(10) for _ in range(3)]
    for number, part in enumerate(slices, start=1):
        part.attrs["slice_number"] = number
    returns = {1: 10.0, 2: -5.0, 3: 2.0}
    trades = {1: 2, 2: 4, 3: 2}

    cooldown_settings = []

    def fake_run(
        data,
        parameters,
        capital,
        drawdown_breaker_enabled,
        coin,
        loss_cooldown_enabled=True,
    ):
        cooldown_settings.append(loss_cooldown_enabled)
        number = data.attrs["slice_number"]
        return {
            **parameters,
            "train_total_return_pct": returns[number],
            "train_sharpe_ratio": float(number),
            "train_max_drawdown_pct": -5.0 * number,
            "train_total_trades": trades[number],
        }

    monkeypatch.setattr(search, "_run_combination", fake_run)

    evaluated = evaluate_stability(
        candidate, slices, 10000.0, False, "BTC", loss_cooldown_enabled=False
    )
    row = evaluated.iloc[0]

    assert row["stability_total_return_pct"] == pytest.approx(6.59)
    assert row["stability_worst_return_pct"] == -5.0
    assert row["stability_total_round_trips"] == 4
    assert bool(row["stability_eligible"]) is True
    assert cooldown_settings == [False, False, False]


def test_rank_stable_candidates_excludes_inactive_candidate():
    results = pd.DataFrame(
        [
            {
                "buy_roc_period": 5,
                "buy_momentum_period": 7,
                "buy_threshold": 0.01,
                "sell_roc_period": 6,
                "sell_momentum_period": 8,
                "sell_threshold": 0.02,
                "stability_total_return_pct": 50.0,
                "stability_worst_return_pct": 10.0,
                "stability_mean_sharpe_ratio": 2.0,
                "stability_max_drawdown_pct": -5.0,
                "stability_eligible": False,
            },
            {
                "buy_roc_period": 9,
                "buy_momentum_period": 11,
                "buy_threshold": 0.03,
                "sell_roc_period": 10,
                "sell_momentum_period": 12,
                "sell_threshold": 0.04,
                "stability_total_return_pct": 5.0,
                "stability_worst_return_pct": -2.0,
                "stability_mean_sharpe_ratio": 0.5,
                "stability_max_drawdown_pct": -8.0,
                "stability_eligible": True,
            },
        ]
    )

    ranked = rank_stable_candidates(results)

    assert ranked["buy_roc_period"].tolist() == [9]
    assert ranked["stability_rank"].tolist() == [1]


@pytest.mark.parametrize(
    "candidate,baseline,expected",
    [
        (
            {
                "validation_total_return_pct": 5.0,
                "validation_total_trades": 4,
                "validation_max_drawdown_pct": -30.0,
            },
            {"validation_total_return_pct": 4.0},
            True,
        ),
        (
            {
                "validation_total_return_pct": 4.0,
                "validation_total_trades": 4,
                "validation_max_drawdown_pct": -20.0,
            },
            {"validation_total_return_pct": 4.0},
            False,
        ),
        (
            {
                "validation_total_return_pct": 5.0,
                "validation_total_trades": 2,
                "validation_max_drawdown_pct": -20.0,
            },
            {"validation_total_return_pct": 4.0},
            False,
        ),
        (
            {
                "validation_total_return_pct": 5.0,
                "validation_total_trades": 4,
                "validation_max_drawdown_pct": -30.01,
            },
            {"validation_total_return_pct": 4.0},
            False,
        ),
    ],
)
def test_profile_adoption_guard(candidate, baseline, expected):
    assert profile_adoption_passes(candidate, baseline) is expected


def test_validate_winner_runs_only_best_setting(monkeypatch):
    import scripts.grid_search_momentum as search

    calls = []

    class FakeEngine:
        def __init__(self, **kwargs):
            calls.append(("engine", kwargs))

        def run_backtest(self, df, strategy, coin):
            calls.append(
                (
                    "run",
                    strategy.buy_roc_period,
                    strategy.buy_momentum_period,
                    strategy.buy_threshold,
                    strategy.sell_roc_period,
                    strategy.sell_momentum_period,
                    strategy.sell_threshold,
                )
            )
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "buy_roc_period": 5,
                "buy_momentum_period": 10,
                "buy_threshold": 0.01,
                "sell_roc_period": 7,
                "sell_momentum_period": 12,
                "sell_threshold": 0.03,
            },
            {
                "rank": 2,
                "buy_roc_period": 10,
                "buy_momentum_period": 14,
                "buy_threshold": 0.02,
                "sell_roc_period": 8,
                "sell_momentum_period": 13,
                "sell_threshold": 0.04,
            },
        ]
    )

    metrics = validate_winner(
        _ohlcv_frame(40),
        ranked,
        capital=10000.0,
        drawdown_breaker_enabled=False,
        coin="BTC",
        loss_cooldown_enabled=False,
    )

    assert [call[0] for call in calls] == ["engine", "run"]
    assert calls[0][1]["drawdown_breaker_enabled"] is False
    assert calls[0][1]["loss_cooldown_enabled"] is False
    assert calls[1][1:] == (5, 10, 0.01, 7, 12, 0.03)
    assert metrics["validation_total_return_pct"] == 12.0
    assert set(metrics) == {
        "validation_total_return_pct",
        "validation_annual_return_pct",
        "validation_sharpe_ratio",
        "validation_max_drawdown_pct",
        "validation_win_rate_pct",
        "validation_total_trades",
    }


def test_validate_current_profile_forwards_disabled_risk_controls(monkeypatch):
    import scripts.grid_search_momentum as search

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_backtest(self, df, strategy, coin):
            return _FakeResult()

    monkeypatch.setattr(search, "BacktestEngine", FakeEngine)

    metrics = validate_current_profile(
        _ohlcv_frame(40),
        capital=10000.0,
        drawdown_breaker_enabled=False,
        coin="BTC",
        interval="5m",
        loss_cooldown_enabled=False,
    )

    assert captured["drawdown_breaker_enabled"] is False
    assert captured["loss_cooldown_enabled"] is False
    assert metrics["validation_total_return_pct"] == 12.0


def test_parse_arguments_has_expected_defaults():
    args = parse_arguments([])

    assert args.roc_periods is None
    assert args.momentum_periods is None
    assert args.thresholds is None
    assert args.train_ratio == 0.7
    assert args.capital == 10000.0
    assert args.disable_drawdown_breaker is False
    assert args.disable_loss_cooldown is False


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
            "--disable-loss-cooldown",
        ]
    )

    assert args.roc_periods == [3, 6]
    assert args.momentum_periods == [4, 8]
    assert args.thresholds == [0.01, 0.03]
    assert args.train_ratio == 0.8
    assert args.capital == 5000.0
    assert args.coin == "ETH"
    assert args.disable_drawdown_breaker is True
    assert args.disable_loss_cooldown is True


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
    assert "No deployable winner" in captured.out
    assert "stability_eligible" in pd.read_csv(output_path).columns


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


def test_quiet_backtest_logs_restores_previous_level():
    logger = logging.getLogger("backtest")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        with quiet_backtest_logs():
            assert logger.level == logging.ERROR
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(previous_level)
