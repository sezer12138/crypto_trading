"""Tests for multi-strategy selection and opt-in validated profiles."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strategies import get_strategy


def data():
    rng = np.random.default_rng(31)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.uniform(500, 1500, 300),
        },
        index=pd.date_range("2024-01-01", periods=300, freq="h"),
    )


def test_stochastic_smoothing_and_thresholds_affect_signals():
    df = data()
    a = get_strategy("stochastic", smooth=1, oversold=20, overbought=80).generate_signals(df)
    b = get_strategy("stochastic", smooth=5, oversold=35, overbought=65).generate_signals(df)
    assert not a.k.equals(b.k)
    assert not a.signal.equals(b.signal)


def test_multifactor_thresholds_are_configurable():
    df = data()
    a = get_strategy("multi_factor", buy_threshold=0.3, sell_threshold=0.3).generate_signals(df)
    b = get_strategy("multi_factor", buy_threshold=0.6, sell_threshold=0.6).generate_signals(df)
    assert not a.signal.equals(b.signal)


def test_vwap_dynamic_band_controls_affect_indicators():
    df = data()
    a = get_strategy("vwap", deviation_multiplier=1, min_deviation=0.002).calculate_indicators(df)
    b = get_strategy("vwap", deviation_multiplier=3, min_deviation=0.01).calculate_indicators(df)
    assert not a.dynamic_dev.equals(b.dynamic_dev)


def test_all_search_spaces_create_executable_strategies():
    import optuna
    from optimization.strategy_search import STRATEGIES, suggest_parameters, build_strategy

    assert len(STRATEGIES) == 12
    assert "momentum" not in STRATEGIES
    for name in STRATEGIES:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=12))
        trial = study.ask()
        params = suggest_parameters(trial, name, max_period=96)
        strategy = build_strategy(name, params, data(), 10000)
        assert "signal" in strategy.generate_signals(data())


def test_adoption_guard_rejects_bad_holdout_even_after_good_training():
    from optimization.strategy_search import adoption_checks

    baseline = {"total_return_pct": 10}
    holdout = {"total_return_pct": -1, "max_drawdown_pct": -12, "total_round_trips": 20}
    checks = adoption_checks(holdout, baseline, [10, 20, 30], [4, 4, 4])
    assert not all(checks.values())
    assert checks["positive_holdout"] is False


def test_special_strategy_calibration_does_not_read_future_prices():
    from optimization.strategy_search import build_strategy

    df = data()
    altered = df.copy()
    altered.loc[altered.index[100] :, ["high", "low", "close"]] *= 10
    params = {
        "calibration_bars": 48,
        "range_margin": 0.2,
        "grid_num": 10,
        "inventory_fraction": 0.4,
    }
    a = build_strategy("grid", params, df, 10000)
    b = build_strategy("grid", params, altered, 10000)
    assert (a.lower_price, a.upper_price, a.amount_per_grid) == (
        b.lower_price,
        b.upper_price,
        b.amount_per_grid,
    )


def test_objective_penalizes_missing_round_trips():
    from optimization.strategy_search import training_score

    active = [{"annual_return_pct": 10, "max_drawdown_pct": -10, "total_round_trips": 3}] * 3
    idle = [{"annual_return_pct": 10, "max_drawdown_pct": -10, "total_round_trips": 0}] * 3
    assert training_score(active) > training_score(idle)


def test_selection_never_uses_holdout_and_negative_candidate_is_not_adopted(monkeypatch):
    import optimization.strategy_search as search

    df = data()
    folds = [df.iloc[:80], df.iloc[80:160]]
    holdout = df.iloc[160:]
    calls = []

    def evaluate(partition, name, parameters, config):
        is_holdout = partition is holdout
        calls.append(is_holdout)
        value = -5 if is_holdout else (20 if parameters else 5)
        return {
            "annual_return_pct": value,
            "total_return_pct": value,
            "max_drawdown_pct": -10,
            "total_round_trips": 4,
        }

    monkeypatch.setattr(search, "evaluate_partition", evaluate)
    result = search.optimize_strategy(
        "ma_cross", folds, holdout, search.SearchConfig(), trials=3, seed=42
    )
    assert calls == [False] * 6 + [True] * 2
    assert result["adopted"] is False
    assert result["parameters"]


def test_profile_rejects_mismatched_risk_settings(tmp_path):
    import json
    from optimization.strategy_search import SearchConfig
    from optimization.strategy_profiles import read_profile

    config = SearchConfig()
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "identity": config.identity(),
                "profiles": {"ma_cross": {"short_window": 24, "long_window": 96}},
            }
        )
    )
    assert read_profile(path, "ma_cross", config)["long_window"] == 96
    with pytest.raises(ValueError, match="identity"):
        read_profile(path, "ma_cross", SearchConfig(loss_cooldown_enabled=True))


def test_adoption_checks_are_json_serializable():
    import json
    from optimization.strategy_search import adoption_checks

    metrics = {
        "total_return_pct": np.float64(20),
        "max_drawdown_pct": np.float64(-10),
        "total_round_trips": 3,
    }
    json.dumps(
        adoption_checks(metrics, {"total_return_pct": np.float64(5)}, [10, 20, 30], [3, 3, 3]),
        allow_nan=False,
    )


def test_cli_supports_daily_data_with_adaptive_search_bounds(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.optimize_strategies as cli

    prices = np.linspace(100, 200, 730)
    df = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000},
        index=pd.date_range("2024-01-01", periods=730, freq="D"),
    )
    input_path = tmp_path / "daily.csv"
    df.rename_axis("timestamp").reset_index().to_csv(input_path, index=False)
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_strategies.py",
            "--data",
            str(input_path),
            "--interval",
            "1d",
            "--trials",
            "2",
            "--strategies",
            "ma_cross",
            "--output",
            str(output),
        ],
    )
    cli.main()
    assert (output / "report.html").exists()


def test_martingale_averaging_threshold_stays_inside_engine_stop():
    import optuna
    from optimization.strategy_search import suggest_parameters

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=4))
    for _ in range(10):
        trial = study.ask()
        parameters = suggest_parameters(trial, "martingale")
        assert parameters["stop_loss"] < 0.05
        study.tell(trial, 0)


def test_cli_rejects_known_cache_coin_mismatch(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.optimize_strategies as cli

    path = tmp_path / "btc_1h_730d.csv"
    path.write_text("timestamp,open,high,low,close,volume\n")
    monkeypatch.setattr(
        sys, "argv", ["optimize_strategies.py", "--coin", "eth", "--data", str(path)]
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()
