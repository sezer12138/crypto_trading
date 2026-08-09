"""Tests for command-line backtest configuration."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_backtest


def test_momentum_profile_normalizes_key_and_returns_independent_copy(monkeypatch):
    import strategies.momentum_profiles as profiles

    configured = {
        "buy_roc_period": 48,
        "buy_momentum_period": 72,
        "buy_threshold": 0.055,
        "sell_roc_period": 48,
        "sell_momentum_period": 24,
        "sell_threshold": 0.045,
    }
    monkeypatch.setattr(profiles, "MOMENTUM_PROFILES", {("btc", "5m"): configured})

    resolved = profiles.get_momentum_profile(" BTC ", "5M")
    resolved["buy_roc_period"] = 999

    assert profiles.get_momentum_profile("btc", "5m") == configured
    assert profiles.get_momentum_profile("eth", "5m") == {}


def test_create_strategy_applies_matching_momentum_profile(monkeypatch):
    parameters = {
        "buy_roc_period": 48,
        "buy_momentum_period": 72,
        "buy_threshold": 0.055,
        "sell_roc_period": 48,
        "sell_momentum_period": 24,
        "sell_threshold": 0.045,
    }
    monkeypatch.setattr(run_backtest, "get_momentum_profile", lambda coin, interval: parameters)

    strategy = run_backtest.create_strategy(
        "momentum", "btc", "5m", pd.DataFrame({"low": [90.0], "high": [110.0]})
    )

    assert strategy.buy_roc_period == 48
    assert strategy.buy_momentum_period == 72
    assert strategy.sell_momentum_period == 24
    assert strategy.sell_threshold == 0.045


def test_create_strategy_uses_experimental_btc_5m_profile():
    from strategies.momentum_profiles import get_momentum_profile

    assert get_momentum_profile("btc", "5m")
    strategy = run_backtest.create_strategy(
        "momentum", "btc", "5m", pd.DataFrame({"low": [90.0], "high": [110.0]})
    )

    assert (
        strategy.buy_roc_period,
        strategy.buy_momentum_period,
        strategy.buy_threshold,
        strategy.sell_roc_period,
        strategy.sell_momentum_period,
        strategy.sell_threshold,
    ) == (48, 48, 0.055, 144, 12, 0.055)


def test_create_strategy_uses_explicit_btc_1h_profile():
    from strategies.momentum_profiles import get_momentum_profile

    assert get_momentum_profile("btc", "1h")
    strategy = run_backtest.create_strategy(
        "momentum", "BTC", "1H", pd.DataFrame({"low": [90.0], "high": [110.0]})
    )

    assert (
        strategy.buy_roc_period,
        strategy.buy_momentum_period,
        strategy.buy_threshold,
        strategy.sell_roc_period,
        strategy.sell_momentum_period,
        strategy.sell_threshold,
    ) == (16, 12, 0.055, 16, 12, 0.055)


def test_create_strategy_keeps_general_defaults_for_unlisted_coin_interval():
    strategy = run_backtest.create_strategy(
        "momentum", "eth", "5m", pd.DataFrame({"low": [90.0], "high": [110.0]})
    )

    assert (
        strategy.buy_roc_period,
        strategy.buy_momentum_period,
        strategy.buy_threshold,
        strategy.sell_roc_period,
        strategy.sell_momentum_period,
        strategy.sell_threshold,
    ) == (16, 12, 0.055, 16, 12, 0.055)


def test_create_strategy_preserves_grid_range_construction(monkeypatch):
    monkeypatch.setattr(run_backtest, "get_momentum_profile", lambda coin, interval: {})
    data = pd.DataFrame({"low": [90.0, 95.0], "high": [100.0, 110.0]})

    strategy = run_backtest.create_strategy("grid", "btc", "5m", data)

    assert strategy.lower_price == 88.0
    assert strategy.upper_price == 112.0


def test_drawdown_breaker_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    args = run_backtest.parse_arguments()
    assert args.disable_drawdown_breaker is False


def test_disable_drawdown_breaker_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--disable-drawdown-breaker"],
    )
    args = run_backtest.parse_arguments()
    assert args.disable_drawdown_breaker is True


def test_loss_cooldown_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    args = run_backtest.parse_arguments()
    assert args.disable_loss_cooldown is False


def test_disable_loss_cooldown_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--disable-loss-cooldown"],
    )
    args = run_backtest.parse_arguments()
    assert args.disable_loss_cooldown is True


def test_main_propagates_disabled_breaker_to_comparison(monkeypatch):
    captured = {}

    def fake_compare(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--compare", "--disable-drawdown-breaker"],
    )
    monkeypatch.setattr(run_backtest, "compare_strategies", fake_compare)

    run_backtest.main()

    assert captured["drawdown_breaker_enabled"] is False


def test_main_propagates_disabled_loss_cooldown_to_comparison(monkeypatch):
    captured = {}

    def fake_compare(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--compare", "--disable-loss-cooldown"],
    )
    monkeypatch.setattr(run_backtest, "compare_strategies", fake_compare)

    run_backtest.main()

    assert captured["loss_cooldown_enabled"] is False


def test_main_propagates_disabled_breaker_to_single_run(monkeypatch):
    captured = {}

    def fake_run_single(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--disable-drawdown-breaker"])
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert captured["drawdown_breaker_enabled"] is False


def test_main_propagates_disabled_loss_cooldown_to_single_run(monkeypatch):
    captured = {}

    def fake_run_single(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--disable-loss-cooldown"])
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert captured["loss_cooldown_enabled"] is False


def test_main_propagates_disabled_breaker_to_all_coin_runs(monkeypatch):
    captured = []

    def fake_run_single(*args, **kwargs):
        captured.append((args, kwargs))
        return None, None

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--coin", "all", "--no-viz", "--disable-drawdown-breaker"],
    )
    monkeypatch.setattr(run_backtest, "HistoricalDataFetcher", lambda **kwargs: object())
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert [args[0] for args, _ in captured] == ["btc", "eth", "sol"]
    assert all(kwargs["drawdown_breaker_enabled"] is False for _, kwargs in captured)


def test_main_propagates_disabled_loss_cooldown_to_all_coin_runs(monkeypatch):
    captured = []

    def fake_run_single(*args, **kwargs):
        captured.append((args, kwargs))
        return None, None

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--coin", "all", "--no-viz", "--disable-loss-cooldown"],
    )
    monkeypatch.setattr(run_backtest, "HistoricalDataFetcher", lambda **kwargs: object())
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert [args[0] for args, _ in captured] == ["btc", "eth", "sol"]
    assert all(kwargs["loss_cooldown_enabled"] is False for _, kwargs in captured)


def test_comparison_forwards_disabled_breaker_to_every_single_run(monkeypatch):
    captured = []

    def fake_run_single(*args, **kwargs):
        captured.append((args, kwargs))
        return None, None

    monkeypatch.setattr(run_backtest, "HistoricalDataFetcher", lambda **kwargs: object())
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    results = run_backtest.compare_strategies(
        "btc",
        30,
        "1h",
        10000.0,
        save_report=False,
        drawdown_breaker_enabled=False,
    )

    assert results == {}
    assert len(captured) == 13
    assert all(kwargs["drawdown_breaker_enabled"] is False for _, kwargs in captured)


def test_comparison_forwards_disabled_loss_cooldown_to_every_single_run(monkeypatch):
    captured = []

    def fake_run_single(*args, **kwargs):
        captured.append((args, kwargs))
        return None, None

    monkeypatch.setattr(run_backtest, "HistoricalDataFetcher", lambda **kwargs: object())
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    results = run_backtest.compare_strategies(
        "btc",
        30,
        "1h",
        10000.0,
        save_report=False,
        loss_cooldown_enabled=False,
    )

    assert results == {}
    assert len(captured) == 13
    assert all(kwargs["loss_cooldown_enabled"] is False for _, kwargs in captured)


def test_single_run_forwards_disabled_loss_cooldown_to_engine(monkeypatch):
    captured = {}
    dates = pd.date_range("2024-01-01", periods=24, freq="h")
    data = pd.DataFrame(
        {
            "low": [90.0] * 24,
            "high": [110.0] * 24,
            "close": [100.0] * 24,
        },
        index=dates,
    )

    class FakeResult:
        metrics = {}

        def save_logs(self, filepath):
            return None

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_backtest(self, df, strategy, coin):
            return FakeResult()

    monkeypatch.setattr(run_backtest.pd, "read_csv", lambda *args, **kwargs: data)
    monkeypatch.setattr(run_backtest, "create_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_backtest, "BacktestEngine", FakeEngine)

    result, returned_data = run_backtest.run_single_backtest(
        "btc",
        "momentum",
        1,
        "1h",
        10000.0,
        object(),
        generate_html=False,
        loss_cooldown_enabled=False,
    )

    assert result is not None
    assert returned_data is data
    assert captured["loss_cooldown_enabled"] is False
