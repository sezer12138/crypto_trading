"""Tests for command-line backtest configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_backtest


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


def test_main_propagates_disabled_breaker_to_single_run(monkeypatch):
    captured = {}

    def fake_run_single(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--disable-drawdown-breaker"])
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert captured["drawdown_breaker_enabled"] is False


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
