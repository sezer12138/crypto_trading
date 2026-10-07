"""Regression tests for causal execution, exits, and actual inventory sizing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import BacktestEngine
from run_backtest import create_strategy
from strategies import get_strategy
from strategies._helpers import apply_trend_filter


class IdentityStrategy:
    name = "Identity"

    def generate_signals(self, df):
        return df.copy()


def candles(prices, signals=None):
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1000},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="h"),
    )
    df["signal"] = signals if signals is not None else 0
    return df


def engine(**kwargs):
    return BacktestEngine(
        commission_rate=0,
        slippage=0,
        drawdown_breaker_enabled=False,
        loss_cooldown_enabled=False,
        **kwargs,
    )


def test_orders_fill_at_next_open_without_last_bar_entry():
    df = candles([100.0] * 50, [1, -1] + [0] * 47 + [1])
    df.loc[df.index[1], "open"] = 105
    df.loc[df.index[2], "open"] = 110
    r = engine(min_holding_bars=0).run_backtest(df, IdentityStrategy())
    assert [(t.timestamp, t.price) for t in r.trades] == [(df.index[1], 105), (df.index[2], 110)]


def test_early_exit_is_deferred_without_another_sell_event():
    df = candles([100.0] * 50, [1, -1] + [0] * 48)
    r = engine(min_holding_bars=5).run_backtest(df, IdentityStrategy())
    assert r.trades[1].timestamp == df.index[6]
    assert r.trades[1].strategy_signal == -1


@pytest.mark.parametrize("opening,low,expected", [(100.0, 94.0, 95.0), (90.0, 89.0, 90.0)])
def test_stop_fills_at_threshold_or_worse_gap(opening, low, expected):
    df = candles([100.0] * 50, [1] + [0] * 49)
    df.loc[df.index[2], ["open", "low"]] = [opening, low]
    r = engine(min_holding_bars=0).run_backtest(df, IdentityStrategy())
    assert r.trades[1].timestamp == df.index[2]
    assert r.trades[1].price == expected
    assert r.trades[1].strategy_signal == -2


def test_filter_preserves_sells_and_blocks_buys():
    df = candles([100.0, 100.0, 150.0, 200.0], [0, 0, -1, 1])
    out = apply_trend_filter(df, True, 2, 0.01)
    assert out.signal.tolist()[-2:] == [-1, 0]


def test_mean_reversion_exits_at_mean():
    df = candles([100.0] * 10 + [90.0, 100.0] + [100.0] * 40)
    out = get_strategy(
        "mean_reversion", window=5, entry_z=1.5, trend_filter_enabled=False
    ).generate_signals(df)
    assert out.signal.iloc[10] == 1
    assert out.signal.iloc[11] == -1


@pytest.mark.parametrize("name", ["rsi", "vwap"])
def test_mean_reversion_filters_enabled_by_default(name):
    strategy = get_strategy(name)
    assert strategy.trend_filter_enabled is True
    assert "trend_filter" in strategy.generate_signals(candles([100.0] * 60))


def test_multifactor_is_prefix_invariant():
    rng = np.random.default_rng(42)
    prices = np.r_[
        100 * np.exp(np.cumsum(rng.normal(0, 0.004, 200))),
        100 * np.exp(np.cumsum(rng.normal(0, 0.08, 200))),
    ]
    df = candles(prices)
    strategy = get_strategy("multi_factor")
    short = strategy.generate_signals(df.iloc[:200])
    long = strategy.generate_signals(df)
    pd.testing.assert_series_equal(short.score, long.score.iloc[:200])


def test_grid_warmup_has_no_orders_or_signals():
    df = candles([100.0, 90.0, 110.0, 95.0] * 40)
    strategy = create_strategy("grid", "btc", "1h", df)
    out = strategy.generate_signals(df)
    assert (out.signal.iloc[:100] == 0).all()
    r = engine(min_holding_bars=0, stop_loss_pct=1).run_backtest(df, strategy)
    assert all(t.timestamp >= df.index[100] for t in r.trades)


def test_grid_adds_inventory_and_partially_sells():
    df = candles([109.0, 99.0, 89.0, 99.0, 109.0] + [109.0] * 45)
    r = engine(min_holding_bars=0, stop_loss_pct=1).run_backtest(
        df, get_strategy("grid", lower_price=80, upper_price=120, grid_num=5, amount_per_grid=0.01)
    )
    buys = [t for t in r.trades if t.action == "buy"]
    sells = [t for t in r.trades if t.action == "sell"]
    assert len(buys) == 2
    assert [t.quantity for t in buys] == pytest.approx([0.01, 0.01])
    assert [t.quantity for t in sells] == pytest.approx([0.01, 0.01])
    assert r.metrics["total_round_trips"] == 1


def test_martingale_uses_fills_and_doubles_actual_quantities():
    df = candles([100.0, 100.0, 94.0, 94.0, 110.0, 110.0] + [110.0] * 44)
    r = engine(min_holding_bars=0, stop_loss_pct=1).run_backtest(
        df, get_strategy("martingale", base_amount=0.01, multiplier=2, max_steps=2)
    )
    assert [t.quantity for t in r.trades[:2]] == pytest.approx([0.01, 0.02])
    assert r.trades[2].action == "sell"
    assert r.trades[2].quantity == pytest.approx(0.03)


def test_exact_costs_are_recorded():
    df = candles([100.0] * 50, [1, -1] + [0] * 48)
    r = BacktestEngine(min_holding_bars=0, loss_cooldown_enabled=False).run_backtest(
        df, IdentityStrategy()
    )
    cost = sum(t.commission + t.slippage_cost for t in r.trades)
    assert r.metrics["total_cost"] == round(cost, 2)
    assert r.trades[0].commission == pytest.approx(9.5)


def test_atr_strategy_enables_its_actual_stop():
    strategy = get_strategy("atr_stop")
    assert strategy.use_atr_stop_loss is True


def test_vwap_true_range_includes_gaps():
    df = candles([100.0] * 20 + [120.0] * 20)
    df["high"] = df.close + 1
    df["low"] = df.close - 1
    out = get_strategy("vwap", atr_window=2).calculate_indicators(df)
    assert out.atr.iloc[20] == pytest.approx(11.5)


def test_sized_orders_cannot_exceed_cash_or_exposure_cap():
    df = candles([100.0, 100.0, 90.0, 90.0] + [90.0] * 46)
    r = engine(
        initial_capital=100, min_holding_bars=0, stop_loss_pct=1, log_decisions=True
    ).run_backtest(df, get_strategy("martingale", base_amount=1, multiplier=2))
    for row in r.decision_log:
        assert row["cash"] >= -1e-9
        assert row["position"] * row["price"] <= row["total_value"] * 0.95 + 1e-9


def test_stop_does_not_reenter_on_a_stale_buy_order():
    df = candles([100.0] * 50, [1, 1] + [0] * 48)
    df.loc[df.index[2], ["open", "low"]] = [90, 90]
    r = engine(min_holding_bars=0).run_backtest(df, IdentityStrategy())
    assert [t.action for t in r.trades] == ["buy", "sell"]


def test_partial_exit_statistics_and_report_use_entire_position():
    from backtest import BacktestResult, Trade
    from visualization.html_report import HTMLReportGenerator

    times = pd.date_range("2024-01-01", periods=4, freq="D")
    trades = [
        Trade(times[0], "buy", 100, 1, 100, "BTC", 1),
        Trade(times[1], "buy", 200, 1, 200, "BTC", 1),
        Trade(times[2], "sell", 160, 1, 160, "BTC", -1),
        Trade(times[3], "sell", 160, 1, 160, "BTC", -1),
    ]
    result = BacktestResult(
        trades=trades,
        initial_capital=1000,
        equity_curve=pd.Series([1000, 1000, 1010, 1020], index=times),
        daily_returns=pd.Series([0, 0, 0.01, 0.01], index=times),
    )
    metrics = result.calculate_metrics()
    assert metrics["total_round_trips"] == 1
    assert metrics["win_rate_pct"] == 100
    assert metrics["average_win_pct"] == pytest.approx(6.67)
    report = HTMLReportGenerator()
    assert report._calculate_avg_holding_time(trades) == "3.0 days"
    assert report._calculate_max_profit_loss(trades) == pytest.approx((20 / 300 * 100, 0))


def test_grid_deferred_partial_exits_accumulate_crossed_levels():
    df = candles([109.0, 99.0, 89.0, 99.0, 109.0] + [109.0] * 45)
    r = engine(min_holding_bars=5, stop_loss_pct=1).run_backtest(
        df, get_strategy("grid", lower_price=80, upper_price=120, grid_num=5, amount_per_grid=0.01)
    )
    assert len(r.trades) == 3
    assert r.trades[-1].quantity == pytest.approx(0.02)
    assert r.trades[-1].timestamp == df.index[7]
    assert r.trades[-1].strategy_signal == -1


def test_same_bar_entry_stop_counts_market_exposure():
    df = candles([100.0] * 50, [1] + [0] * 49)
    df.loc[df.index[1], "low"] = 90
    r = engine(min_holding_bars=0).run_backtest(df, IdentityStrategy())
    assert len(r.trades) == 2
    assert r.metrics["market_exposure_pct"] == pytest.approx(2)


def test_martingale_step_limit_forces_exit_before_holding_minimum():
    df = candles([100.0, 100.0, 94.0, 94.0, 85.0, 85.0] + [85.0] * 44)
    r = engine(min_holding_bars=20, stop_loss_pct=1).run_backtest(
        df, get_strategy("martingale", base_amount=0.01, multiplier=2, max_steps=1)
    )
    assert [t.action for t in r.trades[:3]] == ["buy", "buy", "sell"]
    assert r.trades[2].timestamp == df.index[5]
    assert r.trades[2].strategy_signal == -2


def test_legacy_timing_drawdown_includes_initial_entry_costs():
    df = candles([100.0] * 50, [1] + [0] * 49)
    r = BacktestEngine(execution_mode="same_close", loss_cooldown_enabled=False).run_backtest(
        df, IdentityStrategy()
    )
    assert r.metrics["max_drawdown_pct"] == r.metrics["total_return_pct"]
