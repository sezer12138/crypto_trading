"""
Unit tests for backtest engine risk management features.

Tests cover:
    - Min holding period enforcement
    - Max trades per day enforcement
    - Per-trade stop-loss trigger
    - Drawdown circuit breaker
    - Cost analysis metrics
    - Reset method for risk management state
"""

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine, SIGNAL_BUY, SIGNAL_SELL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class IdentityStrategy:
    """Strategy that returns signals as-is (no processing)."""

    def __init__(self):
        self.name = "IdentityStrategy"

    def generate_signals(self, df):
        return df


def _make_df(prices, start="2024-01-01", freq="h", signals=None):
    """Build a minimal OHLCV DataFrame with optional signal column."""
    dates = pd.date_range(start, periods=len(prices), freq=freq)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1000,
        },
        index=dates,
    )
    if signals is not None:
        df["signal"] = signals
    return df


# ---------------------------------------------------------------------------
# Min holding period
# ---------------------------------------------------------------------------


class TestMinHoldingPeriod:
    """Tests for the minimum holding period constraint."""

    def test_cannot_sell_before_min_bars(self):
        """A sell signal issued before min_holding_bars should be ignored."""
        # Buy at bar 0, try to sell at bar 2 (before default min=5)
        prices = [100.0] * 10
        signals = [SIGNAL_BUY] + [0] * 9
        signals[2] = SIGNAL_SELL
        # A second sell after the holding period to verify it eventually sells
        signals[6] = SIGNAL_SELL
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000, min_holding_bars=5)
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # Should have 1 buy + 1 sell (the sell at bar 2 should be skipped)
        assert len(result.trades) == 2
        assert result.trades[0].action == "buy"
        assert result.trades[1].action == "sell"

    def test_can_sell_after_min_bars(self):
        """A sell signal at exactly min_holding_bars after entry should execute."""
        prices = [100.0] * 10
        signals = [SIGNAL_BUY] + [0] * 9
        signals[5] = SIGNAL_SELL  # exactly 5 bars after buy at bar 0
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000, min_holding_bars=5)
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert len(result.trades) == 2
        assert result.trades[1].action == "sell"

    def test_min_holding_bars_zero_allows_immediate_sell(self):
        """With min_holding_bars=0, a sell on the very next bar should execute."""
        prices = [100.0] * 5
        signals = [SIGNAL_BUY, SIGNAL_SELL, 0, 0, 0]
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000, min_holding_bars=0)
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert len(result.trades) == 2


# ---------------------------------------------------------------------------
# Max trades per day
# ---------------------------------------------------------------------------


class TestMaxTradesPerDay:
    """Tests for the maximum trades per day constraint."""

    def test_respects_max_trades_per_day(self):
        """Trades beyond max_trades_per_day should be skipped within a single day."""
        # Use min_holding_bars=0 so sells can execute immediately after buys.
        # All bars on same day with alternating buy/sell signals.
        n = 20
        prices = [100.0] * n
        signals = [0] * n
        # 4 round-trips = 8 trade actions, then one more buy
        for j in range(4):
            signals[j * 2] = SIGNAL_BUY
            signals[j * 2 + 1] = SIGNAL_SELL
        signals[8] = SIGNAL_BUY  # 9th trade action -- should be blocked (max=6)

        df = _make_df(prices, start="2024-01-01", freq="h", signals=signals)

        engine = BacktestEngine(
            initial_capital=10000, max_trades_per_day=6, min_holding_bars=0
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # max 6 trades in the day: 3 buys + 3 sells = 6 total
        assert len(result.trades) == 6

    def test_new_day_resets_trade_count(self):
        """Trade count should reset on a new day, allowing more trades."""
        # Day 1: 3 round-trips (6 trade actions, max=6)
        # Day 2: 1 more round-trip (should be allowed)
        prices = [100.0] * 48
        signals = [0] * 48
        # Day 1: bars 0-23
        signals[0] = SIGNAL_BUY
        signals[1] = SIGNAL_SELL
        signals[2] = SIGNAL_BUY
        signals[3] = SIGNAL_SELL
        signals[4] = SIGNAL_BUY
        signals[5] = SIGNAL_SELL
        # Day 2: bars 24-47
        signals[24] = SIGNAL_BUY
        signals[25] = SIGNAL_SELL

        df = _make_df(prices, start="2024-01-01", freq="h", signals=signals)

        engine = BacktestEngine(
            initial_capital=10000,
            max_trades_per_day=6,
            min_holding_bars=0,
            max_consecutive_losses=999,  # disable loss cooldown for this test
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # Day 1: 6 trades, Day 2: 2 trades = 8 total
        assert len(result.trades) == 8


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------


class TestStopLoss:
    """Tests for per-trade stop-loss trigger."""

    def test_stop_loss_triggers_on_threshold(self):
        """A price drop exceeding stop_loss_pct should force a sell."""
        # Buy at bar 0 at price 100, price drops to 90 at bar 1 (10% drop > 5% threshold)
        prices = [100.0, 90.0, 90.0, 90.0, 90.0, 90.0]
        signals = [SIGNAL_BUY, 0, 0, 0, 0, 0]
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(
            initial_capital=10000, stop_loss_pct=0.05, min_holding_bars=0
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # Should have 1 buy + 1 forced sell
        assert len(result.trades) == 2
        assert result.trades[1].action == "sell"
        # The forced sell should use signal -2 (force flag)
        assert result.trades[1].strategy_signal == -2

    def test_stop_loss_does_not_trigger_above_threshold(self):
        """A small price drop below stop_loss_pct should not force sell."""
        # Buy at 100, price drops to 97 (3% drop < 5% threshold)
        prices = [100.0, 97.0, 97.0, 97.0, 97.0, 97.0]
        signals = [SIGNAL_BUY, 0, 0, 0, 0, 0]
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(
            initial_capital=10000, stop_loss_pct=0.05, min_holding_bars=0
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # Only the buy + end-of-backtest forced sell (no stop-loss sell)
        sells = [t for t in result.trades if t.action == "sell"]
        # The only sell should come from the end-of-backtest forced close
        assert len(sells) == 1


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------


class TestDrawdownCircuitBreaker:
    """Tests for the max drawdown circuit breaker."""

    def test_explicitly_enabled_breaker_forces_liquidation(self):
        prices = [100.0, 120.0, 90.0, 95.0, 100.0]
        signals = [SIGNAL_BUY, 0, 0, SIGNAL_BUY, SIGNAL_SELL]
        df = _make_df(prices, freq="D", signals=signals)
        engine = BacktestEngine(
            initial_capital=10000,
            drawdown_breaker_enabled=True,
            max_drawdown_pct=0.20,
            stop_loss_pct=1.0,
            min_holding_bars=0,
        )

        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert result.trades[1].timestamp == df.index[2]
        assert result.trades[1].strategy_signal == -2
        assert engine._stopped is True

    def test_disabled_circuit_breaker_does_not_force_liquidation(self):
        prices = [100.0, 120.0, 90.0, 95.0, 100.0]
        signals = [SIGNAL_BUY, 0, 0, 0, SIGNAL_SELL]
        df = _make_df(prices, freq="D", signals=signals)
        engine = BacktestEngine(
            initial_capital=10000,
            drawdown_breaker_enabled=False,
            max_drawdown_pct=0.20,
            stop_loss_pct=1.0,
            min_holding_bars=0,
        )

        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert result.trades[1].timestamp == df.index[4]
        assert result.trades[1].strategy_signal == SIGNAL_SELL
        assert result.metrics["max_drawdown_pct"] < -20.0

    def test_disabled_breaker_keeps_stop_loss_active(self):
        df = _make_df(
            [100.0, 90.0, 90.0],
            signals=[SIGNAL_BUY, 0, 0],
        )
        engine = BacktestEngine(
            drawdown_breaker_enabled=False,
            stop_loss_pct=0.05,
            min_holding_bars=0,
        )

        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert result.trades[1].strategy_signal == -2

    def test_disabled_breaker_configuration_survives_reset_and_reuse(self):
        prices = [100.0, 120.0, 90.0, 95.0, 100.0]
        signals = [SIGNAL_BUY, 0, 0, 0, SIGNAL_SELL]
        df = _make_df(prices, freq="D", signals=signals)
        engine = BacktestEngine(
            initial_capital=10000,
            drawdown_breaker_enabled=False,
            max_drawdown_pct=0.20,
            stop_loss_pct=1.0,
            min_holding_bars=0,
        )

        first_result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")
        engine.reset()
        second_result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert engine.drawdown_breaker_enabled is False
        assert first_result.trades[1].timestamp == df.index[4]
        assert second_result.trades[1].timestamp == df.index[4]
        assert first_result.metrics == second_result.metrics

    def test_disabled_breaker_does_not_log_cooldown(self, caplog):
        caplog.set_level(logging.INFO, logger="backtest")

        BacktestEngine(
            drawdown_breaker_enabled=False,
            breaker_cooldown_bars=10,
        )

        assert "Breaker cooldown" not in caplog.text

    def test_circuit_breaker_stops_trading(self):
        """When drawdown exceeds max_drawdown_pct, all further trading should stop."""
        # Start at 100, rise to 120 (peak), then crash to 90
        # Drawdown from 120 to 90 = 25% > 20% threshold
        prices = [100.0, 120.0, 90.0, 90.0, 90.0]
        signals = [SIGNAL_BUY, 0, SIGNAL_SELL, 0, 0]
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(
            initial_capital=10000,
            max_drawdown_pct=0.20,
            stop_loss_pct=1.0,  # disable stop-loss for this test
            min_holding_bars=0,
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # The sell signal at bar 2 should be blocked by circuit breaker
        # Only the buy at bar 0 + the end-of-backtest forced sell
        assert len(result.trades) == 2  # buy + forced close at end

    def test_circuit_breaker_equity_curve_populated(self):
        """Equity curve should still be fully populated even when circuit breaker is active."""
        prices = [100.0, 120.0, 90.0, 80.0, 70.0]
        signals = [SIGNAL_BUY, 0, 0, 0, 0]
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(
            initial_capital=10000,
            max_drawdown_pct=0.20,
            stop_loss_pct=1.0,
            min_holding_bars=0,
        )
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        # Equity curve should have one entry per bar
        assert len(result.equity_curve) == 5


# ---------------------------------------------------------------------------
# Cost analysis metrics
# ---------------------------------------------------------------------------


class TestCostAnalysis:
    """Tests for cost analysis metrics added to result.metrics."""

    def test_total_cost_and_drag_present(self):
        """Result metrics should include total_cost and cost_drag_pct."""
        prices = [100.0] * 20
        signals = [0] * 20
        signals[0] = SIGNAL_BUY
        signals[10] = SIGNAL_SELL
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000, min_holding_bars=0)
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert "total_cost" in result.metrics
        assert "cost_drag_pct" in result.metrics
        assert result.metrics["total_cost"] > 0
        assert result.metrics["cost_drag_pct"] > 0

    def test_no_trades_zero_cost(self):
        """When there are no trades, total_cost should be 0."""
        prices = [100.0] * 10
        signals = [0] * 10
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000)
        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert result.metrics["total_cost"] == 0.0
        assert result.metrics["cost_drag_pct"] == 0.0


class TestPerformanceAccounting:
    """Regression tests for equity and return-frequency accounting."""

    def test_final_equity_includes_forced_liquidation_costs(self):
        """The last equity point must match cash after end-of-data liquidation."""
        df = _make_df([100.0] * 49, signals=[SIGNAL_BUY] + [0] * 48)
        engine = BacktestEngine(initial_capital=10000, min_holding_bars=0)

        result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

        assert result.equity_curve.iloc[-1] == pytest.approx(engine.cash)
        assert result.metrics["total_return_pct"] < 0

    def test_intraday_equity_is_resampled_before_daily_returns(self):
        """Hourly input should produce one return per completed day, not per bar."""
        df = _make_df([100.0] * 72, signals=[0] * 72)
        result = BacktestEngine().run_backtest(df, IdentityStrategy(), coin="TEST")

        assert len(result.daily_returns) == 2


# ---------------------------------------------------------------------------
# Reset method
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for the reset method covering risk management state."""

    def test_reset_clears_risk_state(self):
        """After reset, all risk management state should return to defaults."""
        engine = BacktestEngine(
            initial_capital=10000,
            min_holding_bars=5,
            max_trades_per_day=6,
            stop_loss_pct=0.05,
            max_drawdown_pct=0.20,
        )
        # Simulate some state
        engine._entry_bar = 10
        engine._trades_today = 3
        engine._current_day = "2024-01-01"
        engine._peak_equity = 15000.0
        engine._stopped = True

        engine.reset()

        assert engine._entry_bar == -1
        assert engine._trades_today == 0
        assert engine._current_day is None
        assert engine._peak_equity == 10000.0
        assert engine._stopped is False

    def test_reset_allows_reuse(self):
        """Engine should produce identical results when run twice after reset."""
        # Use enough bars to span > 1 day so metrics can be calculated
        prices = [100.0] * 50
        signals = [0] * 50
        signals[0] = SIGNAL_BUY
        signals[10] = SIGNAL_SELL
        signals[25] = SIGNAL_BUY
        signals[40] = SIGNAL_SELL
        df = _make_df(prices, signals=signals)

        engine = BacktestEngine(initial_capital=10000, min_holding_bars=0)

        result1 = engine.run_backtest(df.copy(), IdentityStrategy(), coin="TEST")
        engine.reset()
        result2 = engine.run_backtest(df.copy(), IdentityStrategy(), coin="TEST")

        assert len(result1.trades) == len(result2.trades)
        assert result1.equity_curve.iloc[-1] == result2.equity_curve.iloc[-1]


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    """Tests for default risk management parameter values."""

    def test_default_min_holding_bars(self):
        engine = BacktestEngine()
        assert engine.min_holding_bars == 5

    def test_default_max_trades_per_day(self):
        engine = BacktestEngine()
        assert engine.max_trades_per_day == 6

    def test_default_stop_loss_pct(self):
        engine = BacktestEngine()
        assert engine.stop_loss_pct == 0.05

    def test_default_max_drawdown_pct(self):
        engine = BacktestEngine()
        assert engine.max_drawdown_pct == 0.20

    def test_drawdown_breaker_enabled_by_default(self):
        engine = BacktestEngine()
        assert engine.drawdown_breaker_enabled is True

    def test_legacy_positional_breaker_cooldown_argument_is_preserved(self):
        engine = BacktestEngine(
            10000.0,
            0.001,
            0.001,
            0.95,
            5,
            6,
            0.05,
            0.20,
            False,
            False,
            2.0,
            3,
            24,
            10,
        )

        assert engine.breaker_cooldown_bars == 10
        assert engine.drawdown_breaker_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
