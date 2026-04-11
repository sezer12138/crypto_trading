"""
Extended unit tests for untested strategies and HTML report generation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategies import get_strategy
from backtest import BacktestEngine
from visualization.html_report import HTMLReportGenerator, STRATEGY_DESCRIPTIONS


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    np.random.seed(42)
    base = 100 + np.cumsum(np.random.randn(200) * 2)
    return pd.DataFrame(
        {
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "volume": np.random.randint(1000, 10000, 200),
        },
        index=dates,
    )


# --- Untested Strategies ---


class TestGridStrategy:
    def test_grid_generates_signals(self, sample_data):
        strategy = get_strategy("grid", lower_price=90, upper_price=120)
        result = strategy.generate_signals(sample_data)
        assert "signal" in result.columns
        assert "position" in result.columns
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_grid_respects_bounds(self, sample_data):
        strategy = get_strategy("grid", lower_price=90, upper_price=120, grid_num=5)
        result = strategy.generate_signals(sample_data)
        assert result["position"].iloc[0] == 0.0


class TestMartingaleStrategy:
    def test_martingale_generates_signals(self, sample_data):
        strategy = get_strategy("martingale", base_amount=0.001)
        result = strategy.generate_signals(sample_data)
        assert "signal" in result.columns
        assert "position" in result.columns
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_martingale_position_is_non_negative(self, sample_data):
        strategy = get_strategy("martingale")
        result = strategy.generate_signals(sample_data)
        assert (result["position"] >= 0).all()


class TestATRStopStrategy:
    def test_atr_generates_signals(self, sample_data):
        strategy = get_strategy("atr_stop", atr_period=14, multiplier=2.0)
        result = strategy.generate_signals(sample_data)
        assert "atr" in result.columns
        assert "signal" in result.columns
        assert "position" in result.columns

    def test_atr_values_positive(self, sample_data):
        strategy = get_strategy("atr_stop", atr_period=14)
        result = strategy.generate_signals(sample_data)
        valid_atr = result["atr"].dropna()
        assert (valid_atr >= 0).all()


class TestMACDStrategy:
    def test_macd_generates_signals(self, sample_data):
        strategy = get_strategy("macd")
        result = strategy.generate_signals(sample_data)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "position" in result.columns


class TestBreakoutStrategy:
    def test_breakout_generates_signals(self, sample_data):
        strategy = get_strategy("breakout", window=20)
        result = strategy.generate_signals(sample_data)
        assert "signal" in result.columns
        assert "position" in result.columns


class TestVWAPStrategy:
    def test_vwap_generates_signals(self, sample_data):
        strategy = get_strategy("vwap", window=20, deviation=0.01)
        result = strategy.generate_signals(sample_data)
        assert "vwap" in result.columns
        assert "signal" in result.columns
        assert "position" in result.columns


class TestMomentumStrategy:
    def test_momentum_generates_signals(self, sample_data):
        strategy = get_strategy("momentum", roc_period=10, momentum_period=14)
        result = strategy.generate_signals(sample_data)
        assert "roc" in result.columns
        assert "momentum" in result.columns
        assert "signal" in result.columns


class TestStochasticStrategy:
    def test_stochastic_generates_signals(self, sample_data):
        strategy = get_strategy("stochastic", k_period=14, d_period=3)
        result = strategy.generate_signals(sample_data)
        assert "k" in result.columns
        assert "d" in result.columns
        assert "signal" in result.columns


class TestStrategyFactoryComplete:
    def test_all_strategies_instantiate(self):
        all_names = [
            "ma_cross", "rsi", "bollinger", "multi_factor", "mean_reversion",
            "macd", "breakout", "vwap", "momentum", "atr_stop",
            "stochastic", "grid", "martingale",
        ]
        for name in all_names:
            if name == "grid":
                strategy = get_strategy(name, lower_price=90, upper_price=120)
            else:
                strategy = get_strategy(name)
            assert strategy is not None
            assert hasattr(strategy, "generate_signals")

    def test_all_strategies_run(self, sample_data):
        all_names = [
            "ma_cross", "rsi", "bollinger", "multi_factor", "mean_reversion",
            "macd", "breakout", "vwap", "momentum", "atr_stop",
            "stochastic", "grid", "martingale",
        ]
        for name in all_names:
            if name == "grid":
                strategy = get_strategy(name, lower_price=90, upper_price=120)
            else:
                strategy = get_strategy(name)
            result = strategy.generate_signals(sample_data.copy())
            assert "signal" in result.columns, f"{name} missing signal column"


# --- HTML Report Tests ---


class TestHTMLReport:
    @pytest.fixture
    def backtest_result(self, sample_data):
        strategy = get_strategy("ma_cross", short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=10000)
        return engine.run_backtest(sample_data.copy(), strategy, coin="BTC")

    def test_strategy_descriptions_complete(self):
        expected_strategies = [
            "ma_cross", "rsi", "bollinger", "multi_factor", "mean_reversion",
            "macd", "breakout", "vwap", "momentum", "atr_stop",
            "stochastic", "grid", "martingale",
        ]
        for name in expected_strategies:
            assert name in STRATEGY_DESCRIPTIONS, f"Missing description for {name}"
            desc = STRATEGY_DESCRIPTIONS[name]
            assert "name" in desc
            assert "type" in desc

    def test_single_report_generation(self, sample_data, backtest_result):
        generator = HTMLReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = generator.generate_single_report(
                result=backtest_result,
                df=sample_data,
                strategy_name="ma_cross",
                coin="BTC",
                days=30,
                interval="1h",
                capital=10000,
                output_path=f"{tmpdir}/test_report.html",
            )
            assert Path(output).exists()
            content = Path(output).read_text()
            assert "Backtest Report" in content
            assert "Key Performance Metrics" in content
            assert "Summary & Conclusions" in content

    def test_comparison_report_generation(self, sample_data):
        results = {}
        for name in ["ma_cross", "rsi", "bollinger"]:
            strategy = get_strategy(name)
            engine = BacktestEngine(initial_capital=10000)
            engine.reset()
            results[name] = engine.run_backtest(sample_data.copy(), strategy, coin="BTC")

        generator = HTMLReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = generator.generate_comparison_report(
                results=results,
                coin="BTC",
                days=30,
                interval="1h",
                capital=10000,
                output_path=f"{tmpdir}/test_comparison.html",
            )
            assert Path(output).exists()
            content = Path(output).read_text()
            assert "Strategy Comparison Report" in content
            assert "Best Strategy" in content

    def test_avg_holding_time(self, backtest_result):
        generator = HTMLReportGenerator()
        result = generator._calculate_avg_holding_time(backtest_result.trades)
        # Should return a string (either "N/A" or a time string)
        assert isinstance(result, str)

    def test_max_profit_loss(self, backtest_result):
        generator = HTMLReportGenerator()
        max_profit, max_loss = generator._calculate_max_profit_loss(backtest_result.trades)
        assert isinstance(max_profit, float)
        assert isinstance(max_loss, float)
        assert max_profit >= 0
        assert max_loss <= 0
