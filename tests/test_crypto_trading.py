"""
Unit tests for crypto trading data fetcher.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import CryptoDataFetcher
from strategies import (
    MovingAverageCrossStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MultiFactorStrategy,
    MeanReversionStrategy,
    get_strategy,
)
from backtest import BacktestEngine, Trade, BacktestResult


class TestDataFetcher:
    """Test suite for CryptoDataFetcher."""

    def test_initialization(self):
        """Test fetcher initialization."""
        fetcher = CryptoDataFetcher()
        assert fetcher.coins == ["btc", "eth", "sol"]
        assert fetcher.currency == "USD"
        assert "btc" in fetcher.coin_ids
        assert "eth" in fetcher.coin_ids
        assert "sol" in fetcher.coin_ids

    def test_coin_id_mapping(self):
        """Test coin ID mappings."""
        fetcher = CryptoDataFetcher()
        assert fetcher.coin_ids["btc"] == "bitcoin"
        assert fetcher.coin_ids["eth"] == "ethereum"
        assert fetcher.coin_ids["sol"] == "solana"

    def test_symbol_mapping(self):
        """Test symbol mappings."""
        fetcher = CryptoDataFetcher()
        assert fetcher.symbols["btc"] == "BTCUSDT"
        assert fetcher.symbols["eth"] == "ETHUSDT"
        assert fetcher.symbols["sol"] == "SOLUSDT"

    def test_format_coingecko_data(self):
        """Test CoinGecko data formatting."""
        fetcher = CryptoDataFetcher()
        raw_data = {
            "id": "bitcoin",
            "symbol": "btc",
            "current_price": 50000.0,
            "market_cap": 1000000000000,
            "total_volume": 50000000000,
            "high_24h": 51000.0,
            "low_24h": 49000.0,
            "price_change_percentage_1h_in_currency": 0.5,
            "price_change_percentage_24h_in_currency": 2.5,
            "price_change_percentage_7d_in_currency": 5.0,
            "last_updated": "2024-01-01T00:00:00Z",
        }

        formatted = fetcher._format_coingecko_data(raw_data)

        assert formatted["coin"] == "BITCOIN"
        assert formatted["symbol"] == "BTC"
        assert formatted["price"] == 50000.0
        assert formatted["market_cap"] == 1000000000000
        assert formatted["volume_24h"] == 50000000000
        assert formatted["price_change_24h"] == 2.5
        assert formatted["source"] == "coingecko"

    def test_format_binance_data(self):
        """Test Binance data formatting."""
        fetcher = CryptoDataFetcher()
        raw_data = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.00",
            "priceChange": "1000.00",
            "priceChangePercent": "2.00",
            "volume": "1000.00",
            "quoteVolume": "50000000.00",
            "highPrice": "51000.00",
            "lowPrice": "49000.00",
            "openPrice": "49000.00",
            "weightedAvgPrice": "50000.00",
            "bidPrice": "49999.00",
            "askPrice": "50001.00",
            "count": "100000",
        }

        formatted = fetcher._format_binance_data(raw_data, "btc")

        assert formatted["coin"] == "BTC"
        assert formatted["symbol"] == "BTCUSDT"
        assert formatted["price"] == 50000.0
        assert formatted["price_change_percent"] == 2.0
        assert formatted["volume_24h"] == 1000.0
        assert formatted["trades_count"] == 100000
        assert formatted["source"] == "binance"


class TestStrategies:
    """Test suite for trading strategies."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )
        return df

    def test_ma_cross_strategy(self, sample_data):
        """Test Moving Average Cross strategy."""
        strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)
        result = strategy.generate_signals(sample_data)

        assert "ma_short" in result.columns
        assert "ma_long" in result.columns
        assert "signal" in result.columns
        assert "position" in result.columns
        assert result["signal"].isin([-1, 0, 1]).all()

    def test_rsi_strategy(self, sample_data):
        """Test RSI strategy."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.generate_signals(sample_data)

        assert "rsi" in result.columns
        assert "signal" in result.columns
        assert result["rsi"].min() >= 0
        assert result["rsi"].max() <= 100

    def test_bollinger_strategy(self, sample_data):
        """Test Bollinger Bands strategy."""
        strategy = BollingerBandsStrategy(window=20, num_std=2.0)
        result = strategy.generate_signals(sample_data)

        assert "middle_band" in result.columns
        assert "upper_band" in result.columns
        assert "lower_band" in result.columns
        assert "bandwidth" in result.columns
        # Skip NaN values (first window rows)
        valid_data = result.dropna()
        assert (valid_data["upper_band"] > valid_data["lower_band"]).all()

    def test_multi_factor_strategy(self, sample_data):
        """Test Multi-Factor strategy."""
        strategy = MultiFactorStrategy()
        result = strategy.generate_signals(sample_data)

        assert "score" in result.columns
        assert "signal" in result.columns
        # Skip NaN values (first few rows until indicators are calculated)
        valid_scores = result["score"].dropna()
        assert (valid_scores >= -1).all() and (valid_scores <= 1).all()

    def test_mean_reversion_strategy(self, sample_data):
        """Test Mean Reversion strategy."""
        strategy = MeanReversionStrategy(window=20, entry_z=2.0, exit_z=0.5)
        result = strategy.generate_signals(sample_data)

        assert "mean" in result.columns
        assert "std" in result.columns
        assert "zscore" in result.columns

    def test_strategy_factory(self):
        """Test strategy factory function."""
        strategies = [
            "ma_cross",
            "rsi",
            "bollinger",
            "multi_factor",
            "mean_reversion",
        ]

        for name in strategies:
            strategy = get_strategy(name)
            assert strategy is not None
            assert hasattr(strategy, "generate_signals")

        with pytest.raises(ValueError):
            get_strategy("invalid_strategy")


class TestBacktest:
    """Test suite for backtest engine."""

    @pytest.fixture
    def sample_data_with_signals(self):
        """Create sample data with pre-defined signals."""
        dates = pd.date_range("2024-01-01", periods=50, freq="H")
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000, 5000, 50),
                "signal": 0,
            },
            index=dates,
        )

        # Add some buy/sell signals
        df.loc[df.index[10], "signal"] = 1
        df.loc[df.index[20], "signal"] = -1
        df.loc[df.index[30], "signal"] = 1
        df.loc[df.index[40], "signal"] = -1

        return df

    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage=0.001,
            position_size=0.95,
        )

        assert engine.initial_capital == 10000.0
        assert engine.commission_rate == 0.001
        assert engine.slippage == 0.001
        assert engine.position_size == 0.95
        assert engine.cash == 10000.0
        assert engine.position == 0

    def test_trade_creation(self):
        """Test Trade object creation."""
        trade = Trade(
            timestamp=datetime.now(),
            action="buy",
            price=50000.0,
            quantity=0.1,
            value=5000.0,
            coin="BTC",
            strategy_signal=1,
        )

        assert trade.action == "buy"
        assert trade.price == 50000.0
        assert trade.quantity == 0.1
        assert trade.coin == "BTC"

    def test_backtest_result(self):
        """Test BacktestResult object."""
        result = BacktestResult()
        assert len(result.trades) == 0
        assert len(result.decision_log) == 0

        trade = Trade(
            timestamp=datetime.now(),
            action="buy",
            price=50000.0,
            quantity=0.1,
            value=5000.0,
            coin="BTC",
            strategy_signal=1,
        )
        result.add_trade(trade)
        assert len(result.trades) == 1

    def test_simple_backtest(self, sample_data_with_signals):
        """Test simple backtest execution."""

        class SimpleStrategy:
            def __init__(self):
                self.name = "SimpleStrategy"

            def generate_signals(self, df):
                return df

        engine = BacktestEngine(initial_capital=10000.0)
        strategy = SimpleStrategy()

        result = engine.run_backtest(sample_data_with_signals, strategy, coin="TEST")

        assert result is not None
        assert len(result.trades) > 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert "total_return_pct" in result.metrics


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete trading pipeline."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="H")
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000, 5000, 50),
            },
            index=dates,
        )

        # Apply strategy
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=15)
        df_with_signals = strategy.generate_signals(df)

        # Run backtest
        engine = BacktestEngine(initial_capital=10000.0)
        result = engine.run_backtest(df_with_signals, strategy, coin="TEST")

        # Verify results
        assert result is not None
        assert len(result.trades) >= 0
        assert result.metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
