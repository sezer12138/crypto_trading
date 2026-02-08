"""
Crypto Trading Data Fetcher
Main module for fetching cryptocurrency trading data.
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """Main class for fetching cryptocurrency data."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the data fetcher with configuration."""
        self.config = self._load_config(config_path)
        self.coins = self.config.get("coins", ["btc", "eth", "sol"])
        self.currency = self.config.get("display", {}).get("currency", "USD")

        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.binance_base = "https://api.binance.com/api/v3"

        # Coin mappings
        self.coin_ids = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana"}

        self.symbols = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "coins": ["btc", "eth", "sol"],
            "update_interval": 30,
            "display": {"currency": "USD"},
        }

    def get_coin_data_coingecko(self, coin: str, max_retries: int = 3) -> Optional[Dict]:
        """Fetch coin data from CoinGecko API with retry logic."""
        import time

        coin_id = self.coin_ids.get(coin.lower(), coin.lower())
        url = f"{self.coingecko_base}/coins/markets"
        params = {
            "vs_currency": self.currency.lower(),
            "ids": coin_id,
            "order": "market_cap_desc",
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)

                # 处理限流
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 2  # 指数退避
                    logger.warning(f"Rate limited by CoinGecko, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if data and len(data) > 0:
                    return self._format_coingecko_data(data[0])
                return None

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.error(f"Error fetching {coin} from CoinGecko: {e}")
                return None

        return None

    def get_coin_data_binance(self, coin: str) -> Optional[Dict]:
        """Fetch coin data from Binance API."""
        try:
            symbol = self.symbols.get(coin.lower(), f"{coin.upper()}USDT")
            url = f"{self.binance_base}/ticker/24hr"
            params = {"symbol": symbol}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return self._format_binance_data(data, coin)

        except requests.RequestException as e:
            logger.error(f"Error fetching {coin} from Binance: {e}")
            return None

    def _format_coingecko_data(self, data: Dict) -> Dict:
        """Format CoinGecko data to standard format."""
        return {
            "coin": data["id"].upper(),
            "symbol": data["symbol"].upper(),
            "price": data["current_price"],
            "market_cap": data["market_cap"],
            "volume_24h": data["total_volume"],
            "high_24h": data.get("high_24h"),
            "low_24h": data.get("low_24h"),
            "price_change_1h": data.get("price_change_percentage_1h_in_currency"),
            "price_change_24h": data.get("price_change_percentage_24h_in_currency"),
            "price_change_7d": data.get("price_change_percentage_7d_in_currency"),
            "last_updated": data["last_updated"],
            "source": "coingecko",
            "timestamp": datetime.now().isoformat(),
        }

    def _format_binance_data(self, data: Dict, coin: str) -> Dict:
        """Format Binance data to standard format."""
        return {
            "coin": coin.upper(),
            "symbol": data["symbol"],
            "price": float(data["lastPrice"]),
            "price_change": float(data["priceChange"]),
            "price_change_percent": float(data["priceChangePercent"]),
            "volume_24h": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
            "open_price": float(data["openPrice"]),
            "weighted_avg_price": float(data["weightedAvgPrice"]),
            "bid_price": float(data["bidPrice"]),
            "ask_price": float(data["askPrice"]),
            "trades_count": int(data["count"]),
            "source": "binance",
            "timestamp": datetime.now().isoformat(),
        }

    def get_all_coins_data(self, source: str = "binance") -> Dict[str, Dict]:
        """Fetch data for all configured coins."""
        results = {}

        for coin in self.coins:
            logger.info(f"Fetching data for {coin.upper()}...")

            if source == "coingecko":
                data = self.get_coin_data_coingecko(coin)
            else:
                data = self.get_coin_data_binance(coin)

            if data:
                results[coin] = data

            # Rate limiting
            time.sleep(1)

        return results

    def display_data(self, data: Dict[str, Dict]):
        """Display formatted data in console."""
        print("\n" + "=" * 80)
        print(f"🚀 Crypto Trading Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for coin, info in data.items():
            print(f"\n📊 {info['coin']} ({info['symbol']})")
            print(f"   💰 Price: ${info['price']:,.2f}")

            if "price_change_24h" in info and info["price_change_24h"] is not None:
                change = info["price_change_24h"]
                emoji = "🟢" if change >= 0 else "🔴"
                print(f"   {emoji} 24h Change: {change:+.2f}%")

            if "volume_24h" in info:
                print(f"   📈 24h Volume: ${info['volume_24h']:,.0f}")

            if "market_cap" in info and info["market_cap"]:
                print(f"   🏦 Market Cap: ${info['market_cap']:,.0f}")

            if "high_24h" in info and info["high_24h"]:
                print(f"   ⬆️  24h High: ${info['high_24h']:,.2f}")
                print(f"   ⬇️  24h Low: ${info['low_24h']:,.2f}")

            if "trades_count" in info:
                print(f"   💫 Trades: {info['trades_count']:,}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test the fetcher
    fetcher = CryptoDataFetcher()
    data = fetcher.get_all_coins_data(source="coingecko")
    fetcher.display_data(data)
