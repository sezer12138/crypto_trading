"""
Utility functions for crypto trading data fetcher
"""

import json
import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = ["../data/raw", "../data/processed", "../logs", "../config"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_to_json(data: Dict, filename: str, directory: str = "../data/raw"):
    """Save data to JSON file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{directory}/{filename}_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Data saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        return None


def save_to_csv(data: Dict, filename: str, directory: str = "../data/raw"):
    """Save data to CSV file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d")
        filepath = f"{directory}/{filename}_{timestamp}.csv"

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filepath)

        with open(filepath, "a", newline="") as f:
            if data:
                first_item = list(data.values())[0]
                fieldnames = list(first_item.keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                for coin_data in data.values():
                    writer.writerow(coin_data)

        logger.info(f"Data appended to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return None


def format_price(price: float) -> str:
    """Format price with appropriate decimal places."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"


def format_volume(volume: float) -> str:
    """Format large volume numbers."""
    if volume >= 1e9:
        return f"${volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"${volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"${volume/1e3:.2f}K"
    else:
        return f"${volume:.2f}"


def calculate_portfolio_value(holdings: Dict[str, float], prices: Dict[str, float]) -> Dict:
    """Calculate portfolio value based on holdings and current prices.

    Args:
        holdings: Dict of coin -> quantity owned
        prices: Dict of coin -> current price

    Returns:
        Dict with portfolio details
    """
    portfolio = {"total_value": 0.0, "holdings": []}

    for coin, quantity in holdings.items():
        price = prices.get(coin, 0)
        value = quantity * price

        portfolio["holdings"].append(
            {"coin": coin.upper(), "quantity": quantity, "price": price, "value": value}
        )

        portfolio["total_value"] += value

    return portfolio


def generate_alert(coin: str, current_price: float, reference_price: float, threshold: float = 5.0):
    """Generate price change alert.

    Args:
        coin: Coin symbol
        current_price: Current price
        reference_price: Reference price (e.g., entry price or previous price)
        threshold: Alert threshold percentage

    Returns:
        Alert message or None
    """
    change_percent = ((current_price - reference_price) / reference_price) * 100

    if abs(change_percent) >= threshold:
        direction = "📈 UP" if change_percent > 0 else "📉 DOWN"
        return (
            f"ALERT: {coin} {direction} {abs(change_percent):.2f}% (Current: ${current_price:.2f})"
        )

    return None


def load_config(config_path: str = "../config/settings.yaml") -> Dict:
    """Load configuration file."""
    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


class DataCache:
    """Simple in-memory cache for price data."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self._lock = threading.Lock()

    def set(self, key: str, value, ttl: int = 60):
        """Set cache value with TTL (seconds). Thread-safe."""
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict()
            self.cache[key] = {"value": value, "expires": datetime.now().timestamp() + ttl}

    def _evict(self):
        """淘汰缓存条目：优先移除已过期的，否则逐步移除最早到期的直到低于上限。"""
        now = datetime.now().timestamp()
        expired_keys = [k for k, v in self.cache.items() if v["expires"] <= now]
        for k in expired_keys:
            del self.cache[k]
        # 逐步移除最早到期的条目直到低于上限
        while len(self.cache) >= self.max_size and self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["expires"])
            del self.cache[oldest_key]

    def get(self, key: str):
        """Get cached value if not expired. Thread-safe."""
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            if datetime.now().timestamp() < entry["expires"]:
                return entry["value"]
            del self.cache[key]
            return None

    def clear(self):
        """Clear all cached data. Thread-safe."""
        with self._lock:
            self.cache.clear()


if __name__ == "__main__":
    # Test utilities
    ensure_directories()
    print("✅ Directories created")

    # Test formatting
    print(f"Price: {format_price(12345.67)}")
    print(f"Volume: {format_volume(1234567890)}")

    # Test portfolio
    holdings = {"btc": 0.5, "eth": 5.0}
    prices = {"btc": 50000, "eth": 3000}
    portfolio = calculate_portfolio_value(holdings, prices)
    print(f"Portfolio: ${portfolio['total_value']:,.2f}")
