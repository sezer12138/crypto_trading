"""
Historical Data Fetcher for Crypto Trading
Fetches historical data for BTC, ETH, SOL for backtesting
Supports multiple exchanges: Binance (default) and OKX (accessible from China)
"""

import os
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json

from utils import get_proxy, get_binance_base_url, get_coingecko_base_url

logger = logging.getLogger(__name__)

# Default data source: auto-detect from environment or use binance
DEFAULT_DATA_SOURCE = os.environ.get("CRYPTO_DATA_SOURCE", "binance")

# OKX instrument ID mappings
OKX_INSTRUMENTS = {
    "btc": "BTC-USDT",
    "eth": "ETH-USDT",
    "sol": "SOL-USDT",
}

# OKX interval mappings
OKX_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

OKX_BASE_URL = "https://www.okx.com/api/v5"


class HistoricalDataFetcher:
    """Fetches cryptocurrency historical K-line data from multiple exchanges"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verify_ssl: bool = True,
        data_source: str = None,
    ):
        self.data_source = data_source or DEFAULT_DATA_SOURCE
        self.binance_base = get_binance_base_url()
        self.coingecko_base = get_coingecko_base_url()
        self.proxies = get_proxy()

        if self.proxies:
            logger.info(f"Using proxy: {list(self.proxies.values())[0]}")

        logger.info(f"Data source: {self.data_source}")

        # Symbol mappings (Binance format)
        self.symbols = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}

        # Interval mappings (Binance)
        self.intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # SSL verification: only disabled when environment variable CRYPTO_DISABLE_SSL=1
        self.verify_ssl = verify_ssl and os.environ.get("CRYPTO_DISABLE_SSL") != "1"

        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL verification disabled (CRYPTO_DISABLE_SSL=1), ensure network environment is secure")

    def fetch_okx_candles(
        self,
        instrument_id: str,
        interval: str = "1H",
        after: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch K-line data from OKX

        OKX API returns candles in reverse chronological order.
        Each candle: [timestamp, open, high, low, close, volume, ...]

        Args:
            instrument_id: OKX instrument ID (e.g., "BTC-USDT")
            interval: Bar size (1m, 5m, 15m, 1H, 4H, 1D)
            after: Pagination token (timestamp ms of the earliest record to fetch before)
            limit: Number of candles (max 100)

        Returns:
            DataFrame with standardized columns
        """
        url = f"{OKX_BASE_URL}/market/history-candles"
        params = {"instId": instrument_id, "bar": interval, "limit": limit}

        if after:
            params["after"] = after

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url, params=params, timeout=30,
                    verify=self.verify_ssl, proxies=self.proxies,
                )
                response.raise_for_status()
                result = response.json()

                if result.get("code") != "0":
                    logger.warning(f"OKX API error: {result.get('msg', 'unknown')}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return pd.DataFrame()

                data = result.get("data", [])
                if not data:
                    return pd.DataFrame()

                # OKX returns: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                # Data is reverse chronological (newest first)
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close",
                    "volume", "vol_ccy", "vol_ccy_quote", "confirm",
                ])

                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
                df.sort_values("timestamp", inplace=True)
                df.set_index("timestamp", inplace=True)

                # Keep only standard columns
                df = df[["open", "high", "low", "close", "volume"]]

                logger.info(f"OKX: fetched {instrument_id} {interval} data: {len(df)} records")
                return df

            except requests.exceptions.RequestException as e:
                logger.warning(f"OKX request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch {instrument_id} from OKX (max retries reached)")
                    return pd.DataFrame()

            except Exception as e:
                logger.error(f"Failed to fetch {instrument_id} from OKX: {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    def fetch_binance_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch K-line data from Binance (with retry mechanism)

        Args:
            symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
            interval: Time interval
            start_time: Start time
            end_time: End time
            limit: Maximum number of records (max 1000)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, ...]
        """
        url = f"{self.binance_base}/klines"

        params = {"symbol": symbol, "interval": interval, "limit": limit}

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30, verify=self.verify_ssl, proxies=self.proxies)
                response.raise_for_status()
                data = response.json()

                # Check for empty data
                if not data or len(data) == 0:
                    logger.warning(f"{symbol} returned empty data")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades",
                        "taker_buy_base", "taker_buy_quote", "ignore",
                    ],
                )

                # Type conversion
                numeric_cols = [
                    "open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base", "taker_buy_quote",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Timestamp conversion
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

                # Set index
                df.set_index("timestamp", inplace=True)

                logger.info(f"Binance: fetched {symbol} {interval} data: {len(df)} records")
                return df

            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch {symbol} data (SSL error, max retries reached)")
                    return pd.DataFrame()

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch {symbol} data (request error, max retries reached)")
                    return pd.DataFrame()

            except Exception as e:
                logger.error(f"Failed to fetch {symbol} data: {e}")
                return pd.DataFrame()

    def fetch_historical_data(
        self,
        coin: str,
        interval: str = "1h",
        days: int = 730,
        save_path: Optional[str] = None,
        min_data_ratio: float = 0.8,
    ) -> pd.DataFrame:
        """
        Fetch historical data for a specified time period (auto-paginated)

        Automatically selects the configured data source (binance or okx).

        Args:
            coin: Coin symbol (btc, eth, sol)
            interval: Time interval
            days: Number of days (default 2 years)
            save_path: Save path
            min_data_ratio: Minimum required data ratio relative to expected total

        Returns:
            DataFrame with historical data
        """
        if self.data_source == "okx":
            return self._fetch_okx_historical(coin, interval, days, save_path, min_data_ratio)
        else:
            return self._fetch_binance_historical(coin, interval, days, save_path, min_data_ratio)

    def _fetch_okx_historical(
        self,
        coin: str,
        interval: str = "1h",
        days: int = 730,
        save_path: Optional[str] = None,
        min_data_ratio: float = 0.8,
    ) -> pd.DataFrame:
        """Fetch historical data from OKX (paginated, reverse chronological)"""
        instrument_id = OKX_INSTRUMENTS.get(coin.lower(), f"{coin.upper()}-USDT")
        okx_bar = OKX_INTERVALS.get(interval, "1H")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = interval_minutes.get(interval, 60)
        expected_records = int((days * 24 * 60) / minutes)

        logger.info(f"Starting to fetch {coin.upper()} historical data from OKX...")
        logger.info(f"   Time range: {start_time} ~ {end_time}")
        logger.info(f"   Interval: {interval} (OKX bar: {okx_bar})")
        logger.info(f"   Expected records: approximately {expected_records}")

        all_data = []
        failed_attempts = 0
        max_failed_attempts = 5
        # OKX returns newest first; we paginate backwards using "after" (earliest ts from previous batch)
        after_ts = None

        total_fetched = 0

        while True:
            df = self.fetch_okx_candles(
                instrument_id=instrument_id,
                interval=okx_bar,
                after=after_ts,
                limit=100,
            )

            if df.empty:
                failed_attempts += 1
                logger.warning(f"Fetch attempt {failed_attempts}/{max_failed_attempts} failed")
                if failed_attempts >= max_failed_attempts:
                    logger.error(f"Consecutive {max_failed_attempts} fetch failures, stopping")
                    break
                time.sleep(self.retry_delay)
                continue

            failed_attempts = 0

            # Check if we've gone past our start_time
            if df.index[0] < pd.Timestamp(start_time):
                # Trim data before start_time
                df = df[df.index >= pd.Timestamp(start_time)]
                if not df.empty:
                    all_data.append(df)
                    total_fetched += len(df)
                break

            all_data.append(df)
            total_fetched += len(df)

            # Update after to the earliest timestamp in this batch (paginate backwards)
            after_ts = str(int(df.index[0].timestamp() * 1000))

            # Rate limit prevention
            time.sleep(0.2)

            # Progress display
            progress = min(total_fetched / expected_records * 100, 100)
            logger.info(f"   Progress: {progress:.1f}% ({total_fetched}/{expected_records})")

            # Stop if we got fewer than limit (no more data available)
            if len(df) < 100:
                break

        if not all_data:
            logger.error("No data fetched from OKX")
            return pd.DataFrame()

        # Merge all data
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep="first")]
        final_df.sort_index(inplace=True)

        # Validate data completeness
        actual_records = len(final_df)
        data_ratio = actual_records / expected_records

        logger.info(f"Fetched {actual_records} records in total (expected {expected_records}, completeness {data_ratio*100:.1f}%)")

        if data_ratio < min_data_ratio:
            logger.warning(f"Data incomplete! Only fetched {data_ratio*100:.1f}% of data, recommend re-fetching")
        else:
            logger.info("Data completeness check passed")

        # Save data
        if save_path:
            final_df.to_csv(save_path)
            logger.info(f"Data saved to: {save_path}")

        return final_df

    def _fetch_binance_historical(
        self,
        coin: str,
        interval: str = "1h",
        days: int = 730,
        save_path: Optional[str] = None,
        min_data_ratio: float = 0.8,
    ) -> pd.DataFrame:
        """Fetch historical data from Binance (paginated, forward chronological)"""
        symbol = self.symbols.get(coin.lower(), f"{coin.upper()}USDT")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = interval_minutes.get(interval, 60)
        expected_records = int((days * 24 * 60) / minutes)

        logger.info(f"Starting to fetch {coin.upper()} historical data from Binance...")
        logger.info(f"   Time range: {start_time} ~ {end_time}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Expected records: approximately {expected_records}")

        all_data = []
        current_start = start_time
        failed_attempts = 0
        max_failed_attempts = 5

        while current_start < end_time:
            batch_end = min(current_start + timedelta(hours=1000), end_time)

            df = self.fetch_binance_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=batch_end,
                limit=1000,
            )

            if df.empty:
                failed_attempts += 1
                logger.warning(f"Fetch attempt {failed_attempts}/{max_failed_attempts} failed")

                if failed_attempts >= max_failed_attempts:
                    logger.error(f"Consecutive {max_failed_attempts} fetch failures, stopping")
                    break

                current_start = batch_end
                time.sleep(self.retry_delay)
                continue

            failed_attempts = 0
            all_data.append(df)

            current_start = df.index[-1] + timedelta(hours=1)

            # Rate limit prevention
            time.sleep(0.5)

            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"   Progress: {progress:.1f}%")

        if not all_data:
            logger.error("No data fetched from Binance")
            return pd.DataFrame()

        # Merge all data
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep="first")]
        final_df.sort_index(inplace=True)

        # Validate data completeness
        actual_records = len(final_df)
        data_ratio = actual_records / expected_records

        logger.info(f"Fetched {actual_records} records in total (expected {expected_records}, completeness {data_ratio*100:.1f}%)")

        if data_ratio < min_data_ratio:
            logger.warning(f"Data incomplete! Only fetched {data_ratio*100:.1f}% of data, recommend re-fetching")
        else:
            logger.info("Data completeness check passed")

        # Save data
        if save_path:
            final_df.to_csv(save_path)
            logger.info(f"Data saved to: {save_path}")

        return final_df

    def get_all_coins_historical(
        self, coins: List[str] = None, interval: str = "1h", days: int = 730
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple coins"""
        coins = coins or ["btc", "eth", "sol"]
        results = {}

        for coin in coins:
            save_path = f"data/historical/{coin}_{interval}_{days}d.csv"
            df = self.fetch_historical_data(coin, interval, days, save_path)
            results[coin] = df
            time.sleep(1)

        return results


if __name__ == "__main__":
    # Test
    fetcher = HistoricalDataFetcher(data_source="okx")
    df = fetcher.fetch_historical_data("btc", interval="1h", days=30)
    print(df.head())
    print(f"\nData shape: {df.shape}")
