"""
Historical Data Fetcher for Crypto Trading
获取BTC、ETH、SOL的历史数据用于回测
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """获取加密货币历史K线数据"""

    def __init__(self):
        self.binance_base = "https://api.binance.com/api/v3"
        self.coingecko_base = "https://api.coingecko.com/api/v3"

        # 交易对映射
        self.symbols = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}

        # 时间粒度映射 (Binance)
        self.intervals = {
            "1m": "1m",  # 1分钟
            "5m": "5m",  # 5分钟
            "15m": "15m",  # 15分钟
            "1h": "1h",  # 1小时
            "4h": "4h",  # 4小时
            "1d": "1d",  # 1天
        }

    def fetch_binance_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        从Binance获取K线数据

        Args:
            symbol: 交易对 (BTCUSDT, ETHUSDT, etc.)
            interval: 时间粒度
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回条数 (最大1000)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, ...]
        """
        url = f"{self.binance_base}/klines"

        params = {"symbol": symbol, "interval": interval, "limit": limit}

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # 转换为DataFrame
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # 数据类型转换
            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote",
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 时间戳转换
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            # 设置索引
            df.set_index("timestamp", inplace=True)

            logger.info(f"✅ 成功获取 {symbol} {interval} 数据: {len(df)} 条")
            return df

        except Exception as e:
            logger.error(f"❌ 获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        coin: str,
        interval: str = "1h",
        days: int = 730,  # 2年
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指定时间段的历史数据（自动分页）

        Args:
            coin: 币种 (btc, eth, sol)
            interval: 时间粒度
            days: 天数 (默认2年)
            save_path: 保存路径

        Returns:
            DataFrame with historical data
        """
        symbol = self.symbols.get(coin.lower(), f"{coin.upper()}USDT")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        logger.info(f"📊 开始获取 {coin.upper()} 历史数据...")
        logger.info(f"   时间范围: {start_time} ~ {end_time}")
        logger.info(f"   时间粒度: {interval}")

        all_data = []
        current_start = start_time

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
                break

            all_data.append(df)

            # 更新开始时间为最后一条数据的时间
            current_start = df.index[-1] + timedelta(hours=1)

            # 防 rate limit
            time.sleep(0.5)

            # 进度显示
            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"   进度: {progress:.1f}%")

        if not all_data:
            logger.error("❌ 未获取到任何数据")
            return pd.DataFrame()

        # 合并所有数据
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep="first")]
        final_df.sort_index(inplace=True)

        logger.info(f"✅ 共获取 {len(final_df)} 条数据")

        # 保存数据
        if save_path:
            final_df.to_csv(save_path)
            logger.info(f"💾 数据已保存至: {save_path}")

        return final_df

    def get_all_coins_historical(
        self, coins: List[str] = None, interval: str = "1h", days: int = 730
    ) -> Dict[str, pd.DataFrame]:
        """获取多个币种的历史数据"""
        coins = coins or ["btc", "eth", "sol"]
        results = {}

        for coin in coins:
            save_path = f"data/historical/{coin}_{interval}_{days}d.csv"
            df = self.fetch_historical_data(coin, interval, days, save_path)
            results[coin] = df
            time.sleep(1)  # 防限流

        return results


if __name__ == "__main__":
    # 测试
    fetcher = HistoricalDataFetcher()

    # 获取BTC最近30天数据（测试用）
    df = fetcher.fetch_historical_data("btc", interval="1h", days=30)
    print(df.head())
    print(f"\n数据形状: {df.shape}")
