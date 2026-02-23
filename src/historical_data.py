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

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
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

        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_binance_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        从Binance获取K线数据 (带重试机制)

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

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # 检查返回数据是否为空
                if not data or len(data) == 0:
                    logger.warning(f"⚠️  {symbol} 返回空数据")
                    return pd.DataFrame()

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

            except requests.exceptions.SSLError as e:
                logger.warning(f"⚠️  SSL错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.info(f"⏳  {sleep_time:.1f}秒后重试...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"❌ 获取 {symbol} 数据失败 (SSL错误，已达最大重试次数)")
                    return pd.DataFrame()

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  请求错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"⏳  {sleep_time:.1f}秒后重试...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"❌ 获取 {symbol} 数据失败 (请求错误，已达最大重试次数)")
                    return pd.DataFrame()

            except Exception as e:
                logger.error(f"❌ 获取 {symbol} 数据失败: {e}")
                return pd.DataFrame()

    def fetch_historical_data(
        self,
        coin: str,
        interval: str = "1h",
        days: int = 730,  # 2年
        save_path: Optional[str] = None,
        min_data_ratio: float = 0.8,  # 最少需要获取的数据比例
    ) -> pd.DataFrame:
        """
        获取指定时间段的历史数据（自动分页，带错误处理）

        Args:
            coin: 币种 (btc, eth, sol)
            interval: 时间粒度
            days: 天数 (默认2年)
            save_path: 保存路径
            min_data_ratio: 最少需要获取的数据占预期总数的比例

        Returns:
            DataFrame with historical data
        """
        symbol = self.symbols.get(coin.lower(), f"{coin.upper()}USDT")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 计算预期数据条数
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440
        }
        minutes = interval_minutes.get(interval, 60)
        expected_records = int((days * 24 * 60) / minutes)

        logger.info(f"📊 开始获取 {coin.upper()} 历史数据...")
        logger.info(f"   时间范围: {start_time} ~ {end_time}")
        logger.info(f"   时间粒度: {interval}")
        logger.info(f"   预期数据: 约 {expected_records} 条")

        all_data = []
        current_start = start_time
        failed_attempts = 0
        max_failed_attempts = 5  # 最多允许连续失败5次

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
                logger.warning(f"⚠️  第 {failed_attempts}/{max_failed_attempts} 次获取失败")

                if failed_attempts >= max_failed_attempts:
                    logger.error(f"❌ 连续 {max_failed_attempts} 次获取失败，停止获取")
                    break

                # 跳过当前时间段，尝试获取下一批
                current_start = batch_end
                time.sleep(self.retry_delay)
                continue

            # 重置失败计数
            failed_attempts = 0
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

        # 验证数据完整性
        actual_records = len(final_df)
        data_ratio = actual_records / expected_records

        logger.info(f"✅ 共获取 {actual_records} 条数据 (预期 {expected_records} 条，完成度 {data_ratio*100:.1f}%)")

        if data_ratio < min_data_ratio:
            logger.warning(f"⚠️  数据不完整！仅获取到 {data_ratio*100:.1f}% 的数据，建议重新获取")
        else:
            logger.info(f"✅ 数据完整性检查通过")

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
