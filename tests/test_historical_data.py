"""Regression tests for historical-data pagination."""

from datetime import datetime
from pathlib import Path
import sys
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import historical_data
from historical_data import HistoricalDataFetcher


class FrozenDateTime(datetime):
    """Clock fixed at a deterministic pagination end time."""

    @classmethod
    def now(cls) -> "FrozenDateTime":
        return cls(2024, 1, 11)


def _page(index: List[pd.Timestamp]) -> pd.DataFrame:
    """Build a complete minimal Binance-style OHLCV page."""
    return pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
            "volume": [1000.0] * len(index),
        },
        index=pd.DatetimeIndex(index),
    )


def test_binance_5m_pagination_resumes_at_next_candle(monkeypatch) -> None:
    """Changing a 5m resume offset to one hour must fail this test."""
    monkeypatch.setattr(historical_data, "datetime", FrozenDateTime)
    monkeypatch.setattr(historical_data.time, "sleep", lambda seconds: None)
    fetcher = HistoricalDataFetcher(data_source="binance", retry_delay=0)
    starts = []

    def fake_fetch(**kwargs) -> pd.DataFrame:
        starts.append(kwargs["start_time"])
        if len(starts) == 1:
            start = pd.Timestamp(kwargs["start_time"])
            return _page([start, start + pd.Timedelta(minutes=5)])
        return _page([pd.Timestamp(FrozenDateTime.now())])

    monkeypatch.setattr(fetcher, "fetch_binance_klines", fake_fetch)
    fetcher._fetch_binance_historical("btc", "5m", days=10, min_data_ratio=0)

    assert pd.Timestamp(starts[1]) == pd.Timestamp(starts[0]) + pd.Timedelta(minutes=10)


def test_binance_failed_page_retries_same_window(monkeypatch) -> None:
    """Advancing after an empty page must fail this test."""
    monkeypatch.setattr(historical_data, "datetime", FrozenDateTime)
    monkeypatch.setattr(historical_data.time, "sleep", lambda seconds: None)
    fetcher = HistoricalDataFetcher(data_source="binance", retry_delay=0)
    starts = []

    def fake_fetch(**kwargs) -> pd.DataFrame:
        starts.append(kwargs["start_time"])
        if len(starts) == 1:
            return pd.DataFrame()
        return _page([pd.Timestamp(FrozenDateTime.now())])

    monkeypatch.setattr(fetcher, "fetch_binance_klines", fake_fetch)
    fetcher._fetch_binance_historical("btc", "5m", days=10, min_data_ratio=0)

    assert len(starts) >= 2
    assert starts[1] == starts[0]
