"""Historical archive integrity and timestamp semantics."""

import hashlib
import io
import zipfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from learning.archive import (
    continuous_segments,
    gap_report,
    merge_candles,
    parse_klines,
    verified_zip_rows,
)


def row(timestamp: int, close: str = "101") -> list:
    """Build a complete Binance kline fixture with a hand checked candle."""
    return [
        timestamp,
        "100",
        "105",
        "95",
        close,
        "2",
        timestamp + 3599999,
        "202",
        10,
        "1",
        "101",
        "0",
    ]


def archive(rows: list[list]) -> bytes:
    """Make an actual ZIP, avoiding a mocked parser."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as stream:
        stream.writestr("BTCUSDT-1h-2025-01.csv", "\n".join(",".join(map(str, x)) for x in rows))
    return buffer.getvalue()


def test_archive_checks_hash_before_parsing() -> None:
    payload = archive([row(1735689600000)])
    with pytest.raises(ValueError, match="checksum"):
        verified_zip_rows(payload, "0" * 64)
    checksum = hashlib.sha256(payload).hexdigest()
    assert verified_zip_rows(payload, checksum)[0][0] == "1735689600000"


def test_millisecond_and_microsecond_rows_preserve_same_hour() -> None:
    frame = parse_klines([row(1735689600000), row(1735693200000000)])
    assert list(frame.index) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01 01:00")]
    assert frame.close.tolist() == [101.0, 101.0]


@pytest.mark.parametrize("bad", [row(1735689600001), row(1735689600000, "110")])
def test_invalid_candles_fail_without_silent_repair(bad: list) -> None:
    with pytest.raises(ValueError):
        parse_klines([bad])


def test_merge_rejects_conflicting_duplicate_from_two_sources() -> None:
    first = parse_klines([row(1735689600000)])
    second = parse_klines([row(1735689600000, "102")])
    with pytest.raises(ValueError, match="conflict"):
        merge_candles([first, second], pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))
    merged = merge_candles([first, first], pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))
    assert len(merged) == 1


def test_end_is_exclusive_and_gaps_remain_explicit() -> None:
    frame = parse_klines([row(1735689600000), row(1735696800000), row(1735700400000)])
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01 03:00")
    merged = merge_candles([frame], start, end)
    assert len(merged) == 2
    assert gap_report(merged, start, end) == [
        {"start": "2025-01-01T01:00:00", "end_exclusive": "2025-01-01T02:00:00", "hours": 1}
    ]
    segments = continuous_segments(frame)
    assert [len(segment) for segment in segments] == [1, 2]
    assert segments[-1].index[0] == pd.Timestamp("2025-01-01 02:00")


def test_acquisition_reports_real_outage_and_preserves_verified_cache(
    tmp_path, monkeypatch
) -> None:
    from learning.archive import acquire_btc_history

    payload = archive([row(1735689600000), row(1735696800000)])
    digest = hashlib.sha256(payload).hexdigest()

    class Response:
        status_code = 200

        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    def get(url, timeout):
        if url.endswith(".CHECKSUM"):
            return Response(f"{digest}  BTCUSDT-1h-2025-01.zip".encode())
        if url.endswith(".zip"):
            return Response(payload)
        return Response(b"[]")

    monkeypatch.setattr("learning.archive.requests.get", get)
    output, cache = tmp_path / "data.csv", tmp_path / "cache"
    result = acquire_btc_history("2025-01-01", "2025-01-01 03:00", output, cache)
    assert result["rows"] == 2
    assert result["gaps"][0]["hours"] == 1
    assert result["outputs"]["recent_continuous"]["first"] == "2025-01-01T02:00:00"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result["outputs"]["full"]["sha256"]
    assert (output.with_suffix(".manifest.json")).exists()
    with pytest.raises(FileExistsError):
        acquire_btc_history("2025-01-01", "2025-01-01 03:00", output, cache)
    (cache / "monthly" / "BTCUSDT-1h-2025-01.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        acquire_btc_history("2025-01-01", "2025-01-01 03:00", tmp_path / "other.csv", cache)


def test_archive_quarantines_off_grid_rows_without_rounding_timestamps(
    tmp_path, monkeypatch
) -> None:
    from learning.archive import _archive

    payload = archive([row(1735689600000), row(1735693200001)])
    digest = hashlib.sha256(payload).hexdigest()
    directory = tmp_path / "monthly"
    directory.mkdir()
    (directory / "BTCUSDT-1h-2025-01.zip").write_bytes(payload)
    (directory / "BTCUSDT-1h-2025-01.zip.CHECKSUM").write_text(digest)
    frame, source = _archive("2025-01", "monthly", tmp_path)
    assert len(frame) == 1
    assert source["rejected_off_grid_open_times"] == [1735693200001]
    assert source["accepted_rows"] == 1
