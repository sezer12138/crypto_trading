"""Verified Binance spot archives with explicit hourly gap provenance.

Raw exchange data are immutable inputs. Missing candles are never synthesized.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
import zipfile

import numpy as np
import pandas as pd
import requests

ARCHIVE_BASE = "https://data.binance.vision/data/spot"
REST_BASE = "https://data-api.binance.vision/api/v3/klines"
COLUMNS = ["open", "high", "low", "close", "volume"]


def verified_zip_rows(payload: bytes, checksum: str) -> list[list[str]]:
    """Reject damaged archives before extracting the single expected CSV."""
    expected = checksum.split()[0].lower()
    if len(expected) != 64 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as stream:
        names = stream.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("Expected exactly one archived CSV")
        return [line.split(",") for line in stream.read(names[0]).decode().splitlines() if line]


def parse_klines(rows: list[list[Any]]) -> pd.DataFrame:
    """Read Binance milliseconds or post-2025 microseconds into naive UTC."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="timestamp"))
    opening = np.array([int(row[0]) for row in rows], dtype=np.int64)
    nanoseconds = np.where(opening >= 100_000_000_000_000, opening * 1000, opening * 1_000_000)
    index = pd.DatetimeIndex(pd.to_datetime(nanoseconds), name="timestamp")
    if (index != index.floor("h")).any() or index.hasnans:
        raise ValueError("Expected aligned hourly open timestamps")
    values = np.array([[float(value) for value in row[1:6]] for row in rows])
    if values.shape != (len(rows), 5) or not np.isfinite(values).all():
        raise ValueError("Invalid finite OHLCV values")
    frame = pd.DataFrame(values, index=index, columns=COLUMNS)
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if (
        (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Invalid OHLC candle geometry")
    return frame


def merge_candles(
    frames: list[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Deduplicate exact matches, reject source conflicts, retain only closed requested bars."""
    nonempty = [frame for frame in frames if len(frame)]
    frame = pd.concat(nonempty).sort_index() if nonempty else frames[0].copy()
    duplicate = frame[frame.index.duplicated(keep=False)]
    if len(duplicate) and (duplicate.groupby(level=0).nunique(dropna=False) > 1).any().any():
        raise ValueError("Candle conflict between sources")
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def gap_report(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    """Describe missing real hourly candles, including both requested boundaries."""
    missing = pd.date_range(start, end, freq="h", inclusive="left").difference(frame.index)
    if len(missing) == 0:
        return []
    groups = np.split(np.arange(len(missing)), np.flatnonzero(np.diff(missing.asi8) != 3600e9) + 1)
    return [
        {
            "start": missing[g[0]].isoformat(),
            "end_exclusive": (missing[g[-1]] + pd.Timedelta(hours=1)).isoformat(),
            "hours": len(g),
        }
        for g in groups
    ]


def continuous_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Return separate genuine continuous sequences without compressing elapsed time."""
    if not len(frame):
        return []
    boundaries = np.flatnonzero(np.diff(frame.index.asi8) != 3600e9) + 1
    return [frame.iloc[indices] for indices in np.split(np.arange(len(frame)), boundaries)]


def _cached_get(url: str, path: Path) -> bytes | None:
    """Reuse immutable files; retry transient transport errors but preserve 404 absence."""
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.content
            try:
                with path.open("xb") as stream:
                    stream.write(payload)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError(f"Immutable cache conflict: {path}")
            return payload
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable download retry state")


def _archive(period: str, cadence: str, cache: Path) -> tuple[pd.DataFrame | None, dict]:
    """Fetch one published archive and its exchange SHA256 checksum."""
    name = f"BTCUSDT-1h-{period}.zip"
    url = f"{ARCHIVE_BASE}/{cadence}/klines/BTCUSDT/1h/{name}"
    checksum_path = cache / cadence / f"{name}.CHECKSUM"
    checksum = _cached_get(f"{url}.CHECKSUM", checksum_path)
    if checksum is None:
        return None, {"url": url, "status": "unavailable"}
    path = cache / cadence / name
    payload = _cached_get(url, path)
    if payload is None:
        raise ValueError(f"Published checksum has no archive: {url}")
    rows = verified_zip_rows(payload, checksum.decode())
    accepted, rejected = [], []
    for row in rows:
        timestamp = int(row[0])
        hour = 3_600_000_000 if timestamp >= 100_000_000_000_000 else 3_600_000
        if timestamp % hour:
            rejected.append(timestamp)
        else:
            accepted.append(row)
    return parse_klines(accepted), {
        "url": url,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "checksum_url": f"{url}.CHECKSUM",
        "checksum_path": str(checksum_path.resolve()),
        "checksum_sha256": hashlib.sha256(checksum).hexdigest(),
        "verification": "exchange_sha256",
        "bytes": len(payload),
        "rows": len(rows),
        "status": "available",
        "accepted_rows": len(accepted),
        "rejected_off_grid_open_times": rejected,
    }


def _rest_gap(start: pd.Timestamp, end: pd.Timestamp, cache: Path) -> tuple[list, list]:
    """Query actual missing candles from the same Binance spot market, never forward fill."""
    frames, sources = [], []
    cursor = start
    while cursor < end:
        stop = min(end, cursor + pd.Timedelta(hours=1000))
        query = urlencode(
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": cursor.value // 1_000_000,
                "endTime": stop.value // 1_000_000 - 1,
                "limit": 1000,
            }
        )
        url = f"{REST_BASE}?{query}"
        key = hashlib.sha256(url.encode()).hexdigest()
        path = cache / "rest" / f"{key}.json"
        payload = _cached_get(url, path)
        if payload is None:
            raise ValueError("Binance REST klines endpoint returned 404")
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected Binance response: {rows}")
        frames.append(parse_klines(rows))
        sources.append(
            {
                "url": url,
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "verification": "local_sha256_https_response",
                "bytes": len(payload),
            }
        )
        cursor = stop
    return frames, sources


def acquire_btc_history(
    start: str, end: str, output: Path, cache: Path, workers: int = 4
) -> dict[str, Any]:
    """Acquire auditable long BTC history and explicitly export continuous alternatives.

    ``end`` is an exclusive UTC hour. Output files are never overwritten. The full
    CSV can have genuine exchange outages; only continuous alternatives are suitable
    for the existing learning loader without further gap-aware model changes.
    """
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("h")
    if first.tz is not None or stop.tz is not None:
        raise ValueError("Use timezone-naive UTC bounds")
    if first != first.floor("h") or stop != stop.floor("h") or first >= stop or stop > now:
        raise ValueError("Bounds must be ordered closed UTC hours, excluding the current candle")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite acquired dataset: {output}")
    months = pd.period_range(first, stop - pd.Timedelta(nanoseconds=1), freq="M")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        monthly = list(pool.map(lambda period: _archive(str(period), "monthly", cache), months))
    frames = [frame for frame, _ in monthly if frame is not None]
    sources = [source for _, source in monthly]
    for period, (frame, _) in zip(months, monthly):
        if frame is not None:
            continue
        days = pd.date_range(
            max(first, period.start_time),
            min(stop, (period + 1).start_time),
            freq="D",
            inclusive="left",
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            daily = list(
                pool.map(lambda day: _archive(day.strftime("%Y-%m-%d"), "daily", cache), days)
            )
        frames.extend(frame for frame, _ in daily if frame is not None)
        sources.extend(source for _, source in daily)
    empty = pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="timestamp"))
    merged = merge_candles(frames or [empty], first, stop)
    archive_gaps = gap_report(merged, first, stop)
    for gap in archive_gaps:
        recovered, rest_sources = _rest_gap(
            pd.Timestamp(gap["start"]), pd.Timestamp(gap["end_exclusive"]), cache
        )
        frames.extend(recovered)
        sources.extend(rest_sources)
    merged = merge_candles(frames or [empty], first, stop)
    if not len(merged):
        raise ValueError("No actual Binance candles available in requested range")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "market": "Binance spot BTCUSDT",
        "interval": "1h",
        "timestamp_timezone": "UTC",
        "timestamp_semantics": "candle open; timezone-naive CSV; end exclusive",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": first.isoformat(),
        "requested_end_exclusive": stop.isoformat(),
        "rows": len(merged),
        "first": merged.index[0].isoformat(),
        "last": merged.index[-1].isoformat(),
        "archive_gaps_before_rest": archive_gaps,
        "gaps": gap_report(merged, first, stop),
        "sources": sources,
        "outputs": {},
        "segments": [],
        "policy": "No synthetic candles, interpolation, timestamp compression, or exchange mixing",
    }
    segments = continuous_segments(merged)
    selected = {
        "full": (output, merged),
        "longest_continuous": (
            output.with_name(output.stem + "_longest.csv"),
            max(segments, key=len),
        ),
        "recent_continuous": (output.with_name(output.stem + "_recent.csv"), segments[-1]),
    }
    for key, (path, frame) in selected.items():
        with path.open("x") as stream:
            frame.to_csv(stream, float_format="%.12g")
        manifest["outputs"][key] = {
            "path": str(path.resolve()),
            "rows": len(frame),
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["segments"] = [
        {
            "first": segment.index[0].isoformat(),
            "last": segment.index[-1].isoformat(),
            "rows": len(segment),
        }
        for segment in segments
    ]
    with output.with_suffix(".manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
