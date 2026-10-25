# Verified BTC Hourly History

Use Binance's official spot public archive, preserving exchange identity and raw inputs:

```bash
.venv/bin/python scripts/fetch_btc_archive.py \
  --start 2018-01-01 --end 2026-10-24 \
  --output data/historical/btc_binance_1h_20180101_20261024.csv
```

The end is an exclusive UTC hour. A candle at the current UTC hour is never accepted.
Monthly ZIP files are preferred, then published daily ZIP files for unpublished months.
Every ZIP must match the SHA256 from its official `.CHECKSUM` file before parsing.
Raw archives and checksum files are kept in the ignored `data/.binance_archive` cache.
Existing raw files are reused and reverified; existing output datasets are never overwritten.

Actual missing bars are queried from Binance's market-data-only REST endpoint at
`https://data-api.binance.vision/api/v3/klines`. REST responses are preserved as raw
JSON and carry a local SHA256; unlike published ZIP files, these do not have an
exchange-published checksum. The manifest distinguishes these provenance levels.
No credentials, futures prices, alternative exchanges, or synthetic candles are used.

## Timestamp and Integrity Rules

Binance spot archives use milliseconds before 2025 and microseconds starting in 2025.
The parser detects units from each opening timestamp, avoiding an implicit global unit.
CSV timestamps are timezone-naive UTC candle opens. OHLC prices must be positive and
finite, volume must be nonnegative, and candle high/low geometry must be valid.
Duplicate rows can be collapsed only when all five OHLCV values agree exactly.
Conflicting data abort acquisition rather than selecting one source arbitrarily.

Some early Binance archives contain hourly rows with off-grid opening timestamps.
These rows are rejected from the hourly dataset and recorded individually in each
source's `rejected_off_grid_open_times`; their complete original CSV remains inside
the verified ZIP. Timestamps are never rounded. Official REST is queried for the
resulting aligned-hour gaps, but actual exchange outages can still remain missing.

## Outputs

For output `history.csv`, acquisition writes:

- `history.csv`: all accepted real candles in the requested range; it can contain gaps.
- `history_longest.csv`: the longest contiguous hourly sequence.
- `history_recent.csv`: the most recent contiguous hourly sequence.
- `history.manifest.json`: bounds, UTC acquisition time, source URLs, local cache paths,
  source hashes, accepted/rejected counts, archive gaps before REST recovery, remaining
  gap ranges, continuous segments, and output SHA256 hashes.

All CSV files have columns `timestamp,open,high,low,close,volume`. No missing hourly row
is filled or compressed out of the time axis. The current learning loader requires
continuous hourly observations: use an explicitly selected continuous output, and
record the excluded earlier range in the experiment report. The full output remains
available for future models that reset sequence/account state across real gaps.

## Verification

`tests/test_btc_archive.py` covers checksum rejection, mixed timestamp units, invalid
candles, duplicate conflicts, exclusive bounds, real gaps, contiguous sequence splits,
end-to-end manifest production, immutable outputs, cache tampering, and off-grid
archive quarantine. Network calls are replaced only at the external HTTP boundary;
ZIP creation, parsing, normalization, CSV writing, and hashing remain real operations.

Official source documentation:

- [Binance public data repository](https://github.com/binance/binance-public-data)
- [Binance public archives](https://data.binance.vision/)
- [Binance market-data-only endpoints](https://developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only)

## Acquired Experiment Dataset

The October 24, 2026 acquisition produced 77,111 rows from January 1, 2018 00:00 UTC
through October 23, 2026 23:00 UTC. Its full CSV is 4,924,257 bytes and has SHA256:

```text
8aceb05219632794101bff5317067ee051d6ce5b0f9124db4260fcd36a5b0e56
```

The manifest records 128 checksum-verified ZIPs, 43 rejected off-grid raw rows,
and 27 REST recovery requests. REST supplied 42 actual aligned-hour candles that
were absent from the accepted archive rows. The remaining 27 gap ranges cover
121 hours; the last missing candle opens March 24, 2023 13:00 UTC. The longest
and most recent continuous output is the same 31,426-hour sequence from March 24,
2023 14:00 UTC onward. Full-history model experiments must explicitly handle
training gaps rather than implicitly concatenating adjacent rows across them.
