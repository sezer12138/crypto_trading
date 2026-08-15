# BTC 5-Minute Expanded Momentum Search Design

## Goal

Repair deterministic gaps in Binance historical-data pagination, refetch a trustworthy BTC
1800-day 5-minute dataset, and search a materially wider Momentum parameter grid under the same
risk configuration as the user's comparison backtest. Total Return remains the primary objective,
subject to the existing chronological stability and validation checks.

## Evidence and Root Cause

The existing `btc_5m_1800d.csv` spans the requested 1800 days but contains 500,860 rows instead of
approximately 518,400. It has 502 discontinuities and an estimated 17,540 missing candles:

- 499 gaps of exactly one hour;
- one gap of 1 hour 25 minutes;
- one gap of 2 hours 5 minutes;
- one gap of 41 days 17 hours.

Binance pagination advances every successful batch by a fixed one hour, which skips eleven candles
after each 5-minute batch. It also advances a failed request by a fixed 1000 hours, which explains
the 41-day gap. Both behaviors originate in `HistoricalDataFetcher._fetch_binance_historical`.

The current Momentum grid search also controls only the drawdown breaker. It leaves consecutive-loss
cooldown enabled, while the comparison command disabled both mechanisms. Search and comparison
therefore do not currently use equivalent risk settings.

## Historical-Data Repair

Introduce a shared mapping from repository interval labels to candle durations. Binance pagination
must use the selected candle duration for both its request window and its next-page start:

- a successful page resumes at `last_timestamp + candle_duration`;
- an empty or failed page retries the same start time and never skips ahead;
- five consecutive failed outer attempts stop the download rather than creating a silent gap;
- merged data remains sorted and duplicate timestamps are removed.

Regression tests will simulate multiple 5-minute pages and an initially failed page. They must prove
that the next request begins exactly five minutes after the prior page and that failure retries the
same window.

## Refetch and Promotion

Preserve the current dataset as
`data/historical/btc_5m_1800d_pre_pagination_fix.csv`. Download the repaired dataset to a temporary
path first. Do not replace the standard file unless all of these checks pass:

- the timestamp column parses successfully and is strictly increasing;
- duplicate timestamps equal zero;
- the dominant interval is five minutes;
- no repeated one-hour pagination-gap pattern remains;
- completeness is at least 99.9% of the 518,400 expected candles;
- no unexplained multi-day gap remains.

After validation, promote the new file to `data/historical/btc_5m_1800d.csv`. If Binance is
unavailable, stop and report the blocker instead of silently mixing OKX data into the comparison.

## Risk-Control Parity

Add `--disable-loss-cooldown` to `scripts/grid_search_momentum.py`. Convert it to a positive
`loss_cooldown_enabled` setting and pass it to every training, stability, winner-validation, and
current-profile baseline `BacktestEngine` instance. Existing behavior remains enabled by default.

The expanded search command will use both `--disable-drawdown-breaker` and
`--disable-loss-cooldown`, matching the comparison backtest.

## Expanded Grid

Use explicit raw candle counts so the search is specific and reproducible for 5-minute data.

ROC periods:

```text
12,24,48,96,192,432,864,1440,2016,2880,4032
```

These represent 1 hour through 14 days in eleven logarithmically spaced steps.

Momentum periods:

```text
6,12,24,48,96,192,432,864,1440,2016
```

These represent 30 minutes through 7 days in ten steps.

Thresholds:

```text
0.005,0.01,0.02,0.035,0.055,0.08,0.12,0.16,0.20
```

These span 0.5% through 20% in nine steps. The shared grid contains 990 triples. The existing
two-stage process evaluates 990 buy settings and 4,950 sell settings across the top five buy
candidates, for 5,940 primary training evaluations before stability and validation replays.

## Ranking and Validation

Keep the existing chronological 70/30 train-validation split. Rank training candidates primarily
by Total Return, shortlist the top 20, replay them across three ordered stability slices, and reject
candidates that do not complete a round trip in every slice. Evaluate only the highest-ranked
stable candidate on the untouched validation partition and compare it with the active BTC/5m
runtime profile under identical disabled risk controls.

The search result must report:

- separate buy and sell ROC periods, momentum periods, and thresholds;
- training Total Return, Sharpe ratio, maximum drawdown, and trade count;
- stability aggregate and worst-slice returns plus per-slice activity;
- validation Total Return, Sharpe ratio, maximum drawdown, and trade count;
- current-profile validation metrics and adoption-guard outcome.

The production Momentum profile will not be changed automatically. A winning row is the best within
this expanded grid, not proof of future profitability.

## Runtime and Outputs

A benchmark of the present 1800-day file took approximately 7.2 seconds for a one-triple pipeline.
The expanded search is expected to run for roughly 1.5 to 2 hours. Progress output must remain
visible, and the final CSV will be written under `results/` with an explicit BTC, 1800-day, 5-minute,
expanded-search name. Generated data and result files remain outside Git.

## Verification

Run focused pagination and grid-search tests first, followed by the complete test suite. Validate the
refetched CSV before starting the long search. After the search, inspect the CSV schema, row count,
rank ordering, winner metrics, stability eligibility, validation fields, and risk-control parity.
