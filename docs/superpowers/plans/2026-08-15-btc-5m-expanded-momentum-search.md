# BTC 5-Minute Expanded Momentum Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Binance 5-minute pagination, refetch and validate 1800 days of BTC data, align grid-search risk controls with comparison backtests, and run the approved 990-triple expanded Momentum search.

**Architecture:** Historical-data pagination derives request and resume offsets from the selected candle interval and retries failed pages without advancing. Momentum grid search receives one additional loss-cooldown Boolean and threads it through every engine execution; the operational phase validates newly fetched data before promotion and then runs one reproducible wide-grid command.

**Tech Stack:** Python 3.10, pandas, requests, argparse, pytest

## Global Constraints

- Total Return remains the primary search objective.
- Preserve chronological 70/30 validation and three ordered stability slices.
- Existing risk controls remain enabled by default.
- The expanded search disables both drawdown breaker and consecutive-loss cooldown.
- Preserve the existing dataset before promoting newly fetched data.
- Do not mix Binance and OKX data in one comparison.
- Generated historical data and search results remain outside Git.
- All code comments, docstrings, and log messages remain in English.
- Do not stage the untracked `AGENTS.md` file.

---

### Task 1: Interval-aware Binance pagination

**Files:**
- Create: `tests/test_historical_data.py`
- Modify: `src/historical_data.py`

**Interfaces:**
- Consumes: Repository interval labels `1m`, `5m`, `15m`, `1h`, `4h`, and `1d`.
- Produces: `INTERVAL_MINUTES: Dict[str, int]` and gap-free page advancement inside `HistoricalDataFetcher._fetch_binance_historical`.

- [ ] **Step 1: Write failing pagination tests**

Create a frozen clock, a minimal OHLCV page helper, and these tests:

```python
from datetime import datetime

import pandas as pd

import historical_data
from historical_data import HistoricalDataFetcher


class FrozenDateTime(datetime):
    @classmethod
    def now(cls):
        return cls(2024, 1, 11)


def _page(index):
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


def test_binance_5m_pagination_resumes_at_next_candle(monkeypatch):
    monkeypatch.setattr(historical_data, "datetime", FrozenDateTime)
    monkeypatch.setattr(historical_data.time, "sleep", lambda seconds: None)
    fetcher = HistoricalDataFetcher(data_source="binance", retry_delay=0)
    starts = []

    def fake_fetch(**kwargs):
        starts.append(kwargs["start_time"])
        if len(starts) == 1:
            start = pd.Timestamp(kwargs["start_time"])
            return _page([start, start + pd.Timedelta(minutes=5)])
        return _page([pd.Timestamp(FrozenDateTime.now())])

    monkeypatch.setattr(fetcher, "fetch_binance_klines", fake_fetch)
    fetcher._fetch_binance_historical("btc", "5m", days=10, min_data_ratio=0)

    assert pd.Timestamp(starts[1]) == pd.Timestamp(starts[0]) + pd.Timedelta(minutes=10)


def test_binance_failed_page_retries_same_window(monkeypatch):
    monkeypatch.setattr(historical_data, "datetime", FrozenDateTime)
    monkeypatch.setattr(historical_data.time, "sleep", lambda seconds: None)
    fetcher = HistoricalDataFetcher(data_source="binance", retry_delay=0)
    starts = []

    def fake_fetch(**kwargs):
        starts.append(kwargs["start_time"])
        if len(starts) == 1:
            return pd.DataFrame()
        return _page([pd.Timestamp(FrozenDateTime.now())])

    monkeypatch.setattr(fetcher, "fetch_binance_klines", fake_fetch)
    fetcher._fetch_binance_historical("btc", "5m", days=10, min_data_ratio=0)

    assert len(starts) >= 2
    assert starts[1] == starts[0]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest tests/test_historical_data.py -v
```

Expected: the resume test observes a one-hour jump, and the failed-page test observes only one call
because the current implementation skips to the end of the request window.

- [ ] **Step 3: Implement interval-aware pagination**

Add the module constant:

```python
INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}
```

In `_fetch_binance_historical`, derive the duration once:

```python
minutes = INTERVAL_MINUTES.get(interval, 60)
candle_duration = timedelta(minutes=minutes)
```

Use a window sized for at most 1000 candles:

```python
batch_end = min(current_start + candle_duration * 1000, end_time)
```

On an empty page, increment the failure counter, sleep, and retry without assigning a new
`current_start`. After a successful page, advance exactly one candle:

```python
current_start = df.index[-1] + candle_duration
```

Filter the merged frame to the requested timestamps before completeness checks:

```python
final_df = final_df.loc[
    (final_df.index >= pd.Timestamp(start_time))
    & (final_df.index <= pd.Timestamp(end_time))
]
```

- [ ] **Step 4: Verify pagination tests**

Run:

```bash
pytest tests/test_historical_data.py -v
```

Expected: both pagination regressions pass.

- [ ] **Step 5: Commit pagination repair**

```bash
git add src/historical_data.py tests/test_historical_data.py
git commit -m "fix: preserve Binance candle continuity"
```

---

### Task 2: Loss-cooldown parity in Momentum grid search

**Files:**
- Modify: `tests/test_momentum_grid_search.py`
- Modify: `scripts/grid_search_momentum.py`

**Interfaces:**
- Consumes: `BacktestEngine(loss_cooldown_enabled: bool = True)`.
- Produces: `--disable-loss-cooldown` and a `loss_cooldown_enabled: bool` argument propagated through every grid-search engine path.

- [ ] **Step 1: Write failing parsing and engine-propagation tests**

Extend argument tests:

```python
def test_parse_arguments_has_expected_defaults():
    args = parse_arguments([])
    assert args.disable_loss_cooldown is False


def test_parse_arguments_accepts_overrides(tmp_path):
    args = parse_arguments(["--disable-loss-cooldown"])
    assert args.disable_loss_cooldown is True
```

Extend the fake-engine assertions in the buy-grid test. Pass
`loss_cooldown_enabled=False` to `evaluate_buy_grid` and assert:

```python
assert all(engine.kwargs["loss_cooldown_enabled"] is False for engine in engines)
```

Extend winner validation to call:

```python
metrics = validate_winner(
    data,
    ranked,
    capital=10000.0,
    drawdown_breaker_enabled=False,
    loss_cooldown_enabled=False,
    coin="BTC",
)
```

Then assert the captured engine kwargs contain both disabled settings.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest tests/test_momentum_grid_search.py -v
```

Expected: parsing lacks `disable_loss_cooldown`, and affected functions reject the new keyword.

- [ ] **Step 3: Implement the CLI flag and propagation**

Add:

```python
parser.add_argument(
    "--disable-loss-cooldown",
    action="store_true",
    help="Disable pausing new entries after consecutive losing trades",
)
```

Add `loss_cooldown_enabled: bool` beside `drawdown_breaker_enabled: bool` in
`_run_combination`, `evaluate_buy_grid`, `evaluate_sell_grid`, `evaluate_stability`,
`validate_winner`, and `validate_current_profile`. Every internal call must pass the value by
keyword or in the same declared order. Construct every engine with:

```python
engine = BacktestEngine(
    initial_capital=capital,
    drawdown_breaker_enabled=drawdown_breaker_enabled,
    loss_cooldown_enabled=loss_cooldown_enabled,
)
```

In `run_search`, derive:

```python
breaker_enabled = not args.disable_drawdown_breaker
loss_cooldown_enabled = not args.disable_loss_cooldown
```

Pass both settings through training, stability, winner validation, and baseline validation.

- [ ] **Step 4: Verify grid-search tests and help**

Run:

```bash
pytest tests/test_momentum_grid_search.py -v
python scripts/grid_search_momentum.py --help | rg -- "--disable-loss-cooldown"
```

Expected: all grid-search tests pass and help lists the new flag.

- [ ] **Step 5: Commit risk-control parity**

```bash
git add scripts/grid_search_momentum.py tests/test_momentum_grid_search.py
git commit -m "feat: align momentum search risk controls"
```

---

### Task 3: Documentation and full code verification

**Files:**
- Modify: `README.md`
- Modify: `src/strategy_and_backtest_tutorial.md`

**Interfaces:**
- Consumes: Corrected Binance pagination and the new grid-search flag.
- Produces: Reproducible commands and data-quality guidance.

- [ ] **Step 1: Update documentation**

Document that Binance pagination is interval-aware and retries failed pages without skipping time.
Add this exact expanded-search command:

```bash
python scripts/grid_search_momentum.py \
  --data data/historical/btc_5m_1800d.csv \
  --roc-periods 12,24,48,96,192,432,864,1440,2016,2880,4032 \
  --momentum-periods 6,12,24,48,96,192,432,864,1440,2016 \
  --thresholds 0.005,0.01,0.02,0.035,0.055,0.08,0.12,0.16,0.20 \
  --train-ratio 0.7 \
  --coin btc \
  --disable-drawdown-breaker \
  --disable-loss-cooldown \
  --output results/momentum_grid_search_btc_1800d_5m_expanded.csv
```

- [ ] **Step 2: Format and inspect**

Run:

```bash
black src/historical_data.py scripts/grid_search_momentum.py tests/test_historical_data.py tests/test_momentum_grid_search.py --line-length 100
git diff --check
```

Expected: Black succeeds and `git diff --check` produces no output.

- [ ] **Step 3: Run the complete suite**

Run:

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md src/strategy_and_backtest_tutorial.md
git commit -m "docs: explain expanded 5m momentum search"
```

---

### Task 4: Refetch and validate BTC 1800-day 5-minute data

**Files:**
- Preserve: `data/historical/btc_5m_1800d_pre_pagination_fix.csv`
- Create temporarily: `data/historical/btc_5m_1800d_refetching.csv`
- Replace after validation: `data/historical/btc_5m_1800d.csv`

**Interfaces:**
- Consumes: Corrected `HistoricalDataFetcher` from Task 1 and Binance network access.
- Produces: Validated standard BTC 5-minute historical data for Task 5.

- [ ] **Step 1: Preserve the existing dataset**

Run:

```bash
cp -p data/historical/btc_5m_1800d.csv \
  data/historical/btc_5m_1800d_pre_pagination_fix.csv
```

- [ ] **Step 2: Fetch to a temporary path**

Run with unbuffered logs:

```bash
python -u -c 'import sys; sys.path.insert(0, "src"); from historical_data import HistoricalDataFetcher; HistoricalDataFetcher(data_source="binance").fetch_historical_data("btc", "5m", 1800, "data/historical/btc_5m_1800d_refetching.csv", min_data_ratio=0.999)'
```

Expected: the download completes without five consecutive failures and writes approximately
518,400 candles.

- [ ] **Step 3: Validate the temporary CSV**

Run a read-only pandas check that prints rows, start/end timestamps, duplicates, dominant cadence,
non-5-minute gaps, estimated missing candles, maximum gap, and completeness. Required results:

```text
duplicates = 0
dominant cadence = 5 minutes
completeness >= 99.9%
repeated one-hour gaps = 0
maximum gap < 1 day
```

- [ ] **Step 4: Promote only validated data**

Run only after Step 3 passes:

```bash
mv data/historical/btc_5m_1800d_refetching.csv \
  data/historical/btc_5m_1800d.csv
```

Re-run the same validation against the promoted standard path and verify identical statistics.

---

### Task 5: Run and inspect the expanded Momentum search

**Files:**
- Create: `results/momentum_grid_search_btc_1800d_5m_expanded.csv`

**Interfaces:**
- Consumes: Validated dataset from Task 4 and risk-control parity from Task 2.
- Produces: Ranked expanded-grid results with stability, validation, baseline, and adoption fields.

- [ ] **Step 1: Run the approved expanded search**

Run:

```bash
python -u scripts/grid_search_momentum.py \
  --data data/historical/btc_5m_1800d.csv \
  --roc-periods 12,24,48,96,192,432,864,1440,2016,2880,4032 \
  --momentum-periods 6,12,24,48,96,192,432,864,1440,2016 \
  --thresholds 0.005,0.01,0.02,0.035,0.055,0.08,0.12,0.16,0.20 \
  --train-ratio 0.7 \
  --coin btc \
  --disable-drawdown-breaker \
  --disable-loss-cooldown \
  --output results/momentum_grid_search_btc_1800d_5m_expanded.csv
```

Expected startup output: `Searching 5940 parameter combinations (990 buy + 4950 sell)`.

- [ ] **Step 2: Inspect the result CSV**

Verify:

- the file exists and has at most 20 shortlisted rows;
- stability ranks are ascending for eligible rows;
- the first eligible row has all six parameter fields;
- every stability slice has at least one completed round trip;
- validation and baseline-validation fields are populated only for the winner;
- `profile_adoption_passed` is present when a stable winner exists;
- printed winner values match the first CSV row.

- [ ] **Step 3: Report the result without changing production defaults**

Report the winning buy and sell parameters, training/stability/validation metrics, active-profile
baseline comparison, adoption outcome, dataset continuity statistics, elapsed search time, and any
boundary warning. Do not edit `src/strategies/momentum_profiles.py` unless the user explicitly asks
after reviewing the evidence.
