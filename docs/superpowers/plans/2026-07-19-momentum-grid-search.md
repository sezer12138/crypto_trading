# Momentum Grid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested CLI that searches Momentum parameters by training total return and reports chronological validation performance.

**Architecture:** `scripts/grid_search_momentum.py` owns parsing, OHLCV validation, splitting, evaluation, ranking, validation, and output. It reuses `MomentumStrategy` and `BacktestEngine` so execution behavior remains identical to ordinary backtests.

**Tech Stack:** Python 3.10, argparse, itertools, pathlib, pandas, pytest

## Global Constraints

- Rank primarily by training `total_return_pct`.
- Default to a chronological 70/30 split without shuffling.
- Default grids: ROC `5,10,15,20,30`; momentum `5,10,14,20,30`; thresholds `0.005,0.01,0.015,0.02,0.03,0.04`.
- Evaluate all 150 training combinations and only the winner on validation data.
- Keep the drawdown breaker enabled unless explicitly disabled.
- Reuse existing strategy and engine logic; all new prose must be English.

---

### Task 1: Parsing, data validation, and splitting

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/grid_search_momentum.py`
- Create: `tests/test_momentum_grid_search.py`

**Interfaces:**
- Produces: `parse_int_list(value: str) -> List[int]`, `parse_float_list(value: str) -> List[float]`, `load_ohlcv(path: Path) -> pd.DataFrame`, `chronological_split(df, train_ratio, max_lookback) -> Tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write failing tests**

Test successful comma-list parsing, whitespace, duplicate removal, and rejection of empty/nonpositive values. Build ten ordered OHLCV rows and assert a 70/30 split has seven and three rows, no overlap, chronological order, and no input mutation. Test invalid ratios and partitions not longer than the maximum lookback. Test CSV sorting and rejection of missing columns, duplicate/invalid timestamps, and nonnumeric OHLCV data.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_momentum_grid_search.py -v`. Expected: import failure because the module is absent.

- [ ] **Step 3: Implement helpers**

Define `REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")`. Parsers strip items, reject empty/nonpositive values, and preserve order while deduplicating. `load_ohlcv` requires and parses `timestamp`, coerces OHLCV columns with errors raised, rejects duplicates, sorts ascending, and indexes timestamps. The split uses `int(len(df) * train_ratio)`, requires `0 < ratio < 1`, and requires both partitions to contain more rows than `max_lookback`.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/test_momentum_grid_search.py -v`. Expected: Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/grid_search_momentum.py tests/test_momentum_grid_search.py
git commit -m "feat: add momentum grid search data helpers"
```

---

### Task 2: Evaluation, ranking, and validation

**Files:**
- Modify: `scripts/grid_search_momentum.py`
- Modify: `tests/test_momentum_grid_search.py`

**Interfaces:**
- Produces: `evaluate_grid(...) -> pd.DataFrame`, `rank_results(results: pd.DataFrame) -> pd.DataFrame`, `validate_winner(...) -> Dict[str, float]`.

- [ ] **Step 1: Write failing tests**

Test a 2x2x2 grid produces eight unique rows with parameter columns and training total return, annual return, Sharpe, drawdown, win rate, and trades. Monkeypatch `BacktestEngine` and assert a fresh instance per combination plus capital/breaker propagation. Test ranking by descending return, descending Sharpe, descending max-drawdown value (`-10` before `-20`), then ascending parameters. Test that validation runs exactly one fresh engine for the first ranked row and prefixes metrics with `validation_`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_momentum_grid_search.py -v`. Expected: missing evaluation functions fail.

- [ ] **Step 3: Implement evaluation**

Use `itertools.product`. For each tuple instantiate `MomentumStrategy(roc_period=..., momentum_period=..., threshold=...)` and a fresh `BacktestEngine(initial_capital=capital, drawdown_breaker_enabled=...)`, then call `run_backtest`. Extract required metrics. Rank with stable `mergesort` and assign one-based rank. Validate only the first ranked setting on the holdout.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/test_momentum_grid_search.py -v`. Expected: all Task 1-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/grid_search_momentum.py tests/test_momentum_grid_search.py
git commit -m "feat: evaluate momentum parameter grids"
```

---

### Task 3: CLI, CSV, documentation, and end-to-end verification

**Files:**
- Modify: `scripts/grid_search_momentum.py`
- Modify: `tests/test_momentum_grid_search.py`
- Modify: `src/strategy_and_backtest_tutorial.md`

**Interfaces:**
- Produces: `parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace`, `write_results(...) -> pd.DataFrame`, `run_search(args) -> pd.DataFrame`, `main(argv=None) -> int`.

- [ ] **Step 1: Write failing CLI/output tests**

Test exact default grids, ratio `0.7`, capital `10000.0`, and enabled breaker; test overrides. Test CSV parent creation, rank ordering, required schema, and validation fields populated only on rank one. End-to-end, write sufficient synthetic data, run a one-combination search, assert exit zero, one CSV row, and stdout containing winning parameters plus training and validation return. Test invalid input returns nonzero with a clear stderr error.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_momentum_grid_search.py -v`. Expected: missing CLI/output functions fail.

- [ ] **Step 3: Implement CLI and orchestration**

Define default lists and options `--data`, `--roc-periods`, `--momentum-periods`, `--thresholds`, `--train-ratio`, `--capital`, `--output`, `--coin`, and `--disable-drawdown-breaker`. Load/split, train/rank, validate winner, write CSV, and print search size, split ranges, parameters, and returns. Add validation fields only to rank one. Catch `ValueError`, `OSError`, and pandas parser errors, print `Error: ...` to stderr, and return one; use `raise SystemExit(main())`.

- [ ] **Step 4: Update canonical tutorial**

Document the default command, chronological holdout, total-return objective, CSV output, and warning to judge deployment by validation rather than training performance.

- [ ] **Step 5: Full verification**

```bash
black scripts/ tests/test_momentum_grid_search.py --line-length 100 --check
pytest tests/ -v
python scripts/grid_search_momentum.py --help
python scripts/grid_search_momentum.py --data data/historical/btc_1h_730d.csv --roc-periods 10 --momentum-periods 14 --thresholds 0.02 --output /tmp/momentum_grid_smoke.csv
git diff --check
```

Expected: formatting succeeds; all tests pass; help lists all options; smoke search writes one row and prints train/validation returns; whitespace check succeeds.

- [ ] **Step 6: Commit**

```bash
git add scripts/grid_search_momentum.py tests/test_momentum_grid_search.py src/strategy_and_backtest_tutorial.md
git commit -m "feat: add momentum grid search CLI"
```
