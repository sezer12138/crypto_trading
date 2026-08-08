# Interval-Aware Momentum Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve Momentum parameters by coin and interval and prevent sparse, overfit grid-search winners from becoming runtime profiles.

**Architecture:** A focused profile module supplies optional six-parameter mappings to the backtest runner. The grid-search script infers candle duration for omitted ranges, shortlists candidates with its existing two-stage search, then ranks only sufficiently active candidates across three chronological stability slices before touching the outer validation set.

**Tech Stack:** Python 3.10, pandas, argparse, pytest

## Global Constraints

- Keep Total Return as the primary optimization objective.
- Keep one shared set of CLI ranges for buy and sell parameters.
- Explicit CLI periods continue to mean raw candle counts.
- Profiles must be keyed by normalized coin and interval.
- Do not promote a profile without validation improvement, at least two validation round trips, and validation drawdown no worse than -30%.
- Keep comments, docstrings, and logs in English.
- Do not stage the untracked `AGENTS.md` file.

---

### Task 1: Runtime Momentum profile resolution

**Files:**
- Create: `src/strategies/momentum_profiles.py`
- Modify: `run_backtest.py`
- Modify: `tests/test_run_backtest.py`

**Interfaces:**
- Produces: `get_momentum_profile(coin: str, interval: str) -> Dict[str, Union[int, float]]`
- Produces: `create_strategy(strategy_name: str, coin: str, interval: str, df: pd.DataFrame) -> TradingStrategy`

- [ ] **Step 1: Write failing behavioral tests** proving normalization and fallback return independent dictionaries, and proving `create_strategy()` applies a BTC/5m profile only to Momentum while preserving grid and martingale construction.
- [ ] **Step 2: Verify RED** with `pytest tests/test_run_backtest.py -v`; expect missing profile module or helper failures.
- [ ] **Step 3: Implement the profile module and strategy-construction helper.** The mapping starts empty unless the final robust search passes the adoption guard. `run_single_backtest()` delegates its strategy creation to the helper.
- [ ] **Step 4: Verify GREEN** with `pytest tests/test_run_backtest.py -v`.

### Task 2: Interval-aware default search ranges

**Files:**
- Modify: `scripts/grid_search_momentum.py`
- Modify: `tests/test_momentum_grid_search.py`

**Interfaces:**
- Produces: `infer_interval_minutes(index: pd.DatetimeIndex) -> int`
- Produces: `resolve_search_ranges(args: argparse.Namespace, data: pd.DataFrame) -> Tuple[List[int], List[int], List[float]]`

- [ ] **Step 1: Write failing tests** for 5-minute interval inference, rejection of timestamps without a dominant positive cadence, conversion to literal ROC periods `[24, 48, 96, 144, 192, 288, 432]`, momentum periods `[12, 24, 48, 72, 144, 288]`, and thresholds `[0.015, 0.025, 0.035, 0.045, 0.055]`.
- [ ] **Step 2: Add a failing explicit-override test** proving all three supplied CLI lists bypass generated defaults.
- [ ] **Step 3: Verify RED** with the new focused grid-search tests.
- [ ] **Step 4: Implement inference and range resolution.** Change the three argparse defaults to `None`; hourly data resolves to the existing 4,800-combination grids and sub-hourly data resolves from fixed hour durations.
- [ ] **Step 5: Update `run_search()` to resolve ranges before lookback validation and evaluation counts, then verify GREEN.**

### Task 3: Stability ranking and adoption guard

**Files:**
- Modify: `scripts/grid_search_momentum.py`
- Modify: `tests/test_momentum_grid_search.py`

**Interfaces:**
- Produces: `split_stability_slices(train_df, slice_count, max_lookback) -> List[pd.DataFrame]`
- Produces: `evaluate_stability(candidates, slices, capital, drawdown_breaker_enabled, coin) -> pd.DataFrame`
- Produces: `rank_stable_candidates(results: pd.DataFrame) -> pd.DataFrame`
- Produces: `profile_adoption_passes(candidate_metrics, baseline_metrics) -> bool`

- [ ] **Step 1: Write failing tests** for three ordered slices, compounded literal returns, completed-round-trip counts, one-round-trip-per-slice eligibility, and deterministic ranking.
- [ ] **Step 2: Write failing guard tests** covering validation improvement, two-round-trip minimum, and -30% drawdown boundary.
- [ ] **Step 3: Verify RED** with the new focused tests.
- [ ] **Step 4: Implement shortlist and stability evaluation.** Keep the top 20 stage-two candidates, evaluate each on three independent slices, and rank eligible candidates by compounded Total Return followed by the specified tie breakers.
- [ ] **Step 5: Integrate final winner and baseline validation.** If no candidate is eligible, write diagnostic rows and report no deployable winner without failing the command.
- [ ] **Step 6: Verify GREEN** with `pytest tests/test_momentum_grid_search.py -v`.

### Task 4: Documentation, real search, and profile decision

**Files:**
- Modify: `README.md`
- Modify: `src/strategy_and_backtest_tutorial.md`
- Conditionally modify: `src/strategies/momentum_profiles.py`
- Conditionally modify: `tests/test_run_backtest.py`

**Interfaces:**
- Documents: interval-aware defaults, stability slices, adoption criteria, and profile fallback

- [ ] **Step 1: Update documentation** with the runtime resolution and robust-selection behavior.
- [ ] **Step 2: Run the robust BTC/5m search** with `python scripts/grid_search_momentum.py --data data/historical/btc_5m_360d.csv --coin BTC --disable-drawdown-breaker --output results/momentum_grid_search_btc_360d_5m_robust.csv`.
- [ ] **Step 3: Apply the adoption guard.** Add the verified BTC/5m parameters and a behavioral test only if the script reports that all guard conditions pass; otherwise leave the mapping empty and report the evidence.
- [ ] **Step 4: Format changed Python files** with Black at line length 100.
- [ ] **Step 5: Run `pytest tests/ -v` and require zero failures.**
- [ ] **Step 6: Run `git diff --check`, inspect status, and commit intended files without `AGENTS.md`.**
