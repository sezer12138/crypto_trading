# Asymmetric Momentum Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Momentum entries and exits independent parameters and optimize them with a bounded two-stage search.

**Architecture:** `MomentumStrategy` calculates separate buy and sell indicators from six explicit parameters. The grid-search script reuses its existing three CLI ranges first for buy candidates and then for sell candidates paired with the five best buys.

**Tech Stack:** Python, pandas, pytest, argparse

## Global Constraints

- Keep all code comments, docstrings, and log messages in English.
- Keep Momentum signals event-based.
- Preserve the existing three grid-range CLI options; add no side-specific CLI ranges.
- Rank training candidates primarily by Total Return.
- Do not stage the untracked `AGENTS.md` file.

---

### Task 1: Asymmetric Momentum strategy

**Files:**
- Modify: `tests/test_extended.py`
- Modify: `src/strategies/constants.py`
- Modify: `src/strategies/momentum.py`

**Interfaces:**
- Produces: `MomentumStrategy(buy_roc_period, buy_momentum_period, buy_threshold, sell_roc_period, sell_momentum_period, sell_threshold)`
- Produces: buy/sell-prefixed ROC and momentum indicator columns

- [ ] **Step 1: Write failing tests** for six defaults, six explicit overrides, distinct indicator values, and buy/sell signals driven by their respective parameters.
- [ ] **Step 2: Verify RED** with `pytest tests/test_extended.py::TestMomentumStrategy -v`; failures must identify the missing buy/sell interface.
- [ ] **Step 3: Add six constants and implement the six-parameter strategy**, calculating `buy_roc`, `buy_momentum`, `buy_momentum_norm`, `sell_roc`, `sell_momentum`, and `sell_momentum_norm` before applying side-specific crossing rules.
- [ ] **Step 4: Verify GREEN** with `pytest tests/test_extended.py::TestMomentumStrategy -v`.

### Task 2: Two-stage grid search

**Files:**
- Modify: `tests/test_momentum_grid_search.py`
- Modify: `scripts/grid_search_momentum.py`

**Interfaces:**
- Produces: `evaluate_buy_grid(...) -> pd.DataFrame`
- Produces: `select_top_buy_candidates(results, count=5) -> pd.DataFrame`
- Produces: `evaluate_sell_grid(train_df, buy_candidates, ...) -> pd.DataFrame`
- Consumes: the existing `--roc-periods`, `--momentum-periods`, and `--thresholds` lists for both stages

- [ ] **Step 1: Replace the old symmetric-grid tests with failing tests** asserting buy-prefixed stage-one rows, top-five retention, the full candidate-by-sell Cartesian product, six-column deterministic ranking, and six-parameter validation.
- [ ] **Step 2: Verify RED** with `pytest tests/test_momentum_grid_search.py -v`; failures must identify missing two-stage functions or columns.
- [ ] **Step 3: Implement stage-one and stage-two evaluators**, using fresh engines, explicit six-parameter strategies, and `TOP_BUY_CANDIDATES = 5`.
- [ ] **Step 4: Update ranking, validation, run orchestration, and console output** to operate on six parameter columns and report `buy_*` plus `sell_*` winners.
- [ ] **Step 5: Verify GREEN** with `pytest tests/test_momentum_grid_search.py -v`.

### Task 3: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `src/strategy_and_backtest_tutorial.md`

**Interfaces:**
- Documents: the six strategy parameters, shared CLI candidate ranges, two-stage search, 28,800 default evaluations, and overfitting warning

- [ ] **Step 1: Update documentation** so formulas and examples use explicit `buy_*` and `sell_*` names and describe the bounded two-stage process.
- [ ] **Step 2: Format changed Python files** with `black src/strategies/momentum.py scripts/grid_search_momentum.py tests/test_extended.py tests/test_momentum_grid_search.py --line-length 100`.
- [ ] **Step 3: Run focused verification** with `pytest tests/test_extended.py::TestMomentumStrategy tests/test_momentum_grid_search.py -v`.
- [ ] **Step 4: Run the complete suite** with `pytest tests/ -v` and require zero failures.
- [ ] **Step 5: Inspect `git diff --check` and `git status --short`**, confirming only intended files plus untracked `AGENTS.md` remain.
- [ ] **Step 6: Commit the implementation** without staging `AGENTS.md`.
