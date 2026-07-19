# Expanded Momentum Grid Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the default Momentum grid to 4,800 combinations with broader period ranges and finer intervals.

**Architecture:** Only default constant lists, their regression assertions, and canonical documentation change. CLI overrides and the search algorithm remain untouched.

**Tech Stack:** Python 3.10, pytest, Black

## Global Constraints

- ROC defaults must equal `list(range(2, 31, 2))`.
- Momentum defaults must equal `list(range(2, 41, 2))`.
- Threshold defaults must equal `[value / 1000 for value in range(5, 81, 5)]`.
- The Cartesian product must contain exactly 4,800 combinations.
- Do not change ranking, splitting, validation, output schema, or CLI override behavior.

---

### Task 1: Defaults, tests, and documentation

**Files:**
- Modify: `scripts/grid_search_momentum.py`
- Modify: `tests/test_momentum_grid_search.py`
- Modify: `src/strategy_and_backtest_tutorial.md`

- [ ] **Step 1: Update the existing defaults test first**

Assert the exact three expanded lists and assert their length product is 4,800. Keep the explicit override test unchanged.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_momentum_grid_search.py::test_parse_arguments_has_expected_defaults -v`. Expected: current five/five/six defaults fail.

- [ ] **Step 3: Implement the expanded defaults**

Replace the three module constants with the exact range expressions from the global constraints. Do not alter parser or evaluation logic.

- [ ] **Step 4: Update canonical documentation**

Update the example command to the expanded comma-separated values, change 150 to 4,800 combinations, and warn that runtime and selection-overfitting risk increase.

- [ ] **Step 5: Verify**

```bash
black scripts/grid_search_momentum.py tests/test_momentum_grid_search.py --line-length 100 --check
pytest tests/ -q
python scripts/grid_search_momentum.py --help
git diff --check
```

Expected: formatting passes, all tests pass, help works, and no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/grid_search_momentum.py tests/test_momentum_grid_search.py src/strategy_and_backtest_tutorial.md
git commit -m "feat: expand momentum grid defaults"
```
