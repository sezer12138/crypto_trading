# Rank-One Momentum Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update Momentum defaults to rank-one values `16`, `12`, and `0.055` with accurate tests and documentation.

**Architecture:** Shared constants remain the source of truth; the strategy constructor consumes them. Tests verify defaults and overrides, while documentation discloses both training and validation performance.

**Tech Stack:** Python 3.10, pytest, Black

## Global Constraints

- Defaults must be ROC 16, momentum 12, threshold 0.055.
- Explicit overrides must remain unchanged.
- Do not alter signal logic, grid ranges, engine behavior, or risk controls.
- Disclose training return 121.50%, training Sharpe 2.14, training drawdown -16.98%, validation return -18.16%, and validation Sharpe -2.02.

---

### Task 1: Update defaults, regression test, and documentation

**Files:**
- Modify: `src/strategies/constants.py`
- Modify: `src/strategies/momentum.py`
- Modify: `tests/test_extended.py`
- Modify: `src/strategy_and_backtest_tutorial.md`
- Modify: `README.md`

- [ ] Update the existing defaults regression test to expect `16`, `12`, and `0.055` while retaining explicit override assertions.
- [ ] Run `pytest tests/test_extended.py::TestMomentumStrategy -v` and verify the defaults assertion fails.
- [ ] Update the shared constants and strategy docs/example.
- [ ] Update tutorial and README values and performance disclosure.
- [ ] Run `pytest tests/ -q`, `black src/strategies/momentum.py --line-length 100 --check`, and `git diff --check`.
- [ ] Commit with `git commit -m "feat: use rank-one momentum defaults"`.
