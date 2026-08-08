# BTC Momentum Runtime Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate explicit BTC/5m and BTC/1h Momentum profiles without changing other coin or interval behavior.

**Architecture:** Populate the existing normalized runtime profile mapping with two six-parameter entries. Verify profile selection through the real `create_strategy()` boundary and document the BTC/5m stability-guard override.

**Tech Stack:** Python 3.10, pytest

## Global Constraints

- BTC/5m is an experimental user-requested override, not a validated optimum.
- BTC/1h uses the previously selected `16, 12, 0.055` values on both sides.
- ETH, SOL, and unlisted intervals must retain general defaults.
- Do not stage `AGENTS.md`.

---

### Task 1: Activate and verify BTC profiles

**Files:**
- Modify: `tests/test_run_backtest.py`
- Modify: `src/strategies/momentum_profiles.py`

**Interfaces:**
- Consumes: `get_momentum_profile(coin, interval)`
- Produces: exact BTC/5m and BTC/1h parameter dictionaries

- [ ] **Step 1: Write failing behavioral tests** asserting exact BTC/5m and BTC/1h parameters through `create_strategy()`, plus ETH/5m fallback to general defaults.
- [ ] **Step 2: Run `pytest tests/test_run_backtest.py -v`** and verify failures identify the absent runtime mappings.
- [ ] **Step 3: Add both profile entries** with an explicit experimental warning on BTC/5m.
- [ ] **Step 4: Run `pytest tests/test_run_backtest.py -v`** and require all tests to pass.

### Task 2: Document, verify, and commit

**Files:**
- Modify: `README.md`
- Modify: `src/strategy_and_backtest_tutorial.md`

**Interfaces:**
- Documents: exact active parameters and experimental status

- [ ] **Step 1: Update documentation** to state that BTC/5m was activated by explicit user override and BTC/1h is explicitly configured.
- [ ] **Step 2: Format changed Python files** with Black at line length 100.
- [ ] **Step 3: Run `pytest tests/ -v`** and require zero failures.
- [ ] **Step 4: Run `git diff --check`, inspect status, and commit intended files without `AGENTS.md`.**
