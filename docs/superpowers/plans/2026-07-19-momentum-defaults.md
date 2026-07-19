# Momentum Default Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change Momentum strategy defaults to ROC 5, momentum 10, and threshold 0.04 using shared constants and accurate documentation.

**Architecture:** Strategy constants remain the single source of truth. `MomentumStrategy` imports them as constructor defaults; factory tests and documentation verify the public behavior and disclose the negative validation result.

**Tech Stack:** Python 3.10, pytest, Black

## Global Constraints

- Defaults must be exactly `5`, `10`, and `0.04`.
- Explicit constructor/factory arguments must continue to override defaults.
- Do not change signal calculations, grid ranges, engine behavior, or risk controls.
- Document training return `37.43%` and validation return `-17.67%`.
- New non-README prose must be English.

---

### Task 1: Constants, strategy defaults, tests, and documentation

**Files:**
- Modify: `src/strategies/constants.py`
- Modify: `src/strategies/momentum.py`
- Modify: `tests/test_extended.py`
- Modify: `src/strategy_and_backtest_tutorial.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `DEFAULT_MOMENTUM_ROC_PERIOD = 5`, `DEFAULT_MOMENTUM_PERIOD = 10`, `DEFAULT_MOMENTUM_THRESHOLD = 0.04`, and matching `MomentumStrategy()` defaults.

- [ ] **Step 1: Write failing regression tests**

Add a test asserting `get_strategy("momentum")` exposes `roc_period == 5`, `momentum_period == 10`, and `threshold == 0.04`. Add a second assertion that explicit values `9`, `12`, and `0.03` are preserved.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_extended.py::TestMomentumStrategy -v`. Expected: the default test fails with current values `10`, `14`, and `0.02`.

- [ ] **Step 3: Implement constants and defaults**

Add the three constants to the Momentum section of `constants.py`, import them in `momentum.py`, and replace constructor literals. Update its docstring and usage example.

- [ ] **Step 4: Update documentation**

Update the tutorial parameter table and README summary to `5`, `10`, and `0.04`. Add the training/validation disclosure without presenting the change as out-of-sample improvement.

- [ ] **Step 5: Verify GREEN and repository health**

```bash
black src/strategies/momentum.py tests/test_extended.py --line-length 100 --check
pytest tests/ -q
git diff --check
```

Expected: formatting passes, all tests pass, and no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add src/strategies/constants.py src/strategies/momentum.py tests/test_extended.py src/strategy_and_backtest_tutorial.md README.md
git commit -m "feat: update momentum strategy defaults"
```
