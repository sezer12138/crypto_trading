# Drawdown Circuit Breaker Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python API and CLI controls that can disable maximum-drawdown forced liquidation and trading shutdown without disabling other risk controls or drawdown metrics.

**Architecture:** `BacktestEngine` owns a persistent Boolean configuration flag and guards only the circuit-breaker branch with it. `run_backtest.py` converts the negative CLI switch into that positive engine flag and threads it through single, all-coin, and comparison execution paths.

**Tech Stack:** Python 3.10, pandas, argparse, pytest

## Global Constraints

- Keep drawdown circuit breaking enabled by default.
- Keep `max_drawdown_pct` as a numeric threshold independent of enablement.
- Continue calculating the maximum-drawdown performance metric when the breaker is disabled.
- Do not change stop-loss or consecutive-loss cooldown behavior.
- All comments, docstrings, logs, and new documentation must be English.

---

### Task 1: Backtest engine enablement control

**Files:**
- Modify: `tests/test_risk_management.py:214-263,374-393`
- Modify: `src/backtest.py:280-480`

**Interfaces:**
- Consumes: Existing `BacktestEngine(max_drawdown_pct: float, breaker_cooldown_bars: int)` configuration.
- Produces: `BacktestEngine(drawdown_breaker_enabled: bool = True)` and public `engine.drawdown_breaker_enabled` configuration.

- [ ] **Step 1: Write failing engine tests**

Add these tests to `TestDrawdownCircuitBreaker` and `TestDefaultParameters`:

```python
def test_disabled_circuit_breaker_does_not_force_liquidation(self):
    prices = [100.0, 120.0, 90.0, 95.0, 100.0]
    signals = [SIGNAL_BUY, 0, 0, 0, SIGNAL_SELL]
    df = _make_df(prices, signals=signals)
    engine = BacktestEngine(
        initial_capital=10000,
        drawdown_breaker_enabled=False,
        max_drawdown_pct=0.20,
        stop_loss_pct=1.0,
        min_holding_bars=0,
    )

    result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

    assert result.trades[1].timestamp == df.index[4]
    assert result.trades[1].strategy_signal == SIGNAL_SELL
    assert result.metrics["max_drawdown_pct"] < -20.0

def test_disabled_breaker_keeps_stop_loss_active(self):
    df = _make_df(
        [100.0, 90.0, 90.0],
        signals=[SIGNAL_BUY, 0, 0],
    )
    engine = BacktestEngine(
        drawdown_breaker_enabled=False,
        stop_loss_pct=0.05,
        min_holding_bars=0,
    )

    result = engine.run_backtest(df, IdentityStrategy(), coin="TEST")

    assert result.trades[1].strategy_signal == -2

def test_drawdown_breaker_enabled_by_default(self):
    engine = BacktestEngine()
    assert engine.drawdown_breaker_enabled is True
```

- [ ] **Step 2: Run tests and verify the feature is missing**

Run:

```bash
pytest tests/test_risk_management.py::TestDrawdownCircuitBreaker::test_disabled_circuit_breaker_does_not_force_liquidation tests/test_risk_management.py::TestDrawdownCircuitBreaker::test_disabled_breaker_keeps_stop_loss_active tests/test_risk_management.py::TestDefaultParameters::test_drawdown_breaker_enabled_by_default -v
```

Expected: FAIL with `TypeError: BacktestEngine.__init__() got an unexpected keyword argument 'drawdown_breaker_enabled'` and/or missing attribute.

- [ ] **Step 3: Implement the engine flag**

Add the constructor parameter immediately before `breaker_cooldown_bars`:

```python
drawdown_breaker_enabled: bool = True,
```

Store and document it:

```python
self.drawdown_breaker_enabled = drawdown_breaker_enabled
```

Change startup logging to:

```python
logger.info(
    f"   Drawdown breaker: {'enabled' if drawdown_breaker_enabled else 'disabled'}"
)
if drawdown_breaker_enabled:
    logger.info(f"   Max drawdown: {max_drawdown_pct * 100:.1f}%")
```

Guard the breaker trigger without changing its internal behavior:

```python
if self.drawdown_breaker_enabled and drawdown >= self.max_drawdown_pct:
```

Update the class docstring arguments and attributes to describe the new flag.

- [ ] **Step 4: Run focused and complete risk tests**

Run:

```bash
pytest tests/test_risk_management.py -v
```

Expected: all risk-management tests PASS, including existing enabled behavior.

- [ ] **Step 5: Commit the engine behavior**

```bash
git add src/backtest.py tests/test_risk_management.py
git commit -m "feat: make drawdown breaker optional"
```

---

### Task 2: CLI and runner propagation

**Files:**
- Modify: `run_backtest.py:63-492`
- Create: `tests/test_run_backtest.py`

**Interfaces:**
- Consumes: `BacktestEngine(drawdown_breaker_enabled: bool = True)` from Task 1.
- Produces: `--disable-drawdown-breaker`, `run_single_backtest(..., drawdown_breaker_enabled: bool = True)`, and `compare_strategies(..., drawdown_breaker_enabled: bool = True)`.

- [ ] **Step 1: Write failing CLI parsing tests**

Create `tests/test_run_backtest.py`:

```python
"""Tests for command-line backtest configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_backtest


def test_drawdown_breaker_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    args = run_backtest.parse_arguments()
    assert args.disable_drawdown_breaker is False


def test_disable_drawdown_breaker_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_backtest.py", "--disable-drawdown-breaker"],
    )
    args = run_backtest.parse_arguments()
    assert args.disable_drawdown_breaker is True


def test_main_propagates_disabled_breaker_to_comparison(monkeypatch):
    captured = {}

    def fake_compare(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--compare", "--disable-drawdown-breaker"])
    monkeypatch.setattr(run_backtest, "compare_strategies", fake_compare)

    run_backtest.main()

    assert captured["drawdown_breaker_enabled"] is False


def test_main_propagates_disabled_breaker_to_single_run(monkeypatch):
    captured = {}

    def fake_run_single(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--disable-drawdown-breaker"])
    monkeypatch.setattr(run_backtest, "run_single_backtest", fake_run_single)

    run_backtest.main()

    assert captured["drawdown_breaker_enabled"] is False
```

- [ ] **Step 2: Run parsing tests and verify failure**

Run:

```bash
pytest tests/test_run_backtest.py -v
```

Expected: FAIL because the parsed namespace lacks `disable_drawdown_breaker`, argparse rejects the new option, and runner functions do not accept the propagated keyword.

- [ ] **Step 3: Add and propagate the CLI option**

Add to `parse_arguments()`:

```python
parser.add_argument(
    "--disable-drawdown-breaker",
    action="store_true",
    help="Disable forced liquidation and trading halt at the maximum drawdown threshold",
)
```

Add `drawdown_breaker_enabled: bool = True` to `run_single_backtest()` and construct the engine with:

```python
engine = BacktestEngine(
    initial_capital=capital,
    drawdown_breaker_enabled=drawdown_breaker_enabled,
)
```

Add the same defaulted parameter to `compare_strategies()` and forward it into every `run_single_backtest()` call. In `main()`, define:

```python
drawdown_breaker_enabled = not args.disable_drawdown_breaker
```

Pass that value to comparison mode, all-coin mode, and single-backtest mode. Update both function docstrings.

- [ ] **Step 4: Run CLI help and tests**

Run:

```bash
python run_backtest.py --help
pytest tests/test_run_backtest.py -v
```

Expected: help lists `--disable-drawdown-breaker`; all four CLI and propagation tests PASS.

- [ ] **Step 5: Commit the CLI behavior**

```bash
git add run_backtest.py tests/test_run_backtest.py
git commit -m "feat: expose drawdown breaker CLI flag"
```

---

### Task 3: User documentation and full verification

**Files:**
- Modify: `src/strategy_and_backtest_tutorial.md:800-935`
- Modify: `README.md:190-225`

**Interfaces:**
- Consumes: Python and CLI interfaces introduced in Tasks 1 and 2.
- Produces: User-facing examples for enabling and disabling the breaker.

- [ ] **Step 1: Update the canonical tutorial**

Document the default in the risk-control table and add these examples:

```python
engine = BacktestEngine(
    initial_capital=10000,
    drawdown_breaker_enabled=False,
)
```

```bash
python run_backtest.py --strategy momentum --disable-drawdown-breaker
python run_backtest.py --compare --disable-drawdown-breaker
```

State explicitly that disabling this breaker does not disable stop-losses and does not remove the maximum-drawdown metric.

- [ ] **Step 2: Update the README API example**

Add `drawdown_breaker_enabled=True` to the `BacktestEngine` example and a short CLI example using `--disable-drawdown-breaker`.

- [ ] **Step 3: Run formatting and full verification**

Run:

```bash
black src/ tests/ --line-length 100 --check
pytest tests/ -v
python run_backtest.py --help
git diff --check
```

Expected: Black check succeeds; all tests PASS; CLI help displays the new flag; Git reports no whitespace errors.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md src/strategy_and_backtest_tutorial.md
git commit -m "docs: explain optional drawdown breaker"
```
