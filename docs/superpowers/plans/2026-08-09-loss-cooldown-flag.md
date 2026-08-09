# Loss Cooldown CLI Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working `--disable-loss-cooldown` flag that independently disables consecutive-loss tracking, warnings, and buy cooldowns in every backtest execution mode.

**Architecture:** `BacktestEngine` owns a positive `loss_cooldown_enabled` setting and guards only the consecutive-loss mechanism. `run_backtest.py` converts the negative CLI flag into that positive setting and passes it through single, all-coin, and comparison paths.

**Tech Stack:** Python 3.10, pandas, argparse, pytest

## Global Constraints

- Consecutive-loss cooldown remains enabled by default.
- Disabling loss cooldown must not disable per-position stop-losses or the drawdown breaker.
- Append the engine parameter after existing parameters to preserve positional callers.
- The new flag must work with single runs, `--compare`, and `--coin all`.
- All code comments, docstrings, and log messages must remain in English.
- Do not stage the untracked `AGENTS.md` file.

---

### Task 1: Backtest engine loss-cooldown switch

**Files:**
- Modify: `tests/test_risk_management.py`
- Modify: `src/backtest.py`

**Interfaces:**
- Consumes: Existing `max_consecutive_losses` and `consecutive_loss_cooldown` settings.
- Produces: `BacktestEngine(..., loss_cooldown_enabled: bool = True)` and `engine.loss_cooldown_enabled`.

- [ ] **Step 1: Write failing engine tests**

Add a `TestConsecutiveLossCooldown` class using four losing round trips:

```python
class TestConsecutiveLossCooldown:
    @staticmethod
    def _losing_round_trips():
        prices = [100.0, 90.0] * 4
        signals = [SIGNAL_BUY, SIGNAL_SELL] * 4
        return _make_df(prices, signals=signals)

    def test_enabled_cooldown_blocks_buy_and_logs_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="backtest")
        engine = BacktestEngine(
            min_holding_bars=0,
            max_trades_per_day=99,
            stop_loss_pct=1.0,
            max_consecutive_losses=3,
            consecutive_loss_cooldown=24,
        )

        result = engine.run_backtest(self._losing_round_trips(), IdentityStrategy(), coin="TEST")

        assert len(result.trades) == 6
        assert "Consecutive loss limit (3) reached" in caplog.text

    def test_disabled_cooldown_allows_buy_and_emits_no_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="backtest")
        engine = BacktestEngine(
            min_holding_bars=0,
            max_trades_per_day=99,
            stop_loss_pct=1.0,
            max_consecutive_losses=3,
            consecutive_loss_cooldown=24,
            loss_cooldown_enabled=False,
        )

        result = engine.run_backtest(self._losing_round_trips(), IdentityStrategy(), coin="TEST")

        assert len(result.trades) == 8
        assert "Consecutive loss limit" not in caplog.text
        assert engine._consecutive_losses == 0
        assert engine._loss_cooldown_until == -1
```

Add the default-value assertion:

```python
def test_loss_cooldown_enabled_by_default(self):
    assert BacktestEngine().loss_cooldown_enabled is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_risk_management.py::TestConsecutiveLossCooldown tests/test_risk_management.py::TestDefaultParameters::test_loss_cooldown_enabled_by_default -v
```

Expected: the disabled test fails because `loss_cooldown_enabled` is not accepted, and the default test fails because the attribute is missing.

- [ ] **Step 3: Implement the minimal engine behavior**

Append the constructor parameter after `drawdown_breaker_enabled`:

```python
loss_cooldown_enabled: bool = True,
```

Store and log it:

```python
self.loss_cooldown_enabled = loss_cooldown_enabled
logger.info(
    f"   Consecutive-loss cooldown: {'enabled' if loss_cooldown_enabled else 'disabled'}"
)
```

Guard the stop-loss accounting block while retaining its `pre_sell_entry` comparison:

```python
if self.loss_cooldown_enabled:
    if price < pre_sell_entry:
        self._consecutive_losses += 1
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_cooldown_until = i + self.consecutive_loss_cooldown
            logger.warning(
                f"Consecutive loss limit ({self.max_consecutive_losses}) "
                f"reached, cooldown until bar {self._loss_cooldown_until}"
            )
            self._consecutive_losses = 0
    else:
        self._consecutive_losses = 0
```

Apply the same enablement guard to the regular-sell accounting block while retaining its existing
`price >= entry_price` profitable-exit comparison.

Require enablement when deciding whether a bar is cooling down:

```python
in_loss_cooldown = (
    self.loss_cooldown_enabled
    and self._loss_cooldown_until >= 0
    and i < self._loss_cooldown_until
)
```

- [ ] **Step 4: Verify engine behavior**

Run:

```bash
pytest tests/test_risk_management.py -v
```

Expected: every risk-management test passes, including enabled and disabled cooldown behavior.

- [ ] **Step 5: Commit the engine change**

```bash
git add src/backtest.py tests/test_risk_management.py
git commit -m "feat: make loss cooldown optional"
```

---

### Task 2: CLI parsing and execution-path propagation

**Files:**
- Modify: `tests/test_run_backtest.py`
- Modify: `run_backtest.py`

**Interfaces:**
- Consumes: `BacktestEngine(..., loss_cooldown_enabled: bool = True)` from Task 1.
- Produces: `--disable-loss-cooldown`, `run_single_backtest(..., loss_cooldown_enabled: bool = True)`, and `compare_strategies(..., loss_cooldown_enabled: bool = True)`.

- [ ] **Step 1: Write failing CLI and propagation tests**

Add parsing coverage:

```python
def test_loss_cooldown_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    assert run_backtest.parse_arguments().disable_loss_cooldown is False

def test_disable_loss_cooldown_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--disable-loss-cooldown"])
    assert run_backtest.parse_arguments().disable_loss_cooldown is True
```

Add tests following the existing drawdown-propagation pattern and assert
`captured["loss_cooldown_enabled"] is False` for:

```python
run_backtest.main()  # --compare --disable-loss-cooldown
run_backtest.main()  # single run with --disable-loss-cooldown
run_backtest.main()  # --coin all --no-viz --disable-loss-cooldown
run_backtest.compare_strategies(
    "btc", 30, "1h", 10000.0,
    save_report=False,
    loss_cooldown_enabled=False,
)
```

Add a focused `run_single_backtest` test that replaces `BacktestEngine` with a capture fake,
returns a minimal result with `metrics = {}` and `save_logs()`, passes
`loss_cooldown_enabled=False`, and asserts the fake constructor receives false.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
pytest tests/test_run_backtest.py -v
```

Expected: argparse rejects `--disable-loss-cooldown`, and propagation tests fail because the new
keyword argument and namespace attribute do not exist.

- [ ] **Step 3: Implement parsing and propagation**

Add the parser option:

```python
parser.add_argument(
    "--disable-loss-cooldown",
    action="store_true",
    help="Disable pausing new entries after consecutive losing trades",
)
```

Add `loss_cooldown_enabled: bool = True` to `run_single_backtest` and `compare_strategies`. Pass it
to `BacktestEngine` and from comparison into every single run. In `main`, derive and propagate it:

```python
loss_cooldown_enabled = not args.disable_loss_cooldown
```

- [ ] **Step 4: Verify CLI behavior**

Run:

```bash
pytest tests/test_run_backtest.py -v
python run_backtest.py --help | rg -- "--disable-loss-cooldown"
```

Expected: every CLI test passes and help output lists the new flag.

- [ ] **Step 5: Commit the runner change**

```bash
git add run_backtest.py tests/test_run_backtest.py
git commit -m "feat: expose loss cooldown CLI flag"
```

---

### Task 3: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `src/strategy_and_backtest_tutorial.md`

**Interfaces:**
- Consumes: The working `--disable-loss-cooldown` CLI option from Task 2.
- Produces: User-facing explanation and combined command example.

- [ ] **Step 1: Update documentation**

Document that `--disable-loss-cooldown` disables only consecutive-loss tracking and entry pauses,
while stop-losses remain enabled. Include this exact combined example:

```bash
python run_backtest.py --interval 5m --days 720 \
  --disable-drawdown-breaker --disable-loss-cooldown --compare
```

State that the general sub-hourly-interval warning remains visible because it is unrelated to loss
cooldown.

- [ ] **Step 2: Format and inspect changes**

Run:

```bash
black src/backtest.py run_backtest.py tests/test_risk_management.py tests/test_run_backtest.py --line-length 100
git diff --check
```

Expected: Black exits successfully and `git diff --check` produces no output.

- [ ] **Step 3: Run complete verification**

Run:

```bash
pytest tests/ -v
python run_backtest.py --help | rg -- "--disable-loss-cooldown"
```

Expected: all tests pass and help output includes `--disable-loss-cooldown`.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md src/strategy_and_backtest_tutorial.md
git commit -m "docs: explain optional loss cooldown"
```
