# Bayesian Momentum Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Optuna TPE optimizer for the six asymmetric Momentum parameters using chronological folds, a return/drawdown/stability loss, resumable SQLite studies, process-parallel backtests, isolated holdout validation, and auditable reports.

**Architecture:** Focused modules under `src/optimization/` own data validation, pure scoring, trial evaluation, study persistence, coordination, and reporting. A thin CLI composes them. The parent process exclusively owns Optuna and SQLite through ask/tell; workers load immutable data once and only run backtests.

**Tech Stack:** Python 3.10+, pandas, NumPy, Optuna 4.x TPE, SQLite through Optuna, `ProcessPoolExecutor`, pytest, standalone HTML/SVG.

## Global Constraints

- Add exactly `optuna>=4.0,<5.0`; add no plotting dependency.
- Keep source comments, docstrings, exceptions, and logs in English.
- Preserve event-based Momentum signals, costs, slippage, stop-loss, holding, and trade limits.
- Keep the latest 20% holdout invisible to TPE and evaluate it only after selection.
- Defaults: 300 target trials, 30 startup trials, four folds, seed 42, one worker.
- Loss defaults: drawdown target 25, drawdown weight 2.0, stability weight 0.5, two minimum round trips per fold, and 25.0 per missing round trip.
- Never edit `src/strategies/momentum_profiles.py` automatically.
- `--trials` is the target total terminal-trial count after resume, not additional trials.
- Keep studies and generated reports under ignored `results/`.
- Use Black line length 100 and TDD red-green-refactor for every change.

## File Map

- `src/optimization/data.py`: OHLCV validation, cadence, folds/holdout, SHA-256.
- `src/optimization/momentum_objective.py`: configs, bounds, suggestions, loss, Pareto logic.
- `src/optimization/momentum_evaluator.py`: backtests, fold evaluation, process worker state.
- `src/optimization/optuna_study.py`: persistence, metadata, recovery, ask/tell runners.
- `src/optimization/momentum_report.py`: holdout comparison and five result artifacts.
- `scripts/bayesian_search_momentum.py`: CLI and orchestration.
- Tests mirror each module in `tests/test_*.py`.
- `scripts/grid_search_momentum.py` imports shared data helpers without API changes.
- `requirements.txt`, `README.md`, and `src/strategy_and_backtest_tutorial.md` are updated.

---

### Task 1: Shared Optimization Data Utilities

**Files:**
- Create: `src/optimization/__init__.py`
- Create: `src/optimization/data.py`
- Create: `tests/test_optimization_data.py`
- Modify: `scripts/grid_search_momentum.py:24-158`

**Interfaces:**
- Produces `load_ohlcv(path: Path) -> pd.DataFrame`.
- Produces `infer_interval_minutes(index: pd.DatetimeIndex) -> int`.
- Produces `split_holdout_and_folds(df, holdout_ratio, fold_count, max_lookback) -> tuple[list[pd.DataFrame], pd.DataFrame]`.
- Produces `fingerprint_file(path: Path) -> str`.
- Preserves imports from `scripts.grid_search_momentum` for existing callers.

- [ ] **Step 1: Write failing split and fingerprint tests**

```python
def test_split_holdout_and_folds_is_chronological_and_non_overlapping():
    data = _ohlcv_frame(100)
    folds, holdout = split_holdout_and_folds(data, 0.20, 4, max_lookback=5)
    assert [len(fold) for fold in folds] == [20, 20, 20, 20]
    assert len(holdout) == 20
    assert all(folds[i].index.max() < folds[i + 1].index.min() for i in range(3))
    assert folds[-1].index.max() < holdout.index.min()


def test_fingerprint_file_changes_when_content_changes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("first", encoding="utf-8")
    first = fingerprint_file(path)
    path.write_text("second", encoding="utf-8")
    assert fingerprint_file(path) != first
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_optimization_data.py -v`

Expected: import failure because `optimization.data` does not exist.

- [ ] **Step 3: Implement splitting and streaming SHA-256**

```python
def split_holdout_and_folds(df, holdout_ratio, fold_count, max_lookback):
    if not 0 < holdout_ratio < 1:
        raise ValueError("Holdout ratio must be between zero and one")
    if fold_count < 2:
        raise ValueError("Fold count must be at least two")
    split = int(len(df) * (1 - holdout_ratio))
    pool, holdout = df.iloc[:split], df.iloc[split:]
    base, remainder = divmod(len(pool), fold_count)
    sizes = [base + (index < remainder) for index in range(fold_count)]
    if min(sizes) <= max_lookback or len(holdout) <= max_lookback:
        raise ValueError("Every optimization partition must exceed the maximum lookback")
    folds, start = [], 0
    for size in sizes:
        folds.append(pool.iloc[start : start + size].copy())
        start += size
    return folds, holdout.copy()


def fingerprint_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

Move the current `load_ohlcv` and `infer_interval_minutes` bodies unchanged into the shared module,
then import them into the grid script so existing public imports still resolve.

- [ ] **Step 4: Verify GREEN and compatibility**

Run: `pytest tests/test_optimization_data.py tests/test_momentum_grid_search.py -v`

Expected: all new and existing data/grid tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/__init__.py src/optimization/data.py \
  scripts/grid_search_momentum.py tests/test_optimization_data.py
git commit -m "refactor: share optimization data utilities"
```

---

### Task 2: Search Space, Composite Loss, and Pareto Logic

**Files:**
- Create: `src/optimization/momentum_objective.py`
- Create: `tests/test_momentum_objective.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces frozen `LossConfig`, `SearchBounds`, `FoldMetrics`, `TrialEvaluation`.
- Produces `resolve_search_bounds(interval_minutes: int) -> SearchBounds`.
- Produces `suggest_momentum_parameters(trial, bounds) -> dict[str, int | float]`.
- Produces `score_folds(folds, config) -> TrialEvaluation`.
- Produces `pareto_front(rows) -> list[Mapping[str, Any]]`.

- [ ] **Step 1: Add and install Optuna**

Append `optuna>=4.0,<5.0` to `requirements.txt` and run `pip install -r requirements.txt`.

Expected: `python -c 'import optuna; print(optuna.__version__)'` prints 4.x.

- [ ] **Step 2: Write failing bounds/loss/Pareto tests**

```python
def test_5m_bounds_extend_the_previous_boundary():
    bounds = resolve_search_bounds(5)
    assert (bounds.roc_min, bounds.roc_max) == (12, 8064)
    assert (bounds.momentum_min, bounds.momentum_max) == (6, 4032)
    assert (bounds.threshold_min, bounds.threshold_max) == (0.005, 0.25)


def test_score_folds_combines_all_penalties():
    folds = [
        FoldMetrics(20.0, -20.0, 4), FoldMetrics(30.0, -30.0, 2),
        FoldMetrics(10.0, -15.0, 0), FoldMetrics(40.0, -10.0, 6),
    ]
    result = score_folds(folds, LossConfig())
    assert result.robust_annual_return_pct == 25.0
    assert result.worst_drawdown_pct == 30.0
    assert result.missing_round_trips == 3
    assert result.loss == pytest.approx(65.5901699)


def test_pareto_front_excludes_dominated_rows():
    rows = [
        {"trial": 1, "robust_return": 20.0, "worst_drawdown": 20.0},
        {"trial": 2, "robust_return": 25.0, "worst_drawdown": 20.0},
        {"trial": 3, "robust_return": 18.0, "worst_drawdown": 10.0},
    ]
    assert [row["trial"] for row in pareto_front(rows)] == [3, 2]
```

- [ ] **Step 3: Verify RED**

Run: `pytest tests/test_momentum_objective.py -v`

Expected: missing objective module.

- [ ] **Step 4: Implement exact pure loss**

```python
@dataclass(frozen=True)
class LossConfig:
    drawdown_target_pct: float = 25.0
    drawdown_weight: float = 2.0
    stability_weight: float = 0.5
    minimum_round_trips: int = 2
    missing_round_trip_penalty: float = 25.0


def score_folds(folds, config):
    annual = [item.annual_return_pct for item in folds]
    if not folds or not all(math.isfinite(value) for value in annual):
        raise ValueError("Fold metrics must be finite")
    robust = statistics.median(annual)
    worst = max(abs(item.max_drawdown_pct) for item in folds)
    instability = statistics.pstdev(annual)
    missing = sum(max(0, config.minimum_round_trips - item.total_trades // 2)
                  for item in folds)
    loss = (-robust
            + config.drawdown_weight * max(0.0, worst - config.drawdown_target_pct)
            + config.stability_weight * instability
            + config.missing_round_trip_penalty * missing)
    return TrialEvaluation(loss, robust, worst, instability, missing, tuple(folds))
```

- [ ] **Step 5: Implement independent logarithmic suggestions**

```python
def suggest_momentum_parameters(trial, bounds):
    return {
        "buy_roc_period": trial.suggest_int("buy_roc_period", bounds.roc_min,
                                            bounds.roc_max, log=True),
        "buy_momentum_period": trial.suggest_int("buy_momentum_period",
                                                 bounds.momentum_min,
                                                 bounds.momentum_max, log=True),
        "buy_threshold": trial.suggest_float("buy_threshold", bounds.threshold_min,
                                             bounds.threshold_max, log=True),
        "sell_roc_period": trial.suggest_int("sell_roc_period", bounds.roc_min,
                                             bounds.roc_max, log=True),
        "sell_momentum_period": trial.suggest_int("sell_momentum_period",
                                                  bounds.momentum_min,
                                                  bounds.momentum_max, log=True),
        "sell_threshold": trial.suggest_float("sell_threshold", bounds.threshold_min,
                                              bounds.threshold_max, log=True),
    }
```

Implement Pareto ordering by ascending drawdown, descending return, then trial number.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/test_momentum_objective.py -v`

```bash
git add requirements.txt src/optimization/momentum_objective.py tests/test_momentum_objective.py
git commit -m "feat: define Bayesian momentum objective"
```

---

### Task 3: Fold and Holdout Backtest Evaluation

**Files:**
- Create: `src/optimization/momentum_evaluator.py`
- Create: `tests/test_momentum_evaluator.py`

**Interfaces:**
- Produces frozen `BacktestRunConfig`.
- Produces `evaluate_parameters(folds, parameters, loss_config, run_config) -> TrialEvaluation`.
- Produces `run_holdout(df, parameters, run_config) -> dict[str, float]`.
- Produces `initialize_worker(data_path: str, holdout_ratio: float, fold_count: int, max_lookback: int, loss_config: LossConfig, run_config: BacktestRunConfig) -> None`.
- Produces `evaluate_worker(parameters: Mapping[str, int | float]) -> TrialEvaluation`.

- [ ] **Step 1: Write failing risk-propagation test**

```python
def test_evaluate_parameters_propagates_disabled_controls(monkeypatch):
    created = []
    class FakeEngine:
        def __init__(self, **kwargs): created.append(kwargs)
        def run_backtest(self, frame, strategy, coin):
            return SimpleNamespace(metrics={"annual_return_pct": 12.0,
                "max_drawdown_pct": -8.0, "total_trades": 6,
                "total_return_pct": 10.0, "sharpe_ratio": 0.5,
                "win_rate_pct": 50.0})
    monkeypatch.setattr(evaluator, "BacktestEngine", FakeEngine)
    result = evaluate_parameters([_frame(), _frame()], _parameters(), LossConfig(),
        BacktestRunConfig(10000.0, "BTC", False, False))
    assert len(created) == 2
    assert all(item["drawdown_breaker_enabled"] is False for item in created)
    assert all(item["loss_cooldown_enabled"] is False for item in created)
    assert result.robust_annual_return_pct == 12.0
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_momentum_evaluator.py -v`

- [ ] **Step 3: Implement a fresh strategy and engine per partition**

```python
def run_holdout(df, parameters, config):
    engine = BacktestEngine(initial_capital=config.initial_capital,
        drawdown_breaker_enabled=config.drawdown_breaker_enabled,
        loss_cooldown_enabled=config.loss_cooldown_enabled)
    result = engine.run_backtest(df, MomentumStrategy(**parameters), coin=config.coin)
    required = ("total_return_pct", "annual_return_pct", "sharpe_ratio",
                "max_drawdown_pct", "win_rate_pct", "total_trades")
    if any(name not in result.metrics for name in required):
        raise ValueError("Backtest result is missing required optimization metrics")
    return {name: result.metrics[name] for name in required}
```

Convert each fold result to `FoldMetrics`, call `score_folds`, and suppress expected backtest logs
inside a context manager that restores the previous logger level.

- [ ] **Step 4: Implement one-time worker dataset initialization**

```python
def initialize_worker(data_path, holdout_ratio, fold_count, max_lookback,
                      loss_config, run_config):
    data = load_ohlcv(Path(data_path))
    folds, _ = split_holdout_and_folds(data, holdout_ratio, fold_count, max_lookback)
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = WorkerContext(folds, loss_config, run_config)


def evaluate_worker(parameters):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Momentum optimization worker is not initialized")
    return evaluate_parameters(_WORKER_CONTEXT.folds, parameters,
                               _WORKER_CONTEXT.loss_config,
                               _WORKER_CONTEXT.run_config)
```

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_momentum_evaluator.py tests/test_momentum_objective.py -v`

```bash
git add src/optimization/momentum_evaluator.py tests/test_momentum_evaluator.py
git commit -m "feat: evaluate Bayesian momentum trials"
```

---

### Task 4: Persistent Study and Metadata Safety

**Files:**
- Create: `src/optimization/optuna_study.py`
- Create: `tests/test_optuna_study.py`

**Interfaces:**
- Produces `build_metadata(data_path: Path, data: pd.DataFrame, interval_minutes: int, coin: str, capital: float, bounds: SearchBounds, loss_config: LossConfig, holdout_ratio: float, fold_count: int, run_config: BacktestRunConfig) -> dict[str, JSONSerializable]`.
- Produces `create_or_load_study(storage_path, study_name, metadata, seed, startup_trials, parallel) -> Study`.
- Produces `recover_stale_trials(study) -> int`.
- Produces `record_evaluation(trial, evaluation) -> None`.

- [ ] **Step 1: Write failing resume/mismatch/recovery tests**

```python
def test_matching_metadata_resumes_and_mismatch_fails(tmp_path):
    path = tmp_path / "study.db"
    first = create_or_load_study(path, "momentum", {"schema": 1}, 42, 5, False)
    trial = first.ask(); trial.suggest_int("x", 1, 2); first.tell(trial, 1.0)
    assert len(create_or_load_study(path, "momentum", {"schema": 1},
                                    42, 5, False).trials) == 1
    with pytest.raises(ValueError, match="metadata does not match"):
        create_or_load_study(path, "momentum", {"schema": 2}, 42, 5, False)
```

Add a test leaving one ask-created trial RUNNING and assert `recover_stale_trials()` returns one and
changes it to `FAIL`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_optuna_study.py -v`

- [ ] **Step 3: Implement SQLite study creation and exact metadata comparison**

```python
sampler = optuna.samplers.TPESampler(n_startup_trials=startup_trials, seed=seed,
    multivariate=True, constant_liar=parallel)
study = optuna.create_study(study_name=study_name,
    storage=f"sqlite:///{storage_path.resolve()}", sampler=sampler,
    direction="minimize", load_if_exists=True)
existing = study.user_attrs.get("optimizer_metadata")
if existing is None:
    study.set_user_attr("optimizer_metadata", dict(metadata))
elif existing != dict(metadata):
    raise ValueError("Existing study metadata does not match this optimization run")
```

Metadata includes schema, normalized path/SHA, rows/timestamps, interval/coin/capital, bounds, split,
loss configuration, and risk flags.

- [ ] **Step 4: Implement stale recovery and evaluation attributes**

```python
def recover_stale_trials(study):
    running = study.get_trials(deepcopy=False,
        states=(optuna.trial.TrialState.RUNNING,))
    for trial in running:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(running)
```

Store robust return, worst drawdown, annual-return standard deviation, missing round trips, and all
fold metrics as JSON-serializable trial user attributes.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_optuna_study.py -v`

```bash
git add src/optimization/optuna_study.py tests/test_optuna_study.py
git commit -m "feat: persist Bayesian momentum studies"
```

---

### Task 5: Exact Sequential and Process Ask/Tell Runners

**Files:**
- Modify: `src/optimization/optuna_study.py`
- Modify: `tests/test_optuna_study.py`

**Interfaces:**
- Produces frozen `RunSummary(completed, failed, interrupted)`.
- Produces `terminal_trial_count(study) -> int`.
- Produces `run_sequential_trials(study, target_trials, bounds, evaluator) -> RunSummary`.
- Produces `run_process_trials(study, target_trials, workers, bounds, initializer, initargs, evaluator) -> RunSummary`.

- [ ] **Step 1: Write failing resumed-total and evaluator-failure tests**

```python
def test_sequential_target_is_total_trials(tmp_path):
    study = _study(tmp_path)
    trial = study.ask(); suggest_momentum_parameters(trial, _fixed_bounds())
    study.tell(trial, 5.0)
    calls = []
    summary = run_sequential_trials(study, 3, _fixed_bounds(),
        lambda params: calls.append(params) or _evaluation(1.0))
    assert len(calls) == 2
    assert terminal_trial_count(study) == 3
    assert summary.completed == 3
```

Add a first-call `ValueError` case; it must become `FAIL` and the runner must still reach three
terminal trials.

- [ ] **Step 2: Verify RED then implement sequential runner**

Run: `pytest tests/test_optuna_study.py -v -k 'target or failure'`

```python
while terminal_trial_count(study) < target_trials:
    trial = study.ask()
    parameters = suggest_momentum_parameters(trial, bounds)
    try:
        evaluation = evaluator(parameters)
        record_evaluation(trial, evaluation)
        study.tell(trial, evaluation.loss)
    except (ValueError, RuntimeError) as exc:
        trial.set_user_attr("failure", str(exc))
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
```

- [ ] **Step 3: Write failing real two-process coordinator test**

Use a module-level picklable evaluator returning a loss derived from parameters. Run two workers to
four target trials and assert SQLite contains exactly four terminal trials. Workers receive no
storage URL, proving only the parent writes SQLite.

Run: `pytest tests/test_optuna_study.py::test_process_runner_keeps_sqlite_in_parent -v`

- [ ] **Step 4: Implement bounded FIRST_COMPLETED scheduling**

```python
with ProcessPoolExecutor(max_workers=workers, initializer=initializer,
                         initargs=initargs) as pool:
    in_flight = {}
    # Ask only while terminal + running is below the exact target.
    # Map Future -> Trial, wait(FIRST_COMPLETED), tell in the parent, then refill.
```

On `KeyboardInterrupt`, stop refilling, cancel not-started futures, mark cancelled trials failed,
collect already-finished results, preserve storage, and return `interrupted=True`.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_optuna_study.py -v`

```bash
git add src/optimization/optuna_study.py tests/test_optuna_study.py
git commit -m "feat: coordinate parallel Bayesian trials"
```

---

### Task 6: Holdout Comparison and Five Report Artifacts

**Files:**
- Create: `src/optimization/momentum_report.py`
- Create: `tests/test_momentum_report.py`

**Interfaces:**
- Produces `adoption_checks(candidate_metrics, baseline_metrics) -> dict[str, bool]`.
- Produces `compare_holdout(holdout, candidate, baseline, config) -> dict[str, Any]`.
- Produces `write_optimization_outputs(study, prefix, metadata, comparison) -> dict[str, Path]`.

- [ ] **Step 1: Write failing adoption/output tests**

```python
def test_adoption_checks_names_each_gate():
    assert adoption_checks(
        {"total_return_pct": 20.0, "max_drawdown_pct": -31.0, "total_trades": 8},
        {"total_return_pct": 10.0}) == {
            "beats_baseline_return": True, "minimum_round_trips": True,
            "maximum_drawdown": False, "passed": False,
        }


def test_outputs_are_complete_and_standalone(tmp_path):
    paths = write_optimization_outputs(_study(), tmp_path / "momentum",
                                       _metadata(), _comparison())
    assert set(paths) == {"trials", "pareto", "importance", "summary", "report"}
    assert all(path.exists() for path in paths.values())
    html = paths["report"].read_text(encoding="utf-8")
    assert "<svg" in html and "https://" not in html
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_momentum_report.py -v`

- [ ] **Step 3: Implement exactly two holdout runs**

```python
def compare_holdout(holdout, candidate, baseline, config):
    candidate_metrics = run_holdout(holdout, candidate, config)
    baseline_metrics = run_holdout(holdout, baseline, config)
    return {"candidate_parameters": candidate, "baseline_parameters": baseline,
            "candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
            "adoption_checks": adoption_checks(candidate_metrics, baseline_metrics)}
```

If no interval profile exists, serialize `MomentumStrategy()` general defaults for the baseline.

- [ ] **Step 4: Implement CSV/JSON exports and graceful importance**

Flatten trial/fold attributes into stable columns. Write completed/failed states to `_trials.csv`,
active non-dominated rows to `_pareto.csv`, importance rows to `_importance.csv`, and complete config
and holdout evidence to `_summary.json`. Write the standalone report to `_report.html`. If importance
cannot be calculated, write headers and a warning rather than failing the study.

- [ ] **Step 5: Implement escaped standalone HTML/SVG**

Render the exact loss, metadata, selected parameters, top 25 trials, Pareto rows, fold metrics,
optimization-history SVG, importance bars, holdout comparison, adoption gates, and selected-value
boundary warnings. Escape every dynamic label/value and reference no external assets.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/test_momentum_report.py tests/test_momentum_evaluator.py -v`

```bash
git add src/optimization/momentum_report.py tests/test_momentum_report.py
git commit -m "feat: report Bayesian momentum results"
```

---

### Task 7: CLI and End-to-End Resume

**Files:**
- Create: `scripts/bayesian_search_momentum.py`
- Create: `tests/test_momentum_bayesian_cli.py`
- Modify: `src/optimization/__init__.py`

**Interfaces:**
- Produces `parse_arguments(argv=None) -> argparse.Namespace`.
- Produces `run_search(args) -> dict[str, Path]`.
- Produces `main(argv=None) -> int`.

- [ ] **Step 1: Write failing defaults and invalid-worker tests**

```python
def test_defaults_match_design():
    args = parse_arguments([])
    assert (args.trials, args.workers, args.startup_trials, args.seed) == (300, 1, 30, 42)
    assert (args.holdout_ratio, args.folds) == (0.20, 4)
    assert (args.drawdown_target, args.drawdown_weight,
            args.stability_weight) == (25.0, 2.0, 0.5)


def test_main_rejects_zero_workers(capsys):
    assert main(["--workers", "0"]) == 2
    assert "workers must be positive" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_momentum_bayesian_cli.py -v`

- [ ] **Step 3: Implement all documented flags and validation**

Validate positive finite capital/weights, positive trials/startup/workers, folds at least two, ratio
strictly between zero and one, existing data path, and output/storage parent creation. Preserve both
existing `--disable-*` flag meanings.

- [ ] **Step 4: Compose optimization without passing holdout to runners**

```python
data = load_ohlcv(args.data)
interval = infer_interval_minutes(data.index)
bounds = resolve_search_bounds(interval)
max_lookback = max(bounds.roc_max, bounds.momentum_max)
folds, holdout = split_holdout_and_folds(data, args.holdout_ratio,
                                         args.folds, max_lookback)
run_config = BacktestRunConfig(args.capital, args.coin.upper(),
    not args.disable_drawdown_breaker, not args.disable_loss_cooldown)
loss_config = LossConfig(args.drawdown_target, args.drawdown_weight,
    args.stability_weight, args.minimum_round_trips,
    args.missing_round_trip_penalty)
metadata = build_metadata(args.data, data, interval, args.coin.upper(),
    args.capital, bounds, loss_config, args.holdout_ratio, args.folds, run_config)
study = create_or_load_study(args.storage, args.study_name, metadata,
    args.seed, args.startup_trials, args.workers > 1)
recover_stale_trials(study)
if args.workers == 1:
    evaluator = functools.partial(evaluate_parameters, folds,
        loss_config=loss_config, run_config=run_config)
    summary = run_sequential_trials(study, args.trials, bounds, evaluator)
else:
    initargs = (str(args.data), args.holdout_ratio, args.folds,
        max_lookback, loss_config, run_config)
    summary = run_process_trials(study, args.trials, args.workers, bounds,
        initialize_worker, initargs, evaluate_worker)
candidate = dict(study.best_trial.params)
comparison = compare_holdout(holdout, candidate, resolved_baseline, run_config)
return write_optimization_outputs(study, args.output_prefix, metadata, comparison)
```

If interrupted, export partial trial/Pareto/importance artifacts without holdout and return 130. If
no completed trial exists, do not access `best_trial`. Configuration errors return 2 with one English
message.

- [ ] **Step 5: Write a tiny real resume integration test**

Generate 600 synthetic 5-minute rows, override bounds to short lookbacks, run target two trials,
rerun target three on the same SQLite study, and assert exactly three terminal trials plus non-empty
holdout metrics in summary JSON.

- [ ] **Step 6: Verify all optimization tests and commit**

Run:

```bash
pytest tests/test_optimization_data.py tests/test_momentum_objective.py \
  tests/test_momentum_evaluator.py tests/test_optuna_study.py \
  tests/test_momentum_report.py tests/test_momentum_bayesian_cli.py -v
```

```bash
git add scripts/bayesian_search_momentum.py src/optimization/__init__.py \
  tests/test_momentum_bayesian_cli.py
git commit -m "feat: add Bayesian momentum search CLI"
```

---

### Task 8: Documentation, Full Verification, and Real-Data Smoke

**Files:**
- Modify: `README.md:138-150`
- Modify: `src/strategy_and_backtest_tutorial.md:288-370`

**Interfaces:**
- Documents the exact command, loss, resume, reproducibility, artifacts, holdout, and profile policy.

- [ ] **Step 1: Document the approved real command**

```bash
python -u scripts/bayesian_search_momentum.py \
  --data data/historical/btc_5m_1800d.csv \
  --coin btc --trials 300 --workers 4 \
  --disable-drawdown-breaker --disable-loss-cooldown \
  --study-name btc_5m_1800d_momentum_v1 \
  --storage results/btc_5m_1800d_momentum_v1.db \
  --output-prefix results/btc_5m_1800d_momentum_bayesian
```

Explain loss defaults, target-total resume, metadata mismatch behavior, non-bit-identical parallel
ordering, five artifacts, Pareto interpretation, and the prohibition on feeding holdout results back
into optimization.

- [ ] **Step 2: Format and run the full test suite**

```bash
black src/ scripts/bayesian_search_momentum.py tests/ --line-length 100
pytest tests/ -v
git diff --check
```

Expected: zero failures.

- [ ] **Step 3: Run a three-trial smoke study on repaired real data**

```bash
python -u scripts/bayesian_search_momentum.py \
  --data data/historical/btc_5m_1800d.csv \
  --coin btc --trials 3 --workers 1 --startup-trials 3 \
  --disable-drawdown-breaker --disable-loss-cooldown \
  --study-name btc_5m_bayesian_smoke_v1 \
  --storage results/btc_5m_bayesian_smoke_v1.db \
  --output-prefix results/btc_5m_bayesian_smoke_v1
```

Expected: exit 0, three terminal trials, holdout candidate/baseline metrics, five non-empty outputs.
This is operational verification, not a recommendation.

- [ ] **Step 4: Verify resume reaches exactly four trials**

Repeat the command with `--trials 4`.

Expected: one new trial, four total terminal trials, refreshed outputs, no metadata error.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md src/strategy_and_backtest_tutorial.md
git commit -m "docs: explain Bayesian momentum optimization"
```

- [ ] **Step 6: Record final evidence**

```bash
pytest tests/ -q
git status --short
git log -8 --oneline
```

Expected: all tests pass; only the pre-existing untracked `AGENTS.md` remains; generated databases
and reports stay ignored.
