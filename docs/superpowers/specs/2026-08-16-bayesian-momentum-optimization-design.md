# Bayesian Momentum Optimization Design

## Goal

Add a reusable Bayesian optimizer for the six asymmetric Momentum parameters. The optimizer must
prefer robust annualized return rather than the highest return from one historical period, penalize
excessive drawdown and instability, support interrupted/resumed studies, and keep the final holdout
period isolated from parameter selection.

This feature does not make the backtest differentiable and does not use gradient descent. Momentum
lookbacks are integers, signal thresholds create discontinuous events, and drawdown depends on a
stateful equity path. These properties make the existing backtest a black-box objective suited to
Optuna's Tree-structured Parzen Estimator (TPE).

## Scope

The first version will:

- optimize the six BTC/5m Momentum parameters independently;
- work with any supported coin and candle interval when compatible data is supplied;
- evaluate each trial on four chronological optimization folds;
- reserve the final 20% of rows as a one-time holdout;
- minimize one explicit composite loss;
- persist trials and allow safe resume;
- support process-based parallel backtests;
- export trials, robust metrics, Pareto candidates, parameter importance, and a standalone HTML
  report;
- compare the selected candidate with the current coin/interval runtime profile on the holdout;
- report adoption evidence without editing `momentum_profiles.py`.

Training a predictive price model, changing Momentum signal logic, automatic profile replacement,
distributed multi-machine execution, and differentiable surrogate trading are out of scope.

## Dependencies

Add `optuna>=4.0,<5.0` to `requirements.txt`. Optuna is not currently installed in the project
environment. The implementation will use stable Optuna 4 APIs and avoid deprecated journal-storage
aliases.

No plotting dependency is required. The HTML report will use tables and small inline SVG charts so
it remains standalone and consistent with the repository's existing HTML outputs.

## Command-Line Interface

Create `scripts/bayesian_search_momentum.py`. A representative run is:

```bash
python -u scripts/bayesian_search_momentum.py \
  --data data/historical/btc_5m_1800d.csv \
  --coin btc \
  --trials 300 \
  --workers 4 \
  --holdout-ratio 0.20 \
  --folds 4 \
  --drawdown-target 25 \
  --drawdown-weight 2.0 \
  --stability-weight 0.5 \
  --disable-drawdown-breaker \
  --disable-loss-cooldown \
  --study-name btc_5m_1800d_momentum_v1 \
  --storage results/btc_5m_1800d_momentum_v1.db \
  --output-prefix results/btc_5m_1800d_momentum_bayesian
```

Defaults:

- trials: `300`;
- workers: `1` for deterministic sequential execution;
- startup random trials: `30`;
- random seed: `42`;
- holdout ratio: `0.20`;
- optimization folds: `4`;
- minimum completed round trips per fold: `2`;
- drawdown target: `25%`;
- drawdown weight: `2.0`;
- stability weight: `0.5`;
- missing-round-trip penalty: `25.0` per missing round trip.

The existing `--disable-drawdown-breaker` and `--disable-loss-cooldown` meanings will be preserved.
Optimization documentation will recommend disabling both so the objective observes raw strategy
risk; stop-loss, costs, slippage, minimum holding period, and daily trade limits remain active.

## Search Space

The script will infer the dominant candle interval and convert duration bounds into raw bars:

| Parameter | Duration range | Sampling |
|---|---:|---|
| buy/sell ROC period | 1 hour to 28 days | logarithmic integer |
| buy/sell momentum period | 30 minutes to 14 days | logarithmic integer |
| buy/sell threshold | 0.005 to 0.25 | logarithmic float |

Each buy and sell parameter is sampled independently. The wide ROC upper bound deliberately extends
beyond the previous 4,032-bar boundary winner. The resolved raw-bar bounds are stored in study
metadata and written to every report.

The sampler will be `TPESampler` with `n_startup_trials=30`, `multivariate=True`, a configurable
seed, and `constant_liar=True` when more than one trial is in flight. Multivariate TPE is appropriate
because the six entry/exit parameters interact, although its Optuna interface is still marked
experimental; pinning Optuna below version 5 limits compatibility drift.

## Chronological Data Isolation

After validating and sorting the OHLCV input:

1. The first 80% becomes the optimization pool.
2. The latest 20% becomes the untouched holdout.
3. The optimization pool is split into four ordered, non-overlapping folds of nearly equal size.
4. Every trial starts with fresh capital and a flat position in each fold.
5. The maximum search lookback must be shorter than every fold; otherwise the command fails before
   creating trials.

The folds are regime tests, not train/validation pairs for a fitted statistical model. Momentum
parameters are evaluated independently on every fold, and only the aggregate fold metrics enter the
loss. The holdout is never exposed to TPE and is evaluated only after optimization finishes.

## Composite Loss

For fold `i`, collect annualized return `A_i`, maximum drawdown `M_i` (negative percentage), and
completed round trips `T_i = floor(total_trades_i / 2)`.

Define:

```text
robust_return = median(A_i)
worst_drawdown = max(abs(M_i))
instability = population_stddev(A_i)
missing_round_trips = sum(max(0, minimum_round_trips - T_i))

loss =
    -robust_return
    + drawdown_weight * max(0, worst_drawdown - drawdown_target)
    + stability_weight * instability
    + missing_round_trip_penalty * missing_round_trips
```

All return and drawdown values use percentage points, so the default weights have interpretable
units. Drawdown below the target is not rewarded; drawdown above it is penalized linearly. The
activity penalty prevents inactive parameter sets from winning through artificially smooth equity
curves.

The loss implementation will be a pure function with explicit inputs so its behavior can be unit
tested independently of Optuna and the backtest engine. Non-finite or missing fold metrics fail the
trial and are recorded rather than converted into a potentially competitive numeric loss.

## Optimization Architecture

Keep Optuna coordination in the parent process:

1. The parent opens one SQLite-backed study with `load_if_exists=True`.
2. It uses Optuna's ask/tell interface to create RUNNING trials and sample the six parameters.
3. A `ProcessPoolExecutor` sends only resolved parameters and immutable run configuration to worker
   processes.
4. Each worker loads the OHLCV data once, evaluates its assigned parameters on all folds, and
   returns fold and aggregate metrics.
5. Only the parent calls `study.tell()` and writes trial user attributes.

This avoids CPU-bound `n_jobs` threads and prevents concurrent SQLite writers. It also gives an exact
global trial limit. On resume, stale RUNNING trials left by an interrupted parent are marked failed
before new work is scheduled.

Sequential mode uses the same evaluation path without creating a process pool. Parallel scheduling
may finish trials in a different order, so `--workers 1 --seed 42` is the reproducible mode; resumed
or parallel studies preserve all completed evidence but are not promised to produce a bit-for-bit
identical future suggestion sequence.

## Resume Safety

The study stores an immutable metadata record containing:

- normalized data path and SHA-256 digest;
- row count and first/last timestamp;
- inferred interval;
- coin, capital, fold count, and holdout ratio;
- resolved search bounds;
- loss weights and activity requirement;
- all backtest risk-control flags;
- optimizer schema version.

If an existing study's metadata differs, the command exits with a clear error and asks for a new
study name or storage file. This prevents silently mixing trials created from different datasets,
loss definitions, or risk settings.

## Candidate Selection and Holdout

The completed trial with the lowest composite loss is the selected candidate. The report will also
compute a non-dominated Pareto set from all sufficiently active completed trials using:

- maximize robust annualized return;
- minimize worst drawdown.

After selection, run exactly two holdout backtests with identical engine settings:

1. selected Bayesian candidate;
2. current runtime profile returned by `get_momentum_profile(coin, interval)`.

Reuse the existing adoption evidence rules: the candidate must beat the baseline holdout return,
complete at least two holdout round trips, and keep holdout drawdown at or above `-30%`. The report
shows each pass/fail component. No code or profile is changed automatically.

## Outputs

For output prefix `results/btc_5m_1800d_momentum_bayesian`, write:

- `_trials.csv`: one row per trial with parameters, loss, state, duration, fold metrics, and
  aggregates;
- `_pareto.csv`: robust-return/drawdown non-dominated candidates;
- `_importance.csv`: parameter importance when enough valid trials exist;
- `_summary.json`: selected parameters, configuration, holdout comparison, and adoption checks;
- `_report.html`: standalone summary containing loss definition, optimization history, importance,
  top candidates, Pareto table, fold performance, holdout comparison, and boundary warnings.

SQLite storage and generated result files remain ignored artifacts rather than Git-tracked source.

## Error Handling

Fail before optimization for missing/non-finite OHLCV values, duplicate timestamps, ambiguous candle
cadence, invalid ratios or weights, incompatible folds/lookbacks, non-positive capital, or metadata
mismatch on resume. If no coin/interval-specific runtime profile exists, the baseline uses the
Momentum strategy's general defaults, matching `run_backtest.py` behavior.

A single invalid parameter trial is marked failed without stopping the whole study. Keyboard
interrupt stops scheduling new work, waits briefly for in-flight workers, persists completed trials,
exports the current reports, and exits non-zero with a resume command. Unexpected worker exceptions
are attached to the failed trial and summarized without exposing local secrets.

## Testing

Add `tests/test_momentum_bayesian_search.py` with TDD coverage for:

- interval-aware search-bound conversion and parameter types;
- chronological 80/20 isolation and ordered non-overlapping folds;
- exact loss calculations, drawdown threshold behavior, instability, and activity penalties;
- Pareto dominance and deterministic tie handling;
- metadata creation and resume mismatch rejection;
- stale RUNNING-trial recovery;
- disabled risk-control propagation to every fold and both holdout runs;
- sequential ask/tell execution with a tiny synthetic dataset;
- bounded process scheduling without concurrent SQLite access;
- CSV, JSON, and standalone HTML output;
- current-profile holdout comparison and adoption checks;
- CLI defaults, validation errors, interrupt handling, and resume behavior.

The existing full test suite must remain green. Long real-data optimization is an explicit manual
run and will not be part of the unit test suite.

## Documentation

Update `README.md` and `src/strategy_and_backtest_tutorial.md` with:

- why direct gradient descent does not apply;
- the composite-loss formula and default weights;
- a BTC/5m 1,800-day example command;
- storage/resume and reproducibility behavior;
- interpretation of Pareto and parameter-importance outputs;
- the rule that holdout results are evidence, not another optimization target.

## External API Basis

The design follows Optuna's stable documentation for TPE sampling, persistent studies, ask/tell
study coordination, and optimization-result analysis:

- https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html
- https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
- https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
