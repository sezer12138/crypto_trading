# BTC TCN Trading Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Follow TDD.

**Goal:** Train and retrospectively validate a reproducible risk-aware BTC/cash TCN.

**Architecture:** Causal features and purged chronological labels feed a quantile encoder and state-aware allocation policy. A differentiable execution replay trains the policy; the existing BacktestEngine provides authoritative evaluation.

**Tech Stack:** Python, pandas, NumPy, optional PyTorch, pytest.

**Spec:** docs/superpowers/specs/2026-10-07-tcn-trading-design.md

## Global Constraints

- English documentation, comments and logs; no existing strategy defaults changed.
- Long-only BTC/cash, hourly bars, train-only scaling, next-open execution.
- Frozen 60/20/20 partitions; purged 24h labels; retrospective results only.
- No deployment or profile activation; artifact directory ignored by Git.

## Review Focus

- Future perturbations cannot change past features, predictions or fills.
- Invalid candles, gaps and short partitions must fail explicitly.
- Differentiable and canonical execution must agree on gaps, partial fills, costs and stops.
- Reloaded checkpoints must reproduce decisions without refitting the scaler.
- Validation/holdout rejection must remain visible; constant cash cannot masquerade as a profitable improvement.

### Task 1: Causal data and quantile encoder

Files: src/learning/data.py, src/learning/model.py, tests/test_learning.py.
Interfaces: prepare_data(frame, config) -> PreparedData; TCNPolicy.encode(features) -> (encoded, quantiles); allocate(encoded, quantiles, state) -> target_weight.
- [x] Write prefix invariance, label timing/purge, train-only scaling, malformed/gapped data and ordered quantile tests.
- [x] Run tests and verify RED for missing learning package.
- [x] Implement causal features, scaling, split metadata and compact TCN.
- [x] Run tests and verify GREEN.

### Task 2: State-aware execution and objective

Files: src/learning/execution.py, src/learning/strategy.py, tests/test_learning.py.
Interfaces: simulate_path(model, encoded, quantiles, bars, config) -> differentiable equity/turnover; LearnedStrategy.generate_order(...) -> TradeOrder or None.
- [x] Write execution identity, next-open gap, stop, protective exit, objective gradient and costs tests.
- [x] Verify RED then implement shared allocation/risk logic and differentiable accounting.
- [x] Verify GREEN against BacktestEngine and checkpoint replay.

### Task 3: Training CLI and research report

Files: src/learning/train.py, src/learning/report.py, scripts/train_tcn.py, requirements-learning.txt, docs/TCN_TRADING.md.
Interfaces: run_experiment(data_path, output, config, seeds) -> report manifest; load_checkpoint(path) -> LearnedStrategy.
- [x] Write a small real end-to-end training/export/reload test and validation-only checkpoint selection tests; verify RED.
- [x] Implement pretraining, path optimization, frozen checkpoint selection, baseline/stress evaluation, checkpoints and HTML report.
- [x] Verify GREEN, run full suite, train on cached BTC history for three predeclared seeds, inspect report.
- [x] Review complete diff and results, resolve material findings, report limitations and exact reproduction command.
