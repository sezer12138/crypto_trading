# TCN Capacity and Data Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the main integration. The independent archive task uses parallel delegation.

**Goal:** Execute and report a reproducible BTC history/capacity/training-budget comparison.

**Architecture:** Preserve the causal TCN and accounting while adding absolute split controls, training schedules, and an immutable study runner. Download and audit official historical data independently of model implementation.

**Tech Stack:** Python, PyTorch, pandas, TensorBoard, official Binance archives/REST.

**Spec:** docs/superpowers/specs/2026-10-24-tcn-capacity-data-design.md

## Global Constraints

- English-only new docs/comments/logs; preserve prior checkpoints and default training behavior.
- Shared absolute validation dates and no final-tail-based selection.
- Genuine continuous hourly candles; audit rather than synthesize unresolved gaps.
- Seed list and experiment cells frozen before training; no live orders or deployment.

## Review Focus

- Invalid/reversed/nonexistent split timestamps must fail explicitly.
- Forecast labels cannot cross validation/test boundaries.
- Logging, LR decay and early stopping must not accidentally rank on final outcomes.
- Best-pretraining restoration must restore actual weights, not just a loss number.
- Interrupted experiments retain data/protocol/history and failure evidence.

## Tasks

- [x] Data: failing fixture/checksum/time-unit/gap tests, verified downloader, actual acquisition and audit (parallel archive worker).
- [x] Training: failing schedule/checkpoint/diagnostic tests, opt-in configuration, early stopping, LR decay, update accounting and head-gradient diagnostics; legacy behavior tests stay green.
- [x] Protocol: failing absolute-split/label-purge tests, extend prepare_data with explicit indices while preserving ratio defaults.
- [x] Study: frozen cell manifest and shared-date runner; rank only validation, evaluate frozen models on common historical periods and then recent tail, export aggregate evidence and report.
- [x] Execute: predeclare runtime-safe common sampling settings, run all declared cells/seeds to completion, preserve every outcome and compare capacity/budget/history.
- [x] Verify: full suite, formatting/diff checks, independent final review and fix material findings with regression tests; document results and limitations.

## Verification and review record

- Data acquisition: 128 verified ZIPs; full 77,111-row dataset and source manifest retained.
- Absolute partitions/gap handling and study freeze/resume tests were observed RED then GREEN.
- Independent review found two material resume-integrity issues; both were fixed with regression tests before the immutable real run: canonical dependency fingerprints and immediate completed-cell checkpoint/report hashes.
- Independent review of scheduling, gradient diagnostics and MPS handoff found no material training defects.
- Full suite: 311 passed, 43 pre-existing warnings. All 18 real runs completed; unchanged protocol hashes, checkpoint hashes and explicit resume verified.
- Scope decision: use one fixed selection year in this first factorial study rather than a full multi-fold refit study; the latter remains a stated limitation because recipe stability across eras is unproven.
