# BTC TCN Capacity, Training Budget and History Study

This study implements the approved comparison of a longer-trained small TCN and a wider TCN, and extends BTC hourly history. It preserves next-open execution, commission, slippage, trade gates, the 5% position stop and 18% equity breaker. It does not deploy a trading strategy.

## Data

The official Binance BTCUSDT spot archive acquisition produced 77,111 real hourly candles from 2018-01-01 00:00 through 2026-10-23 23:00 UTC. The dataset SHA256 is `8aceb05219632794101bff5317067ee051d6ce5b0f9124db4260fcd36a5b0e56`. This is about 4.4 times the original 17,512-candle snapshot, and uses a consistent Binance spot source across all cells.

The downloader verified 128 ZIP files against the published SHA256 checksums. It quarantined 43 off-grid source rows and recovered 42 aligned candles from the official REST API. Twenty-seven genuine gap ranges remain, containing 121 missing hours. See [the archive audit](BTC_ARCHIVE_DATA.md) and the dataset `.manifest.json` for exact source records and ranges.

The training-only gap mode resets causal features at each continuous block and excludes the first `168 + lookback - 1 = 335` rows of each block from decision windows. It also purges horizon labels before each gap and splits policy segments into separate eligible runs. No missing candle is synthesized. Validation/evaluation and their preceding warmup must remain continuous. The standalone trainer still rejects gaps unless explicitly enabled through the Python API.

## Fixed protocol

| Partition | UTC dates | Purpose |
|---|---|---|
| Short training history | 2023-01-01 to 2025-01-01 exclusive | Recent-history control |
| Long training history | 2018-01-01 to 2025-01-01 exclusive | Expanded regime coverage |
| Shared selection validation | 2025-01-01 to 2026-01-01 exclusive | Select pretraining and policy checkpoints; rank recipes |
| Historical evaluation | 2026-01-01 to 2026-10-07 04:00 exclusive | Common retrospective comparison on previously researched dates |
| Recent tail | 2026-10-07 04:00 to 2026-10-24 exclusive | Evaluated after every recipe/checkpoint is frozen |

The short history contains 17,543 observed training bars and 16,825 eligible training closes after gap/warmup/label purging. The long history contains 61,247 training bars and 52,746 eligible training closes. Both use exactly 8,760 validation bars and 6,700 historical-evaluation bars. The recent tail contains only 404 hours (16 days 20 hours), so its statistical evidence is limited.

Absolute split indices prevent validation dates drifting when earlier training years are added. Scalers fit each cell's training partition only. Forecast targets remain 4/12/24-hour next-open-to-future-close returns, and validation losses purge the last 24 validation bars. A larger history expands market-regime coverage; it does not lengthen the encoder's 127-feature-row receptive field.

## Predeclared cells

Each case runs all seeds `[7, 19, 42]`; seed 7 is the primary seed regardless of results. No seed is selected using test outcomes.

| Case | Hidden width | Parameters | Training history | Pretraining / policy caps |
|---|---:|---:|---|---|
| `short_h32_baseline` | 32 | 27,978 | 2023-2024 | 6 / 4 epochs |
| `short_h128_baseline` | 128 | 419,082 | 2023-2024 | 6 / 4 epochs |
| `short_h32_extended` | 32 | 27,978 | 2023-2024 | 50 / 30 epochs |
| `short_h128_extended` | 128 | 419,082 | 2023-2024 | 50 / 30 epochs |
| `long_h32_extended` | 32 | 27,978 | 2018-2024 | 50 / 30 epochs |
| `long_h128_extended` | 128 | 419,082 | 2018-2024 | 50 / 30 epochs |

All cells share pretraining stride 12, batch size 32, six causal residual blocks, dropout 0.1 and the same execution/risk controls. This yields 44 pretraining updates per short-history epoch and 138 per long-history epoch. Actual completed updates are recorded rather than inferred from maximum epoch caps.

The extended recipe adds:

- Minimum 10 epochs in each stage, up to the declared 50/30 caps.
- Early stopping after eight validation observations without an improvement, subject to the minimum epoch requirement.
- LR reduction by 0.5 after three validation observations without improvement, with floor `1e-6`.
- Restoration of the best validation-quantile pretraining weights before policy optimization.
- Policy checkpoint selection and patience driven by the original lexicographic feasibility/return/cost score, not by recent-tail outcomes.

Pretraining starts at LR 0.001 and policy training starts with a fresh optimizer at LR 0.001/3. Extended-vs-baseline comparisons test this complete optimization recipe, not only epoch count. Long-history runs perform more updates per epoch, so history comparisons are practical training-data comparisons rather than compute-matched scaling experiments.

On the execution host, MPS pretraining was approximately six times faster than CPU for the wider model in a synthetic batch probe. Every study cell therefore uses MPS for pretraining and CPU with two threads for policy simulation. The model returns to CPU after pretraining. Device choice is explicit; unavailable MPS fails instead of silently falling back. Fixed seeds and deterministic algorithms improve repeatability, but do not imply bit-identical CPU/MPS results or reproducibility across dependency versions.

## Frozen evaluation and recovery

The study writes `protocol.json` before fitting any case. It includes exact configs, seeds, absolute dates, data fingerprint, relevant modeling/backtest/metric/strategy source fingerprints and runtime versions. Each completed case immediately saves report/checkpoint hashes in `completion.json`.

Recipes are ranked only by all-seed validation-drawdown feasibility, then median validation net return, then lower median validation costs. All seeds and rejected/cash outcomes remain visible. `frozen.json` stores the selected recipe and every checkpoint fingerprint before recent-tail accounts are opened. The recent tail never selects a width, seed, budget or checkpoint. Historical 2026 outcomes are already researched data; the new tail is a short retrospective check, not a prospective guarantee.

An explicit resume verifies the original protocol, completed development dataset, case report and checkpoint hashes. It rejects changed code/configuration/data or corrupted completed artifacts. Completed cells are reused; incomplete cells are not overwritten. Preserve an incomplete case directory under another name before resuming its deterministic rerun. TensorBoard remains inside each case and seed.

## Run and inspect

```bash
.venv/bin/python scripts/fetch_btc_archive.py \
  --start 2018-01-01 --end 2026-10-24 \
  --output data/historical/btc_binance_1h_20180101_20261024.csv

.venv/bin/python scripts/run_tcn_study.py \
  --data data/historical/btc_binance_1h_20180101_20261024.csv \
  --output results/tcn_capacity_history_20261024 \
  --pretrain-device mps

.venv/bin/python -m tensorboard.main \
  --logdir results/tcn_capacity_history_20261024 \
  --host 127.0.0.1 --port 6007
```

Use `--pretrain-device cpu` on hosts without MPS, with a fresh output directory. To resume the same immutable study, repeat the original command with `--resume`. The default seed list is 7/19/42. Pass `--no-tensorboard` only when predeclaring the study; changing it during resume changes the protocol.

Existing single-run training also supports `--batch-size`, `--pretrain-stride`, `--early-stopping-patience`, `--lr-patience`, `--lr-factor`, `--min-lr`, `--restore-best-pretrain`, `--min-pretrain-epochs`, `--min-policy-epochs`, `--cpu-threads` and `--pretrain-device`. Defaults preserve the original CPU training recipe with no active scheduler or early stopping.

## Diagnostics and artifacts

The [TensorBoard guide](TENSORBOARD.md) describes base scalars and weighted policy losses. This extension also records per-update encoder, quantile-head and policy-head gradient norms before clipping, and policy epoch zero-gradient/update fractions. Zero policy-head gradients can occur when hard predictive/turnover gates suppress orders. This helps distinguish optimization failure from a policy with no active trading gradient.

New history rows include actual optimizer-update counts, LR before/after scheduling, elapsed time and stopping reason. `training_summary` in each checkpoint/audit/report records completed stage epochs and updates. The saved policy weights are the validation-selected candidate and may come from an earlier epoch than the final training update count. Pretraining loss curves include epochs after the restored best pretraining epoch; restoration is explicitly logged.

Study output contains per-case manifests/reports/audits/checkpoints/TensorBoard events, a global protocol/progress/frozen selection, per-run recent-tail and doubled-cost equity/fills, common recent baselines, and aggregate `report.json`, `report.md`, `report.html`. Raw data and full experiment outputs stay under ignored data/results directories; implementation and documentation are reviewable in Git.

Source: [study runner](../src/learning/study.py), [training controls](../src/learning/train.py), [fixed/gapped partitions](../src/learning/data.py), [study CLI](../scripts/run_tcn_study.py).

## Scope of validation evidence

This first study uses one common 2025 selection year; it does not implement multiple expanding-window model-refit folds. Per-seed variation measures optimization randomness on one market history rather than market-sample uncertainty. A full walk-forward refit study remains necessary before treating recipe selection as robust across eras. Historical and recent evaluation accounts each reset to 10,000 cash with fresh risk state; their returns are separate experiments, not a single continuous live account.

## Completed results

See [the completed October 24 results](TCN_STUDY_RESULTS_20261024.md) for all seed ranges, actual budgets, forecast generalization and the zero-fill recent-tail diagnosis. The full local artifact directory is `results/tcn_capacity_history_20261024/`.
