# BTC TCN Study Results — October 24, 2026

The 18-run study completed all predeclared cases and seeds. Expanding the model and training budget did not establish a stable improvement in historical trading performance. The validation-selected recipe was `short_h128_extended`, but its primary seed returned -1.69% with 8.39% maximum drawdown on the historical evaluation; its three-seed median return was -0.69%.

The short-history hidden-32 extended recipe had historical returns of +0.21%, +0.85% and +1.69% across seeds (median +0.85%; the list is sorted by return, not by seed). The generic Momentum baseline returned +8.14% with 15.94% MDD under the same costs and risk settings. Lower learned-policy drawdowns do not establish superior returns at matched realized risk/exposure.

## What the increased budget showed

- The wider model has 419,082 parameters versus 27,978, about 15 times as many.
- Each long-history run executed 2,550–3,240 total optimizer updates, versus 408 in the short-budget controls. Epoch caps were 50/30; validation patience stopped all extended runs before those caps. Across all 18 models, 14,374 pretraining and 11,160 policy updates completed.
- Large-model forecast validation loss often reached its optimum in epochs 1–4, then worsened while training loss continued falling. Best-pretraining restoration and validation-selected policy checkpoints avoided automatically exporting the last epoch.
- Adding history made policies more conservative. Mean zero-policy-gradient fractions were 94.1% for the long-history hidden-32 group and 87.9% for hidden-128. More updates therefore did not translate into equally many active trading-gradient updates.

## Why the recent tail returned zero

All 18 models made **zero fills** during the 404-hour recent tail. There were 34 regular decision bars per model, and zero bars in every model passed the unchanged predictive edge gate:

```text
24h predicted median return > 2*(commission + slippage) + entry_margin
                           > 2*(0.001 + 0.001) + 0.002
                           > 0.006  (0.6%)
```

For the validation-selected recipe, maximum predicted 24h median returns across the entire recent tail were about 0.314%, 0.379% and 0.272% for seeds 7/19/42. The raw allocation heads were not uniformly zero, but the edge gate suppressed entries. The resulting zero return/drawdown reflects cash, not demonstrated profitable prediction. The gate was diagnosed after frozen evaluation; it was not lowered or retuned using these outcomes.

Recent-tail risk-controlled buy-and-hold returned -4.15%, MA cross -7.05%, and both cash and generic Momentum returned 0%. These short-period outcomes do not establish sustained learned alpha.

## Evidence and limitations

All 18 checkpoint fingerprints and completed-cell manifests were verified after training. Maximum tensor/canonical replay equity difference was `7.3142e-7` of initial capital, below the `5e-5` tolerance. Explicit resume reused completed evidence without retraining or changing selection. The study retains all rejected/loss/cash candidates.

The data audit, exact protocol and commands are in [the study design and reproduction guide](TCN_CAPACITY_HISTORY_STUDY.md). Full local experiment artifacts are in `results/tcn_capacity_history_20261024/`, including `protocol.json`, `frozen.json`, `report.json`, `summary.json`, `recent_prediction_audit.json`, checkpoints, equity/fills and TensorBoard events. They are intentionally not versioned as source code.

This is a fixed-2025-validation retrospective study, not a multi-fold expanding-window refit experiment. Historical 2026 dates had already been researched; the new tail spans only 16 days 20 hours. Each evaluation account resets cash/risk state. Extended-vs-baseline comparisons also change scheduling/restoration, and expanded-history runs use more compute per epoch. The results support investigating prediction/cost-gate alignment, policy-gradient availability and rolling refit validation before scaling further; no strategy was activated.


# Full numerical summary

Validation-selected recipe: **short_h128_extended**. Primary seed: **7**.

Every cell ran seeds 7, 19 and 42. Values below show median [minimum, maximum] across the three seeds. These are seed ranges on one market path, not confidence intervals or independent market replications.

| Case | Validation return % | Historical return % | Historical MDD % | Recent return % | Recent MDD % |
|---|---:|---:|---:|---:|---:|
| short_h32_baseline | 7.69 [6.89, 10.65] | 0.54 [-1.35, 1.69] | 3.12 [2.57, 3.45] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| short_h128_baseline | 9.07 [6.10, 10.35] | -5.59 [-15.69, -3.43] | 9.08 [5.36, 16.54] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| short_h32_extended | 9.59 [9.23, 10.97] | 0.85 [0.21, 1.69] | 3.81 [2.69, 5.31] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| short_h128_extended | 14.14 [9.22, 15.33] | -0.69 [-1.69, 1.38] | 6.81 [4.44, 8.39] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| long_h32_extended | 1.27 [0.00, 2.75] | -0.02 [-0.88, 0.44] | 0.25 [0.22, 2.27] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| long_h128_extended | 0.70 [0.00, 4.39] | 0.00 [0.00, 0.04] | 0.00 [0.00, 1.92] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

## Actual training budgets and gradient activity

| Case | Parameters | Pretrain epochs | Pretrain updates | Policy epochs | Policy updates | Mean zero-policy-gradient fraction |
|---|---:|---:|---:|---:|---:|---:|
| short_h32_baseline | 27,978 | 6 [6, 6] | 264 [264, 264] | 4 [4, 4] | 144 [144, 144] | 54.9% |
| short_h128_baseline | 419,082 | 6 [6, 6] | 264 [264, 264] | 4 [4, 4] | 144 [144, 144] | 26.6% |
| short_h32_extended | 27,978 | 12 [12, 12] | 528 [528, 528] | 15 [10, 25] | 540 [360, 900] | 49.9% |
| short_h128_extended | 419,082 | 10 [10, 12] | 440 [440, 528] | 14 [10, 17] | 504 [360, 612] | 38.0% |
| long_h32_extended | 27,978 | 13 [11, 15] | 1794 [1518, 2070] | 10 [10, 10] | 1170 [1170, 1170] | 94.1% |
| long_h128_extended | 419,082 | 10 [10, 12] | 1380 [1380, 1656] | 10 [10, 10] | 1170 [1170, 1170] | 87.9% |

## Forecast generalization diagnostics

| Case | Best validation epoch | Best validation loss | Last validation loss | Last training loss |
|---|---:|---:|---:|---:|
| short_h32_baseline | 4 [4, 4] | 0.3744 [0.3732, 0.3761] | 0.3808 [0.3781, 0.3831] | 0.4149 [0.4133, 0.4157] |
| short_h128_baseline | 2 [1, 4] | 0.3752 [0.3730, 0.3772] | 0.3894 [0.3873, 0.3932] | 0.3974 [0.3937, 0.3987] |
| short_h32_extended | 4 [4, 4] | 0.3744 [0.3732, 0.3761] | 0.3824 [0.3823, 0.3833] | 0.3995 [0.3982, 0.4024] |
| short_h128_extended | 2 [1, 4] | 0.3752 [0.3730, 0.3772] | 0.4278 [0.4186, 0.4674] | 0.3291 [0.2551, 0.3385] |
| long_h32_extended | 5 [3, 7] | 0.3721 [0.3720, 0.3732] | 0.3800 [0.3769, 0.3807] | 0.5085 [0.5036, 0.5117] |
| long_h128_extended | 2 [1, 4] | 0.3734 [0.3713, 0.3782] | 0.4093 [0.4088, 0.4445] | 0.4191 [0.3483, 0.4375] |

## Paired historical comparisons

Differences are return percentage points for matched seeds; historical outcomes do not rank recipes.

| Comparison | Median paired return difference (pp) | Seeds with improved return |
|---|---:|---:|
| Optimization recipe, width 32 | +0.31 | 2/3 |
| Optimization recipe, width 128 | +6.97 | 3/3 |
| Capacity, short history | -0.90 | 0/3 |
| Capacity, long history | +0.02 | 2/3 |
| Added history, width 32 | -1.71 | 1/3 |
| Added history, width 128 | +0.69 | 2/3 |

## Interpretation limits

- The 2026 historical evaluation reuses previously researched dates. It is not a blind test.
- The 404-hour recent tail was opened after all recipes and checkpoints were frozen; it is too short to establish sustained profitability.
- An extended budget includes LR decay and best-pretraining restoration; the comparison is not an isolated epoch-count effect.
- Longer-history cells receive more updates per epoch and therefore more compute.
- Smaller drawdown can reflect cash holdings or permanent breaker halts rather than a better allocation policy.
- Hard trading gates can suppress policy gradients; parameter count and updates alone do not resolve this limitation.
- Recent baselines compute indicators in the evaluation frame, while the learned model receives causal prehistory; this warmup asymmetry remains.
- Selection uses one common validation year, not multiple expanding-window refit folds.
- Historical and recent periods start separate fresh-cash accounts; their returns are not one continuous live account.
- Costs, stops and the 18% breaker match across cells. Results do not activate any production strategy.
