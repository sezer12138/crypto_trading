# TCN capacity, training-budget and history study

## Intent and authorization

The user approved the previously proposed controlled comparison of a longer-trained hidden-32 model and a hidden-128 model, and requested substantially longer BTC hourly history. Existing next-open execution, costs, allocation gates and risk rules remain fixed. This is retrospective model research, not deployment.

## Approaches

Simply increasing width and epochs confounds optimization, capacity and history. Replacing the architecture would add another confound. Use a predeclared factorial study of short/long training histories and hidden widths 32/128, plus short-budget controls for both widths on the short history. The user selected controlled experiments in the conversation; exact dates/budgets below are implementation decisions.

## Data and evaluation protocol

Acquire genuine Binance BTCUSDT spot 1h OHLCV from 2018-01-01 through 2026-10-24 00:00 UTC exclusive where available. Verify archive SHA256 checksums, record every source and the output fingerprint, reject conflicts and invalid candles. Do not invent missing candles or join exchanges. If actual exchange outages cannot be repaired from the official REST endpoint, keep genuine gaps explicit. Opt-in training-only gap support resets features at each continuous block and excludes warmup, labels, windows and policy segments crossing any gap. Validation/evaluation and their preceding warmup must remain continuous. The default standalone trainer remains strict.

Freeze absolute dates across all models: training ends at 2025-01-01, selection validation is 2025-01-01 through 2026-01-01, historical evaluation is 2026-01-01 through the end of the original observed snapshot (2026-10-07 04:00 UTC), and the newly acquired later tail is evaluated only after every configuration and checkpoint is frozen. The recent tail is short and retrospective, not a prospective profitability claim. Short-history training starts at 2023-01-01; long-history starts at the acquired continuous range start. No ratio splits may move shared validation or evaluation dates.

Comparisons report all seeds, return, MDD, costs, fill count and rejected/all-cash candidates. Training-period rolling development evaluations may be used diagnostically, but the shared selection year remains the only ranking source in this first bounded study. The old 2026 observations are explicitly not blind data. Final tail cannot select a width, seed or budget. No repeated final-tail tuning.

## Optimization and diagnostics

Preserve v1 defaults. Add configurable sampling stride, epoch limits, minimum training epochs, validation early stopping, plateau LR decay, restoration of the best pretraining quantile checkpoint, and actual update counters. Extended runs have pretraining cap 50 epochs, policy cap 30 epochs; pretraining/policy minimum 10 epochs, early-stop patience 8, LR patience 3 with factor 0.5 and floor 1e-6. The predeclared common pretraining stride is 12 and batch size is 32. All study cells use MPS pretraining and two CPU threads for policy simulation, following runtime-only probes; this includes both short-budget controls. All seeds are 7, 19 and 42, with 7 predeclared primary. Use common settings across the main factorial cells; compare equal epoch coverage and disclose differing total updates with expanded history. Hidden-32/128 parameter counts remain 27,978/419,082.

Record head-specific gradient norms (including zeros when order gates suppress policy gradients), raw decision opportunity/actual-fill and exposure summaries where feasible without altering paths, optimizer LR, train/validation losses, weighted loss components, epoch time and total updates. Logs must be flushed/closed on failure. Scalars must never require observing holdout outcomes during training.

## Deliverables and boundaries

Reproducible data fetch and study commands, immutable protocol manifest written before outcomes, frozen checkpoints for every run, TensorBoard curves, aggregate JSON/HTML/Markdown report with limitations and data audit, meaningful regression tests. Preserve the original experiment artifacts and checkpoint compatibility. Repository prose, docstrings and logs remain English. Do not push or deploy experiment changes automatically.
