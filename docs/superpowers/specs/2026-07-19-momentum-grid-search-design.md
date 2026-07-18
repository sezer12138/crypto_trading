# Momentum Grid Search Design

## Objective

Add a reproducible command-line grid search for `MomentumStrategy` hyperparameters. Rank
parameter combinations by training-period `total_return_pct` and report the selected
combination's performance on a later, untouched validation period.

## Command-Line Interface

Create `scripts/grid_search_momentum.py` with this primary interface:

```bash
python scripts/grid_search_momentum.py \
  --data data/historical/btc_1h_730d.csv \
  --roc-periods 5,10,15,20,30 \
  --momentum-periods 5,10,14,20,30 \
  --thresholds 0.005,0.01,0.015,0.02,0.03,0.04 \
  --train-ratio 0.7 \
  --output results/momentum_grid_search.csv
```

The displayed values are the defaults, producing 150 combinations. The script also accepts
`--capital` and `--disable-drawdown-breaker`, matching the existing backtest semantics.

## Data and Validation

Read a timestamp-indexed OHLCV CSV and require `open`, `high`, `low`, `close`, and `volume`.
Sort rows chronologically and reject duplicate timestamps, invalid numeric parameters,
train ratios outside `(0, 1)`, or splits too short to initialize the largest requested
lookback.

Split chronologically without shuffling. Use the first 70% by default as training data and
the remaining 30% as validation data. Run every grid combination only on training data.
Choose the winner by highest training `total_return_pct`, then higher training Sharpe ratio,
then lower maximum drawdown magnitude, followed by ascending parameter values for deterministic
ties. Run only the selected combination on validation data.

Each train and validation backtest starts with the requested initial capital and a fresh
`BacktestEngine`. The maximum-drawdown breaker remains enabled unless explicitly disabled.

## Output

Write one CSV row per training combination, sorted by the deterministic ranking. Include:

- rank and the three hyperparameters;
- training total return, annual return, Sharpe ratio, maximum drawdown, win rate, and trades;
- validation metrics only on the winning row, leaving those fields empty for other rows.

Print the search size, split sizes and timestamp ranges, winning parameters, training return,
and validation return. Create the output parent directory when necessary. Exit nonzero with a
clear English error for invalid input or failed output.

## Structure

Keep reusable, testable functions in the script for comma-separated parsing, chronological
splitting, combination evaluation, deterministic ranking, and CSV writing. Reuse
`MomentumStrategy` and `BacktestEngine`; do not duplicate strategy or execution logic.

## Testing

Add focused tests covering:

- integer and floating-point list parsing and validation;
- Cartesian-product evaluation and fresh engine use;
- chronological 70/30 splitting and insufficient-history errors;
- total-return-first deterministic ranking;
- winner-only validation and CSV schema/order;
- invalid/missing OHLCV data;
- default-enabled and explicitly disabled drawdown-breaker propagation.
