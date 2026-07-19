# Asymmetric Momentum Parameters Design

## Goal

Allow Momentum entries and exits to use independently optimized ROC lookbacks, momentum
lookbacks, and thresholds while keeping the grid-search command-line interface compact.

## Strategy Interface

`MomentumStrategy` will expose six explicit parameters:

- `buy_roc_period`, `buy_momentum_period`, and `buy_threshold`
- `sell_roc_period`, `sell_momentum_period`, and `sell_threshold`

All six parameters will have defaults in `src/strategies/constants.py`. Initially, both sides will
use the currently selected rank-one values: ROC period 16, momentum period 12, and threshold
0.055. The former ambiguous constructor names (`roc_period`, `momentum_period`, and
`threshold`) will be replaced throughout the repository by the explicit buy names.

The strategy will calculate independent buy and sell indicator series. Buy signals fire when the
buy ROC crosses above `buy_threshold` while buy momentum is positive. Sell signals fire when the
sell ROC crosses below negative `sell_threshold` while sell momentum is negative. The resulting
signals remain event-based and continue to use the shared position-forward-fill helper.

Generated columns will be named `buy_roc`, `buy_momentum`, `buy_momentum_norm`, `sell_roc`,
`sell_momentum`, and `sell_momentum_norm` so report consumers can distinguish the two sides.

## Grid Search

An exhaustive six-dimensional Cartesian product would require 23,040,000 backtests with the
current 4,800-combination grid. The script will instead use a deterministic two-stage search:

1. Evaluate every shared-grid combination as the buy parameter triple while holding the sell
   triple at its configured defaults.
2. Rank by training Total Return using the existing deterministic tie breakers and retain the five
   best buy triples.
3. For each retained buy triple, evaluate every shared-grid combination as the sell triple.
4. Rank the combined stage-two results by training Total Return and validate only the final winner
   on the chronological validation partition.

This evaluates 28,800 combinations with the default ranges. The existing `--roc-periods`,
`--momentum-periods`, and `--thresholds` options remain unchanged and supply the shared candidate
ranges for both stages. No buy-specific or sell-specific CLI range options will be added. The top
buy candidate count is a module constant rather than a new CLI option.

The output CSV will contain all six parameter columns. It will contain the ranked stage-two rows;
validation metrics will appear only on the winning row, as in the current output. Console output
will report all six winning values and the total number of evaluations.

## Validation and Errors

The existing input validation, chronological 70/30 default split, drawdown-breaker control, quiet
backtest logging, and Total Return objective remain unchanged. The maximum lookback check will
consider the shared candidate ranges and both configured default triples.

Tests will cover independent buy/sell signal timing, six-parameter strategy defaults and explicit
overrides, stage-one candidate retention, the exact stage-two Cartesian product, deterministic
six-parameter ranking, winner validation, CSV columns, and unchanged CLI parsing.

## Documentation

Update the README and canonical strategy tutorial to describe asymmetric signals, the six strategy
parameters, the two-stage search, its 28,800 default evaluations, and the continuing risk of
training overfit.
