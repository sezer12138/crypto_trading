# Rank-One Momentum Defaults Design

## Objective

Set the built-in Momentum defaults to rank 1 from `results/momentum_grid_search.csv`:
ROC period 16, momentum period 12, and threshold 0.055.

## Changes

Update the three shared Momentum constants, constructor documentation and example, factory
regression test, canonical tutorial, and README. Explicit parameters must continue to override
the defaults. Do not change signal logic, grid ranges, backtest behavior, or risk controls.

## Performance Disclosure

Document the measured rank-one results: 121.50% training return, 2.14 training Sharpe,
-16.98% training drawdown, -18.16% validation return, and -2.02 validation Sharpe. Describe
the setting as training-ranked and overfit rather than as an out-of-sample improvement.
