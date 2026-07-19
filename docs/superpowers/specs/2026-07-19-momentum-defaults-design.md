# Momentum Default Parameters Design

## Objective

Change the built-in `MomentumStrategy` defaults to the grid search's highest-training-return
setting: ROC period 5, momentum period 10, and ROC threshold 0.04.

## Implementation

Define these strategy defaults in `src/strategies/constants.py`:

```python
DEFAULT_MOMENTUM_ROC_PERIOD = 5
DEFAULT_MOMENTUM_PERIOD = 10
DEFAULT_MOMENTUM_THRESHOLD = 0.04
```

Import and use the constants as the `MomentumStrategy` constructor defaults. Explicit caller
arguments remain unchanged and continue to override the defaults.

Update the strategy module example, canonical tutorial, and README parameter summary. Add a
factory-level regression test asserting that `get_strategy("momentum")` uses the three new
defaults while explicitly supplied values still work.

## Performance Disclosure

Documentation must identify the values as training-optimized rather than independently
validated. On the existing chronological split, they returned 37.43% on training data and
-17.67% on validation data. This warning prevents the new defaults from being represented as a
demonstrated out-of-sample improvement.

## Scope

Do not change the grid-search ranges, Momentum signal calculations, backtest engine, or risk
controls. All new comments and documentation outside the bilingual README remain English.
