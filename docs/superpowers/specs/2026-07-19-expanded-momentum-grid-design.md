# Expanded Momentum Grid Defaults Design

## Objective

Broaden and refine the default Momentum hyperparameter grid while keeping the existing
chronological training/validation algorithm and user-supplied CLI overrides unchanged.

## Default Grid

Use these deterministic defaults in `scripts/grid_search_momentum.py`:

```python
DEFAULT_ROC_PERIODS = list(range(2, 31, 2))
DEFAULT_MOMENTUM_PERIODS = list(range(2, 41, 2))
DEFAULT_THRESHOLDS = [value / 1000 for value in range(5, 81, 5)]
```

The ranges contain 15 ROC periods from 2 through 30, 20 momentum periods from 2 through 40,
and 16 thresholds from 0.005 through 0.080. Their Cartesian product contains exactly 4,800
training combinations.

## Compatibility and Documentation

Explicit `--roc-periods`, `--momentum-periods`, and `--thresholds` values continue to replace
the defaults. Ranking, chronological splitting, validation, engine configuration, CSV schema,
and risk controls do not change.

Update regression tests and the canonical tutorial command, ranges, and combination count.
Document that the larger grid increases runtime and overfitting risk. All new documentation
outside the bilingual README remains English.
