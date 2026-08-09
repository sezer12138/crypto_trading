# Loss Cooldown CLI Flag Design

## Goal

Add a `--disable-loss-cooldown` command-line flag so users can disable the consecutive-loss
cooldown independently from the maximum-drawdown circuit breaker. The command must work for
single-strategy runs, `--compare`, and `--coin all`.

## Behavior

Consecutive-loss cooldown remains enabled by default, preserving existing behavior. When enabled,
three consecutive losing exits trigger the existing warning and prevent new buys for the configured
number of bars.

When `--disable-loss-cooldown` is supplied, the backtest engine does not count consecutive losing
exits, emit consecutive-loss cooldown warnings, or block new buys because of loss cooldown state.
Per-position stop-losses, maximum-drawdown behavior, daily trade limits, and all other risk controls
remain unchanged. The flag can be combined with `--disable-drawdown-breaker` to disable both
independent mechanisms.

## Implementation

Add a `loss_cooldown_enabled: bool = True` keyword parameter to `BacktestEngine`. Guard consecutive
loss tracking and cooldown enforcement with this setting. Log whether the mechanism is enabled at
engine initialization.

Add `--disable-loss-cooldown` to `run_backtest.py`. Convert the negative CLI flag into a positive
`loss_cooldown_enabled` value and propagate it through `main`, `compare_strategies`, and
`run_single_backtest` into every `BacktestEngine` instance.

## Compatibility

The new engine parameter is appended after existing parameters so positional callers retain their
current argument mapping. The default remains enabled, so existing Python and command-line callers
retain current risk behavior unless they explicitly pass the new flag.

## Testing

Tests will verify:

- the CLI flag is accepted and defaults to false;
- the disabled setting reaches single, comparison, and all-coin runs;
- comparison forwards the setting to all 13 strategy runs;
- the engine emits no cooldown warning and does not block buys when disabled;
- loss cooldown remains active by default;
- the complete test suite still passes.

## Documentation

Update the README and strategy/backtest tutorial with the new flag, its independence from the
drawdown breaker, and an example combining both flags.
