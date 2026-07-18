# Drawdown Circuit Breaker Flag Design

## Objective

Allow users to disable the backtest engine's maximum-drawdown circuit breaker while keeping
the existing enabled behavior as the default.

## Public Interface

- Add `drawdown_breaker_enabled: bool = True` to `BacktestEngine.__init__`.
- Add `--disable-drawdown-breaker` to `run_backtest.py`.
- Pass the CLI choice to every `BacktestEngine` created for single-strategy and comparison runs.

The explicit Boolean separates whether the control is active from `max_drawdown_pct`, which
continues to represent only the configured threshold.

## Behavior

When enabled, behavior is unchanged: reaching the maximum drawdown forces liquidation and
halts trading, subject to the existing breaker cooldown configuration.

When disabled, the engine skips breaker liquidation and shutdown. Other risk controls,
including per-position stop-losses and consecutive-loss cooldowns, remain active. The equity
curve and maximum-drawdown performance metric continue to be calculated normally.

## Compatibility and Logging

The flag defaults to enabled, so existing Python callers and CLI commands retain their current
risk behavior. Engine startup logs report whether the drawdown breaker is enabled and show its
threshold only when relevant.

## Validation

Tests will demonstrate that:

- default and explicitly enabled engines still halt at the drawdown threshold;
- a disabled engine does not force-sell or stop at that threshold;
- disabling the breaker does not disable other risk controls;
- the CLI flag parses correctly and is propagated to both single and comparison runs;
- engine reset and reuse preserve the configured enablement choice.
