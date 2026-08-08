# BTC Momentum Runtime Profiles Design

## Goal

Activate explicit BTC Momentum parameter profiles for 5-minute and 1-hour backtests while keeping
all other coins and intervals on the general strategy defaults.

## Profiles

Add these entries to `MOMENTUM_PROFILES`:

- `("btc", "5m")`: buy ROC period 48, buy momentum period 48, buy threshold 0.055,
  sell ROC period 144, sell momentum period 12, and sell threshold 0.055.
- `("btc", "1h")`: buy ROC period 16, buy momentum period 12, buy threshold 0.055,
  sell ROC period 16, sell momentum period 12, and sell threshold 0.055.

The BTC/5m entry is an experimental override explicitly requested by the user. Its training return
ranked first in the robust search, but it failed the stability activity requirement because the
middle stability slice contained no completed round trip. Code comments and documentation must not
describe it as validated or production-optimal.

The BTC/1h entry makes the previously selected general defaults explicit for that runtime context.
It does not change the numerical behavior of BTC/1h backtests.

## Resolution and Fallback

The existing normalized `(coin, interval)` resolver remains unchanged. BTC 5-minute and BTC 1-hour
backtests receive their matching six parameters. ETH, SOL, and every unlisted interval receive an
empty profile mapping and therefore continue to use `MomentumStrategy` defaults.

## Testing and Documentation

Behavioral tests will instantiate Momentum through `create_strategy()` and assert that BTC/5m and
BTC/1h receive their exact profiles while ETH/5m falls back to the general defaults. README and the
canonical tutorial will identify BTC/5m as experimental and explain that it was activated by an
explicit override of the stability guard.
