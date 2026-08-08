# Interval-Aware Momentum Profiles Design

## Goal

Use Momentum parameters appropriate to each coin and candle interval without promoting sparse,
overfit grid-search winners. Preserve Total Return as the optimization objective and keep the
existing shared ROC, momentum, and threshold CLI ranges.

## Evidence and Constraints

The BTC 360-day 5-minute backtest used the 1-hour-derived defaults over only 80 and 60 minutes.
The 5.5% ROC threshold generated two round trips, both closed by the 5% stop-loss, for a -9.86%
return. Lower thresholds produced hundreds of trades and much larger losses because a round trip
incurs commission and slippage on both sides.

A bounded 5-minute search improved the full-period result to -4.81%, but its winning training
configuration made one round trip and made no trades on validation. It is not suitable for a
runtime profile. A candidate must demonstrate both validation improvement and meaningful activity
before adoption.

## Runtime Profile Resolution

Create `src/strategies/momentum_profiles.py` with a `MomentumProfile` typed dictionary, a mapping
keyed by normalized `(coin, interval)`, and `get_momentum_profile(coin, interval)`. The function
returns a copy of the six explicit `buy_*` and `sell_*` parameters when a profile exists, otherwise
an empty dictionary.

`run_single_backtest()` will call this resolver only for the Momentum strategy and pass the returned
keywords to `get_strategy()`. An absent profile preserves `MomentumStrategy`'s general defaults.
Other strategies and direct factory callers remain unchanged.

Profiles are coin-and-interval-specific because parameters learned on BTC 5-minute data must not be
silently applied to ETH, SOL, or another candle interval. The initial implementation will add a
BTC 5-minute profile only if the robust search described below passes its adoption guard.

## Interval-Aware Candidate Ranges

The grid-search script will infer candle duration from the median positive timestamp difference.
When the user explicitly supplies `--roc-periods`, `--momentum-periods`, or `--thresholds`, those
values continue to mean raw candle counts and take precedence.

When ranges are omitted:

- Hourly and longer data retain the existing default candle-count grids.
- Sub-hourly data converts fixed duration candidates into candle counts. ROC durations are
  2, 4, 8, 12, 16, 24, and 36 hours. Momentum durations are 1, 2, 4, 6, 12, and 24 hours.
- Sub-hourly thresholds are 0.015, 0.025, 0.035, 0.045, and 0.055.
- Converted periods are rounded to the nearest candle and deduplicated while preserving order.

For 5-minute data this creates 210 shared-grid combinations and 1,260 two-stage evaluations rather
than the impractical 28,800 default evaluations over more than 100,000 candles.

## Robust Candidate Selection

The outer chronological split remains 70% training and 30% untouched validation. Candidate search
continues to use the existing two-stage method on the outer training partition:

1. Search every buy triple with the default sell triple.
2. Retain the five best buy triples by training Total Return and deterministic tie breakers.
3. Search every sell triple for each retained buy triple.
4. Retain the top 20 six-parameter candidates by training Total Return.

Each shortlisted candidate is then evaluated independently on three equal, non-overlapping,
chronological stability slices within the outer training partition. The candidate's stability
objective is compounded Total Return across the three slices. A candidate is eligible only when it
completes at least one round trip in every slice and at least three round trips in total. Eligible
candidates are ranked by compounded stability Total Return, then worst-slice return, Sharpe ratio,
maximum drawdown, and ascending parameter values.

Only the highest-ranked eligible candidate is evaluated on the untouched outer validation
partition. If no candidate satisfies the activity requirement, the script exits successfully,
writes diagnostic results, and explicitly reports that no deployable winner was found.

## Profile Adoption Guard

The robust winner will be compared with the currently resolved profile on the same outer validation
partition. It is eligible for a committed runtime profile only if:

- its validation Total Return is strictly higher than the current profile's validation return;
- it completes at least two validation round trips; and
- its validation maximum drawdown is no worse than 30%.

The grid-search script reports whether these conditions pass but does not modify source files. This
keeps optimization runs from silently changing production defaults. The verified parameters will
be added to the profile mapping as an intentional code change.

## Results and Errors

The CSV will contain the shortlisted candidates, their original training metrics, per-slice
returns and trade counts, aggregate stability metrics, eligibility status, and validation metrics
only for the selected winner and current-profile baseline. Existing OHLCV validation, quiet
backtest logging, drawdown-breaker control, and deterministic output remain.

Invalid or non-uniform timestamps will produce a clear error when interval inference is required.
Explicit period ranges bypass interval-based range generation but still require valid chronological
timestamps for splitting.

## Testing and Documentation

Tests will cover profile resolution and fallback, Momentum profile propagation from
`run_single_backtest()`, interval inference, 5-minute duration conversion, explicit CLI overrides,
shortlist construction, stability activity filtering, deterministic stability ranking, no-winner
handling, and the adoption guard. The README and canonical tutorial will explain why parameters are
coin-and-interval-specific and why a positive training return alone is insufficient.
