"""Coin-and-interval-specific Momentum parameter profiles."""

from typing import Dict, Tuple, TypedDict


class MomentumProfile(TypedDict, total=False):
    """Six explicit Momentum parameters for one coin and candle interval."""

    buy_roc_period: int
    buy_momentum_period: int
    buy_threshold: float
    sell_roc_period: int
    sell_momentum_period: int
    sell_threshold: float


MOMENTUM_PROFILES: Dict[Tuple[str, str], MomentumProfile] = {
    # Explicit experimental override: ranked first on BTC/5m training data but did not
    # complete a round trip in every stability slice.
    ("btc", "5m"): {
        "buy_roc_period": 48,
        "buy_momentum_period": 48,
        "buy_threshold": 0.055,
        "sell_roc_period": 144,
        "sell_momentum_period": 12,
        "sell_threshold": 0.055,
    },
    # Previous BTC/1h defaults made explicit for interval-aware runtime resolution.
    ("btc", "1h"): {
        "buy_roc_period": 16,
        "buy_momentum_period": 12,
        "buy_threshold": 0.055,
        "sell_roc_period": 16,
        "sell_momentum_period": 12,
        "sell_threshold": 0.055,
    },
}


def get_momentum_profile(coin: str, interval: str) -> MomentumProfile:
    """Return an independent copy of a matching Momentum profile or an empty mapping."""
    key = (coin.strip().lower(), interval.strip().lower())
    return MOMENTUM_PROFILES.get(key, {}).copy()
