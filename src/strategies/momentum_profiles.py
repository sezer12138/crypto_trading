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


# Profiles are added only after passing the documented validation and activity guard.
MOMENTUM_PROFILES: Dict[Tuple[str, str], MomentumProfile] = {}


def get_momentum_profile(coin: str, interval: str) -> MomentumProfile:
    """Return an independent copy of a matching Momentum profile or an empty mapping."""
    key = (coin.strip().lower(), interval.strip().lower())
    return MOMENTUM_PROFILES.get(key, {}).copy()
