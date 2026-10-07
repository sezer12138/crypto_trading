"""Validated strategy defaults selected by coin and candle interval."""

from .constants import BTC_1H_RSI_PARAMETERS, BTC_1H_VWAP_PARAMETERS

STRATEGY_PROFILES: dict[tuple[str, str, str], dict[str, int | float | bool]] = {
    ("rsi", "btc", "1h"): BTC_1H_RSI_PARAMETERS,
    ("vwap", "btc", "1h"): BTC_1H_VWAP_PARAMETERS,
}


def get_strategy_profile(
    strategy_name: str, coin: str, interval: str
) -> dict[str, int | float | bool]:
    """Return a copy of matching defaults, leaving generic constructors unchanged."""
    key = (strategy_name.strip().lower(), coin.strip().lower(), interval.strip().lower())
    return STRATEGY_PROFILES.get(key, {}).copy()
