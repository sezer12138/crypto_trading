"""Opt-in validated profiles tied to a precise execution and risk identity."""

import json
from pathlib import Path
from typing import Any

from optimization.strategy_search import STRATEGIES, SearchConfig


def read_profile(path: Path, name: str, config: SearchConfig) -> dict[str, Any]:
    """Reject incompatible profiles before returning an independent parameter mapping."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get("identity") != config.identity():
        raise ValueError(
            "Strategy profile identity does not match coin, interval, execution, capital, costs, or risk settings"
        )
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or any(key not in STRATEGIES for key in profiles):
        raise ValueError("Profile contains unsupported strategies")
    return dict(profiles.get(name, {}))
