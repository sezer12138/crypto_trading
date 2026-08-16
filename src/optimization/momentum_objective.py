"""Pure search-space and scoring helpers for Bayesian Momentum optimization."""

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from optuna.trial import Trial

ROC_MINUTES_MIN = 60
ROC_MINUTES_MAX = 28 * 24 * 60
MOMENTUM_MINUTES_MIN = 30
MOMENTUM_MINUTES_MAX = 14 * 24 * 60
THRESHOLD_MIN = 0.005
THRESHOLD_MAX = 0.25


@dataclass(frozen=True)
class LossConfig:
    """Weights and requirements for the composite trial loss."""

    drawdown_target_pct: float = 25.0
    drawdown_weight: float = 2.0
    stability_weight: float = 0.5
    minimum_round_trips: int = 2
    missing_round_trip_penalty: float = 25.0


@dataclass(frozen=True)
class SearchBounds:
    """Resolved inclusive parameter bounds for one candle interval."""

    roc_min: int
    roc_max: int
    momentum_min: int
    momentum_max: int
    threshold_min: float
    threshold_max: float


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics collected from one chronological optimization fold."""

    annual_return_pct: float
    max_drawdown_pct: float
    total_trades: int


@dataclass(frozen=True)
class TrialEvaluation:
    """Immutable aggregate scoring result for one parameter trial."""

    loss: float
    robust_annual_return_pct: float
    worst_drawdown_pct: float
    instability: float
    missing_round_trips: int
    folds: tuple[FoldMetrics, ...]


def resolve_search_bounds(interval_minutes: int) -> SearchBounds:
    """Convert fixed duration limits into valid raw-bar bounds."""
    if interval_minutes <= 0:
        raise ValueError("Candle interval must be positive")
    return SearchBounds(
        roc_min=max(1, round(ROC_MINUTES_MIN / interval_minutes)),
        roc_max=max(1, round(ROC_MINUTES_MAX / interval_minutes)),
        momentum_min=max(1, round(MOMENTUM_MINUTES_MIN / interval_minutes)),
        momentum_max=max(1, round(MOMENTUM_MINUTES_MAX / interval_minutes)),
        threshold_min=THRESHOLD_MIN,
        threshold_max=THRESHOLD_MAX,
    )


def suggest_momentum_parameters(trial: Trial, bounds: SearchBounds) -> dict[str, int | float]:
    """Sample all independent buy and sell Momentum parameters logarithmically."""
    return {
        "buy_roc_period": trial.suggest_int(
            "buy_roc_period", bounds.roc_min, bounds.roc_max, log=True
        ),
        "buy_momentum_period": trial.suggest_int(
            "buy_momentum_period", bounds.momentum_min, bounds.momentum_max, log=True
        ),
        "buy_threshold": trial.suggest_float(
            "buy_threshold", bounds.threshold_min, bounds.threshold_max, log=True
        ),
        "sell_roc_period": trial.suggest_int(
            "sell_roc_period", bounds.roc_min, bounds.roc_max, log=True
        ),
        "sell_momentum_period": trial.suggest_int(
            "sell_momentum_period", bounds.momentum_min, bounds.momentum_max, log=True
        ),
        "sell_threshold": trial.suggest_float(
            "sell_threshold", bounds.threshold_min, bounds.threshold_max, log=True
        ),
    }


def score_folds(folds: Sequence[FoldMetrics], config: LossConfig) -> TrialEvaluation:
    """Compute the specified robust loss from independent fold metrics."""
    annual = [item.annual_return_pct for item in folds]
    if not folds or not all(
        math.isfinite(item.annual_return_pct) and math.isfinite(item.max_drawdown_pct)
        for item in folds
    ):
        raise ValueError("Fold metrics must be finite")
    robust = statistics.median(annual)
    worst = max(abs(item.max_drawdown_pct) for item in folds)
    instability = statistics.pstdev(annual)
    missing = sum(max(0, config.minimum_round_trips - item.total_trades // 2) for item in folds)
    loss = (
        -robust
        + config.drawdown_weight * max(0.0, worst - config.drawdown_target_pct)
        + config.stability_weight * instability
        + config.missing_round_trip_penalty * missing
    )
    return TrialEvaluation(loss, robust, worst, instability, missing, tuple(folds))


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return nondominated rows ordered by drawdown, return, and trial number."""
    front = [
        row
        for row in rows
        if not any(
            candidate is not row
            and candidate["worst_drawdown"] <= row["worst_drawdown"]
            and candidate["robust_return"] >= row["robust_return"]
            and (
                candidate["worst_drawdown"] < row["worst_drawdown"]
                or candidate["robust_return"] > row["robust_return"]
            )
            for candidate in rows
        )
    ]
    return sorted(
        front,
        key=lambda row: (row["worst_drawdown"], -row["robust_return"], row["trial"]),
    )
