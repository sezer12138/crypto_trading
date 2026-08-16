"""Tests for the Bayesian Momentum optimization objective helpers."""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.momentum_objective import (
    FoldMetrics,
    LossConfig,
    SearchBounds,
    TrialEvaluation,
    pareto_front,
    resolve_search_bounds,
    score_folds,
    suggest_momentum_parameters,
)


class RecordingTrial:
    """Record parameter suggestions while returning each lower bound."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | float, int | float, bool]] = []

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Record one integer suggestion."""
        self.calls.append(("int", name, low, high, log))
        return low

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Record one floating-point suggestion."""
        self.calls.append(("float", name, low, high, log))
        return low


def test_5m_bounds_extend_the_previous_boundary() -> None:
    """Converts duration limits to the required five-minute bar counts."""
    bounds = resolve_search_bounds(5)

    assert (bounds.roc_min, bounds.roc_max) == (12, 8064)
    assert (bounds.momentum_min, bounds.momentum_max) == (6, 4032)
    assert (bounds.threshold_min, bounds.threshold_max) == (0.005, 0.25)


def test_score_folds_combines_all_penalties() -> None:
    """Uses median return, drawdown, instability, and activity penalties."""
    folds = [
        FoldMetrics(20.0, -20.0, 4),
        FoldMetrics(30.0, -30.0, 2),
        FoldMetrics(10.0, -15.0, 0),
        FoldMetrics(40.0, -10.0, 6),
    ]

    result = score_folds(folds, LossConfig())

    assert result.robust_annual_return_pct == 25.0
    assert result.worst_drawdown_pct == 30.0
    assert result.missing_round_trips == 3
    assert result.loss == pytest.approx(65.5901699)


def test_score_folds_rejects_non_finite_annual_return() -> None:
    """Rejects a fold whose annual return cannot produce a valid loss."""
    folds = [FoldMetrics(float("nan"), -20.0, 4)]

    with pytest.raises(ValueError, match="Fold metrics must be finite"):
        score_folds(folds, LossConfig())


def test_score_folds_rejects_non_finite_drawdown() -> None:
    """Rejects a fold whose drawdown cannot produce a valid loss."""
    folds = [FoldMetrics(20.0, float("inf"), 4)]

    with pytest.raises(ValueError, match="Fold metrics must be finite"):
        score_folds(folds, LossConfig())


def test_suggestions_use_independent_logarithmic_buy_and_sell_parameters() -> None:
    """Suggests all six Momentum inputs independently with logarithmic sampling."""
    bounds = SearchBounds(12, 8064, 6, 4032, 0.005, 0.25)
    trial = RecordingTrial()

    parameters = suggest_momentum_parameters(trial, bounds)  # type: ignore[arg-type]

    assert parameters == {
        "buy_roc_period": 12,
        "buy_momentum_period": 6,
        "buy_threshold": 0.005,
        "sell_roc_period": 12,
        "sell_momentum_period": 6,
        "sell_threshold": 0.005,
    }
    assert trial.calls == [
        ("int", "buy_roc_period", 12, 8064, True),
        ("int", "buy_momentum_period", 6, 4032, True),
        ("float", "buy_threshold", 0.005, 0.25, True),
        ("int", "sell_roc_period", 12, 8064, True),
        ("int", "sell_momentum_period", 6, 4032, True),
        ("float", "sell_threshold", 0.005, 0.25, True),
    ]


def test_pareto_front_excludes_dominated_rows() -> None:
    """Keeps nondominated rows in deterministic presentation order."""
    rows = [
        {"trial": 1, "robust_return": 20.0, "worst_drawdown": 20.0},
        {"trial": 2, "robust_return": 25.0, "worst_drawdown": 20.0},
        {"trial": 3, "robust_return": 18.0, "worst_drawdown": 10.0},
    ]

    assert [row["trial"] for row in pareto_front(rows)] == [3, 2]


def test_domain_types_are_immutable() -> None:
    """Exposes immutable values suitable for worker-process boundaries."""
    config = LossConfig()
    evaluation = TrialEvaluation(0.0, 0.0, 0.0, 0.0, 0, ())

    with pytest.raises(FrozenInstanceError):
        config.drawdown_weight = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evaluation.loss = 1.0  # type: ignore[misc]
