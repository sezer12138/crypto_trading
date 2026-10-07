"""Seeded development-only searches for the twelve non-Momentum strategies."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial

from backtest import BACKTEST_MODEL_VERSION, BacktestEngine
from optimization.momentum_evaluator import _suppress_backtest_logs
from strategies import TradingStrategy, get_strategy

STRATEGIES = (
    "ma_cross",
    "rsi",
    "bollinger",
    "multi_factor",
    "mean_reversion",
    "macd",
    "breakout",
    "vwap",
    "atr_stop",
    "stochastic",
    "grid",
    "martingale",
)
PERIOD_CHOICES = (6, 12, 24, 48, 96, 168, 336, 672, 1008)


@dataclass(frozen=True)
class SearchConfig:
    """Immutable cost, capital, and risk settings shared by training and validation."""

    initial_capital: float = 10000.0
    coin: str = "btc"
    interval: str = "1h"
    commission_rate: float = 0.001
    slippage: float = 0.001
    position_size: float = 0.95
    min_holding_bars: int = 5
    max_trades_per_day: int = 6
    stop_loss_pct: float = 0.05
    drawdown_breaker_enabled: bool = False
    loss_cooldown_enabled: bool = False

    def identity(self) -> dict[str, Any]:
        """Return exact settings required to reuse an exported parameter profile."""
        return {"execution_model": BACKTEST_MODEL_VERSION, **asdict(self)}


def suggest_parameters(
    trial: Trial, name: str, max_period: int = 1008, engine_stop_loss_pct: float = 0.05
) -> dict[str, Any]:
    """Suggest valid parameters; every sampled control affects an executable strategy."""
    if name not in STRATEGIES:
        raise ValueError(f"Unsupported optimization strategy: {name}")
    periods = [p for p in PERIOD_CHOICES if p <= max_period]
    if len(periods) < 2:
        raise ValueError("Search partitions must support at least a 12-bar lookback")

    def period(key: str) -> int:
        return trial.suggest_categorical(key, periods)

    def ordered(fast_key: str, slow_key: str) -> dict[str, Any]:
        fast = trial.suggest_categorical(fast_key, [p for p in periods if p < max_period])
        ratio = trial.suggest_categorical("slow_ratio", [2, 3, 4, 6, 8])
        return {fast_key: fast, slow_key: min(max_period, fast * ratio)}

    params: dict[str, Any] = {}
    if name in ("ma_cross", "multi_factor"):
        keys = ("short_window", "long_window") if name == "ma_cross" else ("ma_short", "ma_long")
        params.update(ordered(*keys))
    if name == "macd":
        params.update(ordered("fast", "slow"))
        params["signal"] = trial.suggest_categorical("signal", [3, 6, 9, 18, 36])
    if name in ("rsi", "bollinger", "mean_reversion", "stochastic", "vwap"):
        params.update(
            trend_filter_enabled=True,
            trend_filter_window=period("trend_filter_window"),
            trend_filter_tolerance=trial.suggest_float(
                "trend_filter_tolerance", 0.005, 0.05, log=True
            ),
        )
    if name == "rsi":
        params.update(
            period=period("period"),
            oversold=trial.suggest_int("oversold", 15, 40, step=5),
            overbought=trial.suggest_int("overbought", 55, 85, step=5),
        )
    elif name == "bollinger":
        params.update(
            window=period("window"), num_std=trial.suggest_float("num_std", 1.5, 3.5, step=0.25)
        )
    elif name == "mean_reversion":
        params.update(
            window=period("window"),
            entry_z=trial.suggest_float("entry_z", 1.5, 3.5, step=0.25),
            exit_z=trial.suggest_float("exit_z", 0.1, 1, step=0.1),
        )
    elif name == "multi_factor":
        params.update(
            rsi_period=period("rsi_period"),
            volume_threshold=trial.suggest_float("volume_threshold", 1.1, 3, step=0.1),
            buy_threshold=trial.suggest_float("buy_threshold", 0.25, 0.65, step=0.05),
            sell_threshold=trial.suggest_float("sell_threshold", 0.25, 0.65, step=0.05),
        )
    elif name == "breakout":
        params.update(window=period("window"), confirmation=True)
    elif name == "vwap":
        params.update(
            window=period("window"),
            atr_window=period("atr_window"),
            deviation_multiplier=trial.suggest_float("deviation_multiplier", 1, 4, step=0.25),
            min_deviation=trial.suggest_float("min_deviation", 0.003, 0.03, log=True),
        )
    elif name == "atr_stop":
        params.update(
            atr_period=trial.suggest_categorical("atr_period", [7, 14, 28, 56]),
            trend_ma=period("trend_ma"),
            multiplier=trial.suggest_float("multiplier", 1.5, 6, step=0.5),
        )
    elif name == "stochastic":
        params.update(
            k_period=period("k_period"),
            d_period=trial.suggest_categorical("d_period", [3, 6, 9, 18]),
            smooth=trial.suggest_categorical("smooth", [1, 3, 5]),
            oversold=trial.suggest_int("oversold", 10, 35, step=5),
            overbought=trial.suggest_int("overbought", 60, 90, step=5),
        )
    elif name == "grid":
        params.update(
            calibration_bars=period("calibration_bars"),
            grid_num=trial.suggest_int("grid_num", 5, 20),
            range_margin=trial.suggest_float("range_margin", 0.1, 1, step=0.1),
            inventory_fraction=trial.suggest_float("inventory_fraction", 0.1, 0.7, step=0.1),
        )
    elif name == "martingale":
        params.update(
            initial_fraction=trial.suggest_float("initial_fraction", 0.001, 0.02, log=True),
            multiplier=trial.suggest_float("multiplier", 1.25, 2, step=0.25),
            max_steps=trial.suggest_int("max_steps", 1, 4),
            target_profit=trial.suggest_float("target_profit", 0.01, 0.06, step=0.005),
            stop_loss=trial.suggest_float(
                "stop_loss", 0.2 * engine_stop_loss_pct, 0.8 * engine_stop_loss_pct
            ),
        )
    return params


def build_strategy(
    name: str, parameters: dict[str, Any], df: pd.DataFrame, capital: float
) -> TradingStrategy:
    """Resolve special strategy quantity/range from past calibration data only."""
    params = dict(parameters)
    if name == "grid":
        calibration = min(int(params.pop("calibration_bars", 100)), len(df))
        history = df.iloc[:calibration]
        low, high = float(history.low.min()), float(history.high.max())
        width = max(high - low, float(history.close.iloc[-1]) * 0.001)
        margin = width * float(params.pop("range_margin", 0.1))
        fraction = params.pop("inventory_fraction", None)
        levels = int(params.get("grid_num", 10))
        if fraction is not None:
            params["amount_per_grid"] = capital * fraction / float(history.close.iloc[-1]) / levels
        params.update(
            lower_price=max(low - margin, low * 0.1),
            upper_price=high + margin,
            warmup_bars=calibration,
        )
    elif name == "martingale" and "initial_fraction" in params:
        params["base_amount"] = capital * params.pop("initial_fraction") / float(df.close.iloc[0])
    return get_strategy(name, **params)


def evaluate_partition(
    df: pd.DataFrame, name: str, parameters: dict[str, Any], config: SearchConfig
) -> dict[str, Any]:
    """Evaluate one fresh, independent chronological partition with identical costs."""
    settings = asdict(config)
    settings.pop("coin")
    settings.pop("interval")
    with _suppress_backtest_logs():
        result = BacktestEngine(**settings).run_backtest(
            df, build_strategy(name, parameters, df, config.initial_capital), config.coin.upper()
        )
    return result.metrics


def training_score(metrics: list[dict[str, Any]]) -> float:
    """Reward typical net return while penalizing instability, drawdown, and inactivity."""
    returns = np.array([m["annual_return_pct"] for m in metrics], dtype=float)
    drawdown = max(abs(m["max_drawdown_pct"]) for m in metrics)
    missing = sum(max(0, 2 - m["total_round_trips"]) for m in metrics)
    return float(
        np.median(returns) - 0.5 * np.std(returns) - 2 * max(0, drawdown - 25) - 25 * missing
    )


def adoption_checks(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    training_returns: list[float],
    training_round_trips: list[int],
) -> dict[str, bool]:
    """Guard deployment without using validation to rank or retry training candidates."""
    checks = {
        "positive_holdout": candidate["total_return_pct"] > 0,
        "beats_default_holdout": candidate["total_return_pct"] > baseline["total_return_pct"],
        "holdout_drawdown_at_most_30": abs(candidate["max_drawdown_pct"]) <= 30,
        "holdout_at_least_two_positions": candidate["total_round_trips"] >= 2,
        "positive_typical_training_return": float(np.median(training_returns)) > 0,
        "each_training_fold_at_least_two_positions": min(training_round_trips) >= 2,
    }

    return {key: bool(value) for key, value in checks.items()}


def optimize_strategy(
    name: str,
    folds: list[pd.DataFrame],
    holdout: pd.DataFrame,
    config: SearchConfig,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Select using development data only, then evaluate one frozen candidate on holdout."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(15, trials)),
    )
    max_period = min(1008, min(len(fold) for fold in folds) // 3)
    baseline_training = [evaluate_partition(f, name, {}, config) for f in folds]

    def objective(trial: Trial) -> float:
        parameters = (
            {}
            if trial.number == 0
            else suggest_parameters(trial, name, max_period, config.stop_loss_pct)
        )
        metrics = (
            baseline_training
            if trial.number == 0
            else [evaluate_partition(f, name, parameters, config) for f in folds]
        )
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("fold_metrics", metrics)
        return training_score(metrics)

    study.optimize(objective, n_trials=trials)
    selected = study.best_trial
    parameters = selected.user_attrs["parameters"]
    training = selected.user_attrs["fold_metrics"]
    # No holdout data is accessed by the objective or the training selection above.
    baseline_holdout = evaluate_partition(holdout, name, {}, config)
    candidate_holdout = evaluate_partition(holdout, name, parameters, config)
    checks = adoption_checks(
        candidate_holdout,
        baseline_holdout,
        [m["annual_return_pct"] for m in training],
        [m["total_round_trips"] for m in training],
    )
    return {
        "strategy": name,
        "selected_trial": selected.number,
        "parameters": parameters,
        "training_score": selected.value,
        "baseline_training_score": training_score(baseline_training),
        "training_metrics": training,
        "baseline_holdout": baseline_holdout,
        "candidate_holdout": candidate_holdout,
        "adoption_checks": checks,
        "adopted": all(checks.values()),
        "trials": [
            {
                "number": t.number,
                "search_parameters": t.params,
                "parameters": t.user_attrs["parameters"],
                "training_score": t.value,
                "fold_metrics": t.user_attrs["fold_metrics"],
            }
            for t in study.trials
        ],
    }
