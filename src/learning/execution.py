"""Piecewise differentiable replay of the canonical single-asset execution rules."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from backtest import BacktestEngine
from learning.config import LearningConfig


@dataclass
class TensorOrder:
    """A close-observed coin quantity, executed at the next open."""

    signal: int
    quantity: torch.Tensor
    force: bool = False
    liquidate: bool = False


@dataclass
class PathResult:
    """Differentiable net equity and executed turnover along an ordered path."""

    equity: torch.Tensor
    turnover: torch.Tensor


def make_engine(
    config: LearningConfig, initial_capital: float = 10000.0, cost_multiplier: float = 1.0
) -> BacktestEngine:
    """Construct authoritative evaluation with identical risk and cost settings."""
    return BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=config.commission * cost_multiplier,
        slippage=config.slippage * cost_multiplier,
        position_size=config.max_weight,
        min_holding_bars=config.min_holding_bars,
        max_trades_per_day=config.max_daily_entries,
        stop_loss_pct=config.stop_loss,
        max_drawdown_pct=config.breaker,
        loss_cooldown_enabled=False,
        breaker_cooldown_bars=0,
        execution_mode="next_open",
    )


def policy_order(
    model: object,
    encoded: torch.Tensor,
    quantiles: torch.Tensor,
    cash: torch.Tensor,
    quantity: torch.Tensor,
    close: float,
    peak: torch.Tensor,
    volatility: float,
    index: int,
    config: LearningConfig,
) -> Optional[TensorOrder]:
    """Use actual filled inventory, predictive edge and global drawdown for sizing."""
    equity = cash + quantity * close
    exposure = quantity * close / equity
    drawdown = (1 - equity / peak).clamp(min=0)
    dd = float(drawdown.detach())
    emergency = dd >= 0.1
    if index % config.rebalance_bars and not emergency:
        return None
    state = torch.stack((exposure, drawdown, cash.new_tensor(volatility)))
    target = model.allocate(encoded, quantiles, state)
    ceiling = config.max_weight * (0.25 if dd >= 0.15 else 0.5 if dd >= 0.1 else 1.0)
    target = target.clamp(0, ceiling)
    edge = float(quantiles[-1, 1].detach())
    if edge <= 2 * (config.commission + config.slippage) + config.entry_margin:
        target = target * 0
    # Volatility scaling is causal and never increases the model allocation.
    target = target * min(1.0, 0.01 / max(float(volatility), 1e-8))
    difference = target - exposure
    if abs(float(difference.detach())) < config.rebalance_threshold:
        return None
    signal = 1 if float(difference.detach()) > 0 else -1
    amount = difference.abs() * equity / close
    return TensorOrder(
        signal, amount, emergency and signal == -1, signal == -1 and float(target.detach()) == 0.0
    )


def simulate_path(
    model: object,
    encoded: torch.Tensor,
    quantiles: torch.Tensor,
    frame: pd.DataFrame,
    volatility: np.ndarray,
    config: LearningConfig,
) -> PathResult:
    """Replay costs, sized orders, holding deferrals, gaps, stops and final exit.

    Discrete action branches are detached, while filled quantities, accounting,
    portfolio state and growth retain gradients. Each supplied path starts in cash.
    """
    prices = frame[["open", "low", "close"]].to_numpy(dtype=float)
    cash = encoded.new_tensor(1.0)
    quantity = encoded.new_tensor(0.0)
    basis = encoded.new_tensor(0.0)
    peak = cash
    stop = encoded.new_tensor(0.0)
    turnover = encoded.new_tensor(0.0)
    entry_bar = -1
    entries = 0
    day = None
    stopped = False
    pending = deferred = None
    values = []

    def sell(price: float, amount: torch.Tensor) -> None:
        nonlocal cash, quantity, basis, entry_bar, stop, turnover, entries
        entries += 1
        sold = torch.minimum(amount, quantity)
        allocated = basis * sold / quantity
        before = cash + quantity * price
        cash = cash + sold * price * (1 - config.slippage) * (1 - config.commission)
        quantity = quantity - sold
        basis = basis - allocated
        turnover = turnover + sold * price / before
        if float(quantity.detach()) <= 1e-12:
            quantity = quantity * 0
            basis = basis * 0
            stop = stop * 0
            entry_bar = -1

    for i, (opening, low, close) in enumerate(prices):
        if frame.index[i].date() != day:
            day, entries = frame.index[i].date(), 0
        if stopped:
            values.append(cash)
            continue
        order = pending
        pending = None
        risk_exit = False
        opening_equity = cash + quantity * opening
        if float((1 - opening_equity / peak).detach()) >= config.breaker:
            if float(quantity.detach()) > 0:
                sell(opening, quantity)
                risk_exit = True
            stopped = True
        elif float(quantity.detach()) > 0 and opening <= float(stop.detach()):
            sell(opening, quantity)
            risk_exit = True
        if risk_exit or stopped:
            deferred = None
        else:
            if order is not None and order.signal == -1 and float(quantity.detach()) > 0:
                if deferred is None or order.force or order.liquidate:
                    deferred = order
                elif not deferred.liquidate:
                    deferred = TensorOrder(
                        -1, torch.minimum(quantity, deferred.quantity + order.quantity)
                    )
            if deferred is not None and float(quantity.detach()) > 0:
                if deferred.force or i - entry_bar >= config.min_holding_bars:
                    sell(opening, quantity if deferred.liquidate else deferred.quantity)
                    deferred = None
            elif order is not None and order.signal == 1 and entries < config.max_daily_entries:
                executed = opening * (1 + config.slippage)
                equity = cash + quantity * opening
                capacity = (equity * config.max_weight - quantity * opening).clamp(min=0)
                requested = order.quantity * executed / (1 - config.commission)
                budget = torch.minimum(torch.minimum(cash, capacity), requested)
                bought = budget * (1 - config.commission) / executed
                if float(bought.detach()) > 1e-12:
                    if float(quantity.detach()) == 0:
                        entry_bar = i
                    cash = cash - budget
                    quantity = quantity + bought
                    basis = basis + budget
                    entries += 1
                    stop = basis / quantity * (1 - config.stop_loss)
                    turnover = turnover + bought * opening / equity
        if float(quantity.detach()) > 0 and low <= float(stop.detach()):
            sell(min(opening, float(stop.detach())), quantity)
            deferred = None
            risk_exit = True
        value = cash + quantity * close
        peak = torch.maximum(peak, value)
        if float((1 - value / peak).detach()) >= config.breaker:
            if float(quantity.detach()) > 0:
                sell(close, quantity)
                value = cash
                risk_exit = True
            stopped = True
            deferred = None
        values.append(value)
        if i < len(frame) - 1 and not stopped and not risk_exit:
            pending = policy_order(
                model,
                encoded[i],
                quantiles[i],
                cash,
                quantity,
                close,
                peak,
                float(volatility[i]),
                i,
                config,
            )
    if float(quantity.detach()) > 0:
        sell(float(prices[-1, 2]), quantity)
        values[-1] = cash
    return PathResult(torch.stack(values), turnover)


def path_objective(path: PathResult, config: LearningConfig) -> torch.Tensor:
    """Optimize per-bar log growth plus path drawdown and executed turnover."""
    equity = torch.cat((path.equity.new_ones(1), path.equity))
    drawdown = 1 - equity / torch.cummax(equity, dim=0).values
    eta = torch.quantile(drawdown, 0.95)
    cdar = eta + torch.relu(drawdown - eta).mean() / 0.05
    excess = torch.relu(drawdown.max() - config.drawdown_limit)
    n = len(path.equity)
    return (
        -torch.log(equity[-1]) / n
        + config.cdar_penalty * cdar / n
        + config.mdd_penalty * excess.square() / n
        + config.turnover_penalty * path.turnover / n
    )
