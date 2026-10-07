"""
Backtest Engine - Simulates historical trading and calculates performance metrics

This module provides complete backtesting functionality, including:
    - Simulated real trading environment (commission, slippage)
    - Detailed trade records and decision logs
    - Comprehensive performance metric calculations
    - Position management support

Classes:
    Trade: Single trade record
    BacktestResult: Backtest result container
    BacktestEngine: Main backtest engine class

Example:
    >>> from backtest import BacktestEngine
    >>> from strategies import MovingAverageCrossStrategy
    >>>
    >>> engine = BacktestEngine(initial_capital=10000.0)
    >>> strategy = MovingAverageCrossStrategy()
    >>> result = engine.run_backtest(df, strategy, coin='BTC')
    >>> print(f"Total return: {result.metrics['total_return_pct']:.2f}%")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from strategies._base import PortfolioState, TradeOrder

from strategies.constants import (
    DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MAX_TRADES_PER_DAY,
    DEFAULT_MIN_HOLDING_BARS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_ATR_STOP_LOSS_MULTIPLIER,
    DEFAULT_MAX_CONSECUTIVE_LOSSES,
    DEFAULT_CONSECUTIVE_LOSS_COOLDOWN,
    DEFAULT_BREAKER_COOLDOWN_BARS,
)

# Configure logging
logger = logging.getLogger(__name__)

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

# Backtest constants
BACKTEST_MODEL_VERSION = "next_open_intrabar_v2"
SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_HOLD = 0
FORCED_SELL_SIGNAL = -2  # Marks stop-loss / end-of-data forced liquidations
ACTION_BUY = "buy"
ACTION_SELL = "sell"
ACTION_HOLD = "hold"
SIGNAL_TO_ACTION = {SIGNAL_BUY: ACTION_BUY, SIGNAL_SELL: ACTION_SELL, SIGNAL_HOLD: ACTION_HOLD}
RISK_FREE_RATE = 0.02
DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30


@dataclass
class Trade:
    """
    Single trade record

    Records a complete trade operation, including time, price, quantity, etc.

    Attributes:
        timestamp: Trade time
        action: Trade action ('buy' or 'sell')
        price: Execution price
        quantity: Trade quantity
        value: Trade value
        coin: Traded coin
        strategy_signal: Strategy signal value (1=buy, -1=sell)

    Example:
        >>> trade = Trade(
        ...     timestamp=datetime.now(),
        ...     action='buy',
        ...     price=50000.0,
        ...     quantity=0.1,
        ...     value=5000.0,
        ...     coin='BTC',
        ...     strategy_signal=1
        ... )
    """

    timestamp: datetime
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    coin: str
    strategy_signal: int
    commission: float = 0.0
    slippage_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary format"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "value": self.value,
            "coin": self.coin,
            "signal": self.strategy_signal,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
        }


def summarize_closed_positions(trades: List[Trade]) -> List[Dict[str, float]]:
    """Summarize flat-to-flat positions using average-cost allocation for partial exits."""
    quantity = cost_basis = episode_cost = episode_pnl = 0.0
    episode_start = None
    episodes = []
    for trade in trades:
        if trade.action == ACTION_BUY:
            if quantity <= 1e-12:
                episode_start = trade.timestamp
            quantity += trade.quantity
            cost_basis += trade.value
            episode_cost += trade.value
        elif trade.action == ACTION_SELL and quantity > 0:
            allocated_cost = cost_basis * min(trade.quantity / quantity, 1.0)
            episode_pnl += trade.value - allocated_cost
            cost_basis -= allocated_cost
            quantity = max(0.0, quantity - trade.quantity)
            if quantity <= 1e-12:
                episodes.append(
                    {
                        "profit": episode_pnl,
                        "return_pct": episode_pnl / episode_cost * 100,
                        "holding_hours": (trade.timestamp - episode_start).total_seconds() / 3600,
                    }
                )
                quantity = cost_basis = episode_cost = episode_pnl = 0.0
    return episodes


@dataclass
class BacktestResult:
    """
    Backtest result container

    Stores all backtest result data, including trade records, equity curve,
    performance metrics, etc.

    Attributes:
        trades: List of trade records
        daily_returns: Daily return series
        cumulative_returns: Cumulative return series
        equity_curve: Equity curve (capital changes)
        metrics: Performance metrics dictionary
        decision_log: Decision log list

    Methods:
        add_trade: Add a trade record
        add_decision: Add a decision record
        calculate_metrics: Calculate performance metrics
        save_logs: Save logs to file
    """

    trades: List[Trade] = field(default_factory=list)
    daily_returns: Optional[pd.Series] = None
    cumulative_returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    initial_capital: Optional[float] = None
    execution_mode: str = "next_open"

    def add_trade(self, trade: Trade) -> None:
        """
        Add a trade record

        Args:
            trade: Trade object
        """
        self.trades.append(trade)

    def add_decision(self, timestamp: datetime, decision: str, reason: str, **kwargs) -> None:
        """
        Log each decision step

        Args:
            timestamp: Decision time
            decision: Decision type ('hold', 'buy', 'sell')
            reason: Reason for the decision
            **kwargs: Other relevant data (e.g., price, cash, position, etc.)
        """
        self.decision_log.append(
            {"timestamp": timestamp.isoformat(), "decision": decision, "reason": reason, **kwargs}
        )

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate backtest performance metrics

        Calculated metrics include:
        - total_return_pct: Total return (%)
        - annual_return_pct: Annualized return (%)
        - volatility_pct: Annualized volatility (%)
        - sharpe_ratio: Sharpe ratio
        - max_drawdown_pct: Maximum drawdown (%)
        - win_rate_pct: Win rate (%)
        - total_trades: Total number of trades
        - trades_per_month: Average trades per month

        Returns:
            Dictionary containing all metrics
        """
        if self.daily_returns is None or len(self.daily_returns) == 0:
            logger.warning("No daily return data available, cannot calculate metrics")
            return {}

        returns = self.daily_returns.dropna()

        if len(returns) == 0:
            logger.warning("Daily return data is empty")
            return {}

        # Basic metrics
        starting_equity = (
            self.initial_capital
            if self.initial_capital is not None
            else float(self.equity_curve.iloc[0])
        )
        total_return = (self.equity_curve.iloc[-1] / starting_equity - 1) * 100

        # Calculate time span (days)
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            logger.warning("Invalid data time span")
            return {}

        # Annualized return
        annual_return = ((1 + total_return / 100) ** (DAYS_PER_YEAR / days) - 1) * 100

        # Annualized volatility
        volatility = returns.std() * np.sqrt(DAYS_PER_YEAR) * 100

        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (annual_return / 100 - RISK_FREE_RATE) / (volatility / 100)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cummax = self.equity_curve.cummax().clip(lower=starting_equity)
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        episodes = summarize_closed_positions(self.trades)
        episode_profits = [p["profit"] for p in episodes]
        episode_returns = [p["return_pct"] for p in episodes]
        holding_hours = [p["holding_hours"] for p in episodes]
        wins = [r for r in episode_returns if r > 0]
        losses = [r for r in episode_returns if r <= 0]
        win_rate = len(wins) / len(episode_returns) * 100 if episode_returns else 0.0
        gross_profit = sum(p for p in episode_profits if p > 0)
        gross_loss = -sum(p for p in episode_profits if p < 0)

        # Trade statistics
        num_trades = len(self.trades)
        trades_per_month = num_trades / (days / DAYS_PER_MONTH) if days > 0 else 0

        self.metrics = {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "volatility_pct": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": num_trades,
            "trades_per_month": round(trades_per_month, 2),
            "total_round_trips": len(episode_returns),
            "average_win_pct": round(float(np.mean(wins)), 2) if wins else 0.0,
            "average_loss_pct": round(float(np.mean(losses)), 2) if losses else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
            "median_holding_hours": (
                round(float(np.median(holding_hours)), 2) if holding_hours else 0.0
            ),
        }

        return self.metrics

    def save_logs(self, filepath: str) -> None:
        """
        Save decision log to JSON file

        Args:
            filepath: Save path
        """
        log_data = {
            "execution_model": BACKTEST_MODEL_VERSION,
            "execution_mode": self.execution_mode,
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
            "decisions": self.decision_log,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Decision log saved: {filepath}")


class BacktestEngine:
    """
    Main backtest engine class

    Simulates a historical trading environment, executes strategies and calculates returns.
    Supports real-world trading factors such as commission, slippage, position management,
    and risk management controls (stop-loss, drawdown circuit breaker, min holding period,
    max trades per day).

    Args:
        initial_capital: Initial capital (default 10000.0)
        commission_rate: Commission rate (default 0.001 = 0.1%)
        slippage: Slippage rate (default 0.001 = 0.1%)
        position_size: Position ratio (default 0.95 = 95%)
        min_holding_bars: Minimum holding period in bars after entry (default 5)
        max_trades_per_day: Maximum number of trades per day (default 6)
        stop_loss_pct: Per-trade stop-loss percentage (default 0.05 = 5%)
        max_drawdown_pct: Max drawdown circuit breaker percentage (default 0.20 = 20%)
        drawdown_breaker_enabled: Whether to enable the drawdown circuit breaker (default True)
        loss_cooldown_enabled: Whether to pause entries after consecutive losses (default True)
        execution_mode: next_open (default) or explicit same_close compatibility timing.
        log_decisions: When True, append a per-bar decision row to ``BacktestResult.decision_log``
            (default False — skipped to avoid ~N dict allocations on long backtests).

    Attributes:
        initial_capital: Initial capital
        commission_rate: Commission rate
        slippage: Slippage rate
        position_size: Position ratio
        min_holding_bars: Minimum holding period in bars
        max_trades_per_day: Maximum trades allowed per day
        stop_loss_pct: Per-trade stop-loss threshold
        max_drawdown_pct: Drawdown circuit breaker threshold
        drawdown_breaker_enabled: Whether the drawdown circuit breaker is enabled
        loss_cooldown_enabled: Whether consecutive-loss cooldown is enabled
        cash: Current cash
        position: Current position quantity
        position_value: Current position value

    Example:
        >>> engine = BacktestEngine(
        ...     initial_capital=10000.0,
        ...     commission_rate=0.001,
        ...     slippage=0.001
        ... )
        >>> result = engine.run_backtest(df, strategy, coin='BTC')
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.001,
        position_size: float = 0.95,
        min_holding_bars: int = DEFAULT_MIN_HOLDING_BARS,
        max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
        log_decisions: bool = False,
        use_atr_stop_loss: bool = False,
        atr_stop_loss_multiplier: float = DEFAULT_ATR_STOP_LOSS_MULTIPLIER,
        max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES,
        consecutive_loss_cooldown: int = DEFAULT_CONSECUTIVE_LOSS_COOLDOWN,
        breaker_cooldown_bars: int = 0,
        drawdown_breaker_enabled: bool = True,
        loss_cooldown_enabled: bool = True,
        execution_mode: str = "next_open",
    ):
        if execution_mode not in {"next_open", "same_close"}:
            raise ValueError("execution_mode must be next_open or same_close")
        self.execution_mode = execution_mode
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size = position_size
        self.min_holding_bars = min_holding_bars
        self.max_trades_per_day = max_trades_per_day
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.log_decisions = log_decisions
        self.use_atr_stop_loss = use_atr_stop_loss
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.max_consecutive_losses = max_consecutive_losses
        self.consecutive_loss_cooldown = consecutive_loss_cooldown
        self.breaker_cooldown_bars = breaker_cooldown_bars
        self.drawdown_breaker_enabled = drawdown_breaker_enabled
        self.loss_cooldown_enabled = loss_cooldown_enabled

        self.cash = initial_capital
        self.position = 0.0
        self.position_value = 0.0

        self._entry_bar = -1
        self._trades_today = 0
        self._current_day = None
        self._peak_equity = initial_capital
        self._stopped = False
        self._consecutive_losses = 0
        self._loss_cooldown_until = -1
        self._breaker_triggered_at = -1
        self._entry_count = 0
        self._last_entry_quantity = 0.0
        self._episode_pnl = 0.0
        self._stop_price = 0.0

        logger.info("Backtest engine initialized")
        logger.info(f"   Initial capital: ${initial_capital:,.2f}")
        logger.info(f"   Commission: {commission_rate * 100:.2f}%")
        logger.info(f"   Slippage: {slippage * 100:.2f}%")
        logger.info(f"   Position size: {position_size * 100:.0f}%")
        logger.info(f"   Min holding bars: {min_holding_bars}")
        logger.info(f"   Max trades per day: {max_trades_per_day}")
        logger.info(f"   Stop-loss: {stop_loss_pct * 100:.1f}%")
        logger.info(
            f"   Consecutive-loss cooldown: "
            f"{'enabled' if loss_cooldown_enabled else 'disabled'}"
        )
        logger.info(f"   Drawdown breaker: {'enabled' if drawdown_breaker_enabled else 'disabled'}")
        if drawdown_breaker_enabled:
            logger.info(f"   Max drawdown: {max_drawdown_pct * 100:.1f}%")
        if use_atr_stop_loss:
            logger.info(f"   ATR stop-loss multiplier: {atr_stop_loss_multiplier}")
        if drawdown_breaker_enabled and breaker_cooldown_bars > 0:
            logger.info(f"   Breaker cooldown: {breaker_cooldown_bars} bars")

    def run_backtest(self, df: pd.DataFrame, strategy: object, coin: str = "BTC") -> BacktestResult:
        """Run causal next-open execution, with gap-aware intrabar protective stops.

        Signals are observed at close and orders fill at the following open. Protective
        stops are active immediately after entry and use only previously known ATR.
        ``same_close`` is an explicit compatibility mode for historical experiments.
        Execution-aware strategies generate orders from actual filled inventory.
        """
        if df.empty or "close" not in df:
            raise ValueError("Input data must contain nonempty close prices")
        if self.execution_mode == "next_open" and "open" not in df:
            raise ValueError("next_open execution requires an open column")
        self.reset()
        result = BacktestResult(
            initial_capital=self.initial_capital, execution_mode=self.execution_mode
        )
        df = strategy.generate_signals(df.copy())
        if "signal" not in df:
            raise ValueError("Strategy did not generate signal column")
        prices = df["close"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float) if "open" in df else prices
        lows = df["low"].to_numpy(dtype=float) if "low" in df else prices
        if (
            not np.isfinite(prices).all()
            or not np.isfinite(opens).all()
            or (prices <= 0).any()
            or (opens <= 0).any()
        ):
            raise ValueError("Execution prices must be finite and positive")
        signals = df["signal"].to_numpy(dtype=int)
        use_atr = self.use_atr_stop_loss or getattr(strategy, "use_atr_stop_loss", False)
        atr_multiplier = (
            getattr(strategy, "multiplier", self.atr_stop_loss_multiplier)
            if getattr(strategy, "use_atr_stop_loss", False)
            else self.atr_stop_loss_multiplier
        )
        atr = None
        if use_atr and "high" in df and "low" in df:
            if "atr" in df:
                atr = df["atr"].shift(1).to_numpy(dtype=float)
            else:
                previous = df["close"].shift(1)
                tr = pd.concat(
                    [
                        df["high"] - df["low"],
                        (df["high"] - previous).abs(),
                        (df["low"] - previous).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = tr.ewm(span=14, adjust=False).mean().shift(1).to_numpy(dtype=float)
        equity = np.empty(len(df))
        pending_order = None
        pending_exit = None
        exposure_bars = blocked_entries = deferred_exits = 0

        def make_order(i: int) -> Optional[TradeOrder]:
            if hasattr(strategy, "generate_order"):
                state = PortfolioState(
                    self.cash,
                    self.position,
                    self.position_value,
                    self._entry_count,
                    self._last_entry_quantity,
                )
                return strategy.generate_order(df.iloc[i], df.iloc[i - 1] if i else None, state, i)
            signal = int(signals[i])
            return TradeOrder(signal) if signal in (SIGNAL_BUY, SIGNAL_SELL) else None

        for i, timestamp in enumerate(df.index):
            opening = opens[i] if self.execution_mode == "next_open" else prices[i]
            current_day = timestamp.date()
            if self._current_day != current_day:
                self._current_day, self._trades_today = current_day, 0
            if (
                self._stopped
                and self.breaker_cooldown_bars > 0
                and i - self._breaker_triggered_at >= self.breaker_cooldown_bars
            ):
                self._stopped = False
                self._peak_equity = self.cash
            if self._stopped:
                equity[i] = self.cash
                pending_order = pending_exit = None
                continue
            order = pending_order if self.execution_mode == "next_open" else make_order(i)
            pending_order = None
            risk_exit = False
            held_at_open = self.position > 0

            # Gap protection takes precedence over an outstanding entry or normal exit.
            opening_equity = self.cash + self.position * opening
            opening_drawdown = 1 - opening_equity / self._peak_equity
            if self.drawdown_breaker_enabled and opening_drawdown >= self.max_drawdown_pct:
                if self.position > 0:
                    self._sell(timestamp, opening, coin, result, TradeOrder(-1, force=True), i)
                    risk_exit = True
                self._stopped = True
                self._breaker_triggered_at = i
            elif self.position > 0 and opening <= self._stop_price:
                self._sell(timestamp, opening, coin, result, TradeOrder(-1, force=True), i)
                risk_exit = True

            if risk_exit or self._stopped:
                pending_exit = None
            else:
                if order is not None and order.signal == SIGNAL_SELL and self.position > 0:
                    if pending_exit is None or order.force or order.quantity is None:
                        pending_exit = order
                    elif pending_exit.quantity is not None:
                        # Preserve every crossed grid level while an exit is deferred.
                        pending_exit = TradeOrder(
                            SIGNAL_SELL, min(self.position, pending_exit.quantity + order.quantity)
                        )
                if pending_exit is not None and self.position > 0:
                    if pending_exit.force or i - self._entry_bar >= self.min_holding_bars:
                        self._sell(timestamp, opening, coin, result, pending_exit, i)
                        pending_exit = None
                    else:
                        deferred_exits += 1
                elif order is not None and order.signal == SIGNAL_BUY:
                    sized = order.quantity is not None
                    cooldown = self.loss_cooldown_enabled and i < self._loss_cooldown_until
                    allowed = (
                        (sized or self.position == 0)
                        and self._trades_today < self.max_trades_per_day
                        and not cooldown
                    )
                    if allowed:
                        before = self.position
                        self._execute_buy(
                            timestamp, opening, coin, result, SIGNAL_BUY, order.quantity
                        )
                        if self.position > before:
                            held_at_open = True
                            if before == 0:
                                self._entry_bar = i
                            self._trades_today += 1
                            average = self.position_value / self.position
                            distance = (
                                atr_multiplier * atr[i]
                                if atr is not None and np.isfinite(atr[i]) and atr[i] > 0
                                else average * self.stop_loss_pct
                            )
                            self._stop_price = average - distance
                        else:
                            blocked_entries += 1
                    elif self.position == 0 or sized:
                        blocked_entries += 1

            # Use the bar low for stop detection, but never assume a gap filled at the stop.
            stop_observation = lows[i] if self.execution_mode == "next_open" else prices[i]
            if self.position > 0 and stop_observation <= self._stop_price:
                self._sell(
                    timestamp,
                    min(opening, self._stop_price),
                    coin,
                    result,
                    TradeOrder(-1, force=True),
                    i,
                )
                pending_exit = None
                risk_exit = True
            total_value = self.cash + self.position * prices[i]
            self._peak_equity = max(self._peak_equity, total_value)
            if (
                self.drawdown_breaker_enabled
                and 1 - total_value / self._peak_equity >= self.max_drawdown_pct
            ):
                if self.position > 0:
                    self._sell(timestamp, prices[i], coin, result, TradeOrder(-1, force=True), i)
                    total_value = self.cash
                    risk_exit = True
                self._stopped = True
                self._breaker_triggered_at = i
                pending_exit = None
            equity[i] = total_value
            exposure_bars += int(held_at_open or self.position > 0)
            if (
                self.execution_mode == "next_open"
                and i < len(df) - 1
                and not self._stopped
                and not risk_exit
            ):
                pending_order = make_order(i)
            if self.log_decisions:
                result.add_decision(
                    timestamp,
                    SIGNAL_TO_ACTION.get(int(signals[i]), ACTION_HOLD),
                    (
                        "Close signal observed; next-open execution"
                        if self.execution_mode == "next_open"
                        else "Compatibility close execution"
                    ),
                    price=prices[i],
                    cash=self.cash,
                    position=self.position,
                    total_value=total_value,
                    signal=int(signals[i]),
                )

        if self.position > 0:
            self._sell(
                df.index[-1], prices[-1], coin, result, TradeOrder(-1, force=True), len(df) - 1
            )
            equity[-1] = self.cash
        result.equity_curve = pd.Series(equity, index=df.index)
        daily = result.equity_curve.resample("1D").last().dropna()
        result.daily_returns = daily.pct_change()
        result.daily_returns.iloc[0] = daily.iloc[0] / self.initial_capital - 1
        result.cumulative_returns = (result.equity_curve / self.initial_capital - 1) * 100
        result.calculate_metrics()
        total_cost = sum(t.commission + t.slippage_cost for t in result.trades)
        result.metrics.update(
            {
                "total_cost": round(total_cost, 2),
                "cost_drag_pct": round(total_cost / self.initial_capital * 100, 2),
                "market_exposure_pct": round(exposure_bars / len(df) * 100, 2),
                "blocked_entry_orders": blocked_entries,
                "deferred_exit_bars": deferred_exits,
                "buy_hold_return_pct": round((prices[-1] / prices[0] - 1) * 100, 2),
            }
        )
        logger.info(
            "Backtest completed | %s | Return: %.2f%% | Completed positions: %s",
            strategy.name,
            result.metrics.get("total_return_pct", 0),
            result.metrics.get("total_round_trips", 0),
        )
        return result

    def _execute_buy(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        signal: int,
        quantity: Optional[float] = None,
    ) -> None:
        """Fill a cash-budgeted entry or a requested quantity under the exposure cap."""
        executed_price = price * (1 + self.slippage)
        if quantity is None:
            budget = self.cash * self.position_size
            bought = budget * (1 - self.commission_rate) / executed_price
        else:
            equity = self.cash + self.position * price
            capacity = max(0.0, equity * self.position_size - self.position * price)
            budget = min(
                self.cash, capacity, quantity * executed_price / (1 - self.commission_rate)
            )
            bought = budget * (1 - self.commission_rate) / executed_price
        if bought <= 1e-12:
            return
        if self.position == 0:
            self._episode_pnl = 0.0
        self.cash -= budget
        self.position += bought
        self.position_value += budget
        self._entry_count += 1
        self._last_entry_quantity = bought
        result.add_trade(
            Trade(
                timestamp,
                ACTION_BUY,
                executed_price,
                bought,
                budget,
                coin,
                signal,
                budget * self.commission_rate,
                bought * (executed_price - price),
            )
        )

    def _execute_sell(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        signal: int,
        force: bool = False,
        quantity: Optional[float] = None,
    ) -> float:
        """Fill a partial or complete exit and return net realized profit."""
        if self.position <= 0:
            return 0.0
        sold = self.position if quantity is None else min(quantity, self.position)
        if sold <= 0:
            return 0.0
        executed_price = price * (1 - self.slippage)
        gross = sold * executed_price
        commission = gross * self.commission_rate
        net = gross - commission
        allocated = self.position_value * sold / self.position
        self.cash += net
        self.position -= sold
        self.position_value -= allocated
        self._episode_pnl += net - allocated
        if self.position <= 1e-12:
            self.position = self.position_value = 0.0
            self._entry_count = 0
            self._last_entry_quantity = 0.0
            self._entry_bar = -1
            self._stop_price = 0.0
        result.add_trade(
            Trade(
                timestamp,
                ACTION_SELL,
                executed_price,
                sold,
                net,
                coin,
                FORCED_SELL_SIGNAL if force else signal,
                commission,
                sold * (price - executed_price),
            )
        )
        return net - allocated

    def _sell(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        order: TradeOrder,
        bar_index: int,
    ) -> None:
        """Execute an exit and update cooldown only when the entire position closes."""
        self._execute_sell(timestamp, price, coin, result, SIGNAL_SELL, order.force, order.quantity)
        self._trades_today += 1
        if self.position == 0 and self.loss_cooldown_enabled:
            self._consecutive_losses = self._consecutive_losses + 1 if self._episode_pnl < 0 else 0
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._loss_cooldown_until = bar_index + self.consecutive_loss_cooldown
                logger.warning(
                    "Consecutive loss limit (%s) reached, cooldown until bar %s",
                    self.max_consecutive_losses,
                    self._loss_cooldown_until,
                )
                self._consecutive_losses = 0

    def reset(self) -> None:
        """Reset all cash, inventory, order-accounting, and risk state for a fresh run."""
        self.cash = self.initial_capital
        self.position = self.position_value = 0.0
        self._entry_bar = -1
        self._entry_count = 0
        self._last_entry_quantity = 0.0
        self._episode_pnl = 0.0
        self._stop_price = 0.0
        self._trades_today = 0
        self._current_day = None
        self._peak_equity = self.initial_capital
        self._stopped = False
        self._consecutive_losses = 0
        self._loss_cooldown_until = -1
        self._breaker_triggered_at = -1


if __name__ == "__main__":
    from strategies import MovingAverageCrossStrategy

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Run backtest
    strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run_backtest(df, strategy, coin="TEST")

    print("\nBacktest results:")
    for key, value in result.metrics.items():
        print(f"   {key}: {value}")
