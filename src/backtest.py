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
        }


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
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Win rate based on net round-trip cash flow, including commissions and slippage.
        profitable_sells = 0
        total_sells = 0
        last_buy_value = None
        for trade in self.trades:
            if trade.action == "buy":
                last_buy_value = trade.value
            elif trade.action == "sell":
                total_sells += 1
                if last_buy_value is not None and trade.value > last_buy_value:
                    profitable_sells += 1
                last_buy_value = None

        win_rate = (profitable_sells / total_sells * 100) if total_sells > 0 else 0.0

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
        }

        return self.metrics

    def save_logs(self, filepath: str) -> None:
        """
        Save decision log to JSON file

        Args:
            filepath: Save path
        """
        log_data = {
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
    ):
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

        logger.info("Backtest engine initialized")
        logger.info(f"   Initial capital: ${initial_capital:,.2f}")
        logger.info(f"   Commission: {commission_rate * 100:.2f}%")
        logger.info(f"   Slippage: {slippage * 100:.2f}%")
        logger.info(f"   Position size: {position_size * 100:.0f}%")
        logger.info(f"   Min holding bars: {min_holding_bars}")
        logger.info(f"   Max trades per day: {max_trades_per_day}")
        logger.info(f"   Stop-loss: {stop_loss_pct * 100:.1f}%")
        logger.info(f"   Max drawdown: {max_drawdown_pct * 100:.1f}%")
        if use_atr_stop_loss:
            logger.info(f"   ATR stop-loss multiplier: {atr_stop_loss_multiplier}")
        if breaker_cooldown_bars > 0:
            logger.info(f"   Breaker cooldown: {breaker_cooldown_bars} bars")

    def run_backtest(self, df: pd.DataFrame, strategy: object, coin: str = "BTC") -> BacktestResult:
        """
        Run backtest

        Executes a backtest on historical data using the given strategy.

        Args:
            df: DataFrame containing price data, must have a 'close' column
            strategy: Trading strategy object, must implement generate_signals method
            coin: Coin name (for logging and records)

        Returns:
            BacktestResult object containing complete backtest results

        Raises:
            ValueError: If input data is invalid
        """
        result = BacktestResult(initial_capital=self.initial_capital)

        if df.empty:
            raise ValueError("Input data is empty")
        if "close" not in df.columns:
            raise ValueError("Data missing 'close' column")

        self.reset()

        df = strategy.generate_signals(df.copy())

        if "signal" not in df.columns:
            raise ValueError("Strategy did not generate 'signal' column")

        timestamps = df.index
        prices = df["close"].values
        signals = df["signal"].values.astype(int)
        equity_curve = np.empty(len(df))

        # Pre-compute ATR for dynamic stop-loss (if enabled)
        atr_values = None
        atr_window = 14
        if self.use_atr_stop_loss and "high" in df.columns and "low" in df.columns:
            high = df["high"].values
            low = df["low"].values
            close_prev = np.roll(df["close"].values, 1)
            close_prev[0] = df["close"].values[0]
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - close_prev),
                    np.abs(low - close_prev),
                ),
            )
            # Exponential moving average of true range, shifted to avoid look-ahead
            atr_raw = pd.Series(tr).ewm(span=atr_window, adjust=False).mean().shift(1).values
            atr_values = atr_raw
            logger.info(f"   ATR stop-loss: enabled (multiplier={self.atr_stop_loss_multiplier}, window={atr_window})")

        logger.info(f"Starting backtest for {coin}...")
        logger.info(f"   Strategy: {strategy.name}")
        logger.info(f"   Data points: {len(df)}")

        for i in range(len(df)):
            timestamp = timestamps[i]
            price = prices[i]
            signal = int(signals[i])

            current_day = timestamp.date() if hasattr(timestamp, "date") else timestamp
            if self._current_day != current_day:
                self._current_day = current_day
                self._trades_today = 0

            total_value = self.cash + self.position * price
            if total_value > self._peak_equity:
                self._peak_equity = total_value
            drawdown = (
                (self._peak_equity - total_value) / self._peak_equity
                if self._peak_equity > 0
                else 0
            )
            if drawdown >= self.max_drawdown_pct:
                if self.position > 0:
                    self._execute_sell(
                        timestamp, price, coin, result, FORCED_SELL_SIGNAL, force=True
                    )
                    self._trades_today += 1
                self._stopped = True
                self._breaker_triggered_at = i
            if self._stopped:
                if (
                    self.breaker_cooldown_bars > 0
                    and i - self._breaker_triggered_at >= self.breaker_cooldown_bars
                ):
                    self._stopped = False
                    self._peak_equity = self.cash
                    self._breaker_triggered_at = -1
                    logger.info(
                        f"Breaker cooldown elapsed at bar {i}, trading resumed "
                        f"(equity: ${self.cash:,.2f})"
                    )
                else:
                    equity_curve[i] = self.cash
                    continue

            if self.position > 0:
                entry_price = self.position_value / self.position
                stop_triggered = False
                if self.use_atr_stop_loss and atr_values is not None:
                    current_atr = atr_values[i]
                    if current_atr > 0:
                        stop_distance = self.atr_stop_loss_multiplier * current_atr
                        if (entry_price - price) >= stop_distance:
                            stop_triggered = True
                elif (entry_price - price) / entry_price >= self.stop_loss_pct:
                    stop_triggered = True

                if stop_triggered:
                    pre_sell_entry = entry_price
                    self._execute_sell(timestamp, price, coin, result, signal, force=True)
                    self._trades_today += 1
                    # Track consecutive stop-losses for cooldown
                    if price < pre_sell_entry:
                        self._consecutive_losses += 1
                        if self._consecutive_losses >= self.max_consecutive_losses:
                            self._loss_cooldown_until = (
                                i + self.consecutive_loss_cooldown
                            )
                            logger.warning(
                                f"Consecutive loss limit ({self.max_consecutive_losses}) "
                                f"reached, cooldown until bar {self._loss_cooldown_until}"
                            )
                            self._consecutive_losses = 0
                    else:
                        self._consecutive_losses = 0

            can_sell = self._entry_bar < 0 or (i - self._entry_bar >= self.min_holding_bars)

            # Check loss cooldown before allowing new buy
            in_loss_cooldown = (
                self._loss_cooldown_until >= 0 and i < self._loss_cooldown_until
            )

            if (
                signal == SIGNAL_BUY
                and self.position == 0
                and self._trades_today < self.max_trades_per_day
                and not in_loss_cooldown
            ):
                self._execute_buy(timestamp, price, coin, result, signal)
                self._entry_bar = i
                self._trades_today += 1

            elif signal == SIGNAL_SELL and self.position > 0 and can_sell:
                entry_price = self.position_value / self.position if self.position > 0 else 0
                self._execute_sell(timestamp, price, coin, result, signal)
                self._trades_today += 1
                # Reset consecutive loss counter on profitable regular sell
                if price >= entry_price:
                    self._consecutive_losses = 0
                else:
                    self._consecutive_losses += 1
                    if self._consecutive_losses >= self.max_consecutive_losses:
                        self._loss_cooldown_until = (
                            i + self.consecutive_loss_cooldown
                        )
                        logger.warning(
                            f"Consecutive loss limit ({self.max_consecutive_losses}) "
                            f"reached, cooldown until bar {self._loss_cooldown_until}"
                        )
                        self._consecutive_losses = 0

            # Record end-of-bar equity after all executions and their costs.
            total_value = self.cash + self.position * price
            equity_curve[i] = total_value

            if self.log_decisions:
                result.add_decision(
                    timestamp=timestamp,
                    decision=SIGNAL_TO_ACTION[signal],
                    reason=(
                        f"Signal: {signal}, Price: {price:.2f}, "
                        f"Cash: {self.cash:.2f}, Position: {self.position:.6f}"
                    ),
                    price=price,
                    cash=self.cash,
                    position=self.position,
                    total_value=total_value,
                    signal=signal,
                )

        if self.position > 0:
            final_price = df["close"].iloc[-1]
            self._execute_sell(
                df.index[-1], final_price, coin, result, int(signals[-1]), force=True
            )
            equity_curve[-1] = self.cash

        result.equity_curve = pd.Series(equity_curve, index=timestamps)
        daily_equity = result.equity_curve.resample("1D").last().dropna()
        result.daily_returns = daily_equity.pct_change().dropna()
        result.cumulative_returns = (
            result.equity_curve / self.initial_capital - 1
        ) * 100

        result.calculate_metrics()

        total_cost = 0.0
        for trade in result.trades:
            total_cost += trade.value * (self.commission_rate + self.slippage)
        result.metrics["total_cost"] = round(total_cost, 2)
        result.metrics["cost_drag_pct"] = round(total_cost / self.initial_capital * 100, 2)

        logger.info("Backtest completed")
        logger.info(f"   Final assets: ${result.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"   Total return: {result.metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"   Trade count: {len(result.trades) // 2}")

        return result

    def _execute_buy(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        signal: int,
    ) -> None:
        """
        Execute buy operation

        Args:
            timestamp: Trade time
            price: Current price
            coin: Coin symbol
            result: BacktestResult object
            signal: Strategy signal that triggered this trade
        """
        executed_price = price * (1 + self.slippage)

        position_value = self.cash * self.position_size
        commission = position_value * self.commission_rate
        quantity = (position_value - commission) / executed_price

        self.position = quantity
        self.position_value = position_value
        self.cash -= position_value

        trade = Trade(
            timestamp=timestamp,
            action=ACTION_BUY,
            price=executed_price,
            quantity=quantity,
            value=position_value,
            coin=coin,
            strategy_signal=signal,
        )
        result.add_trade(trade)

        logger.debug(
            f"   Buy @ ${executed_price:.2f}, "
            f"Quantity: {quantity:.6f}, "
            f"Value: ${position_value:.2f}"
        )

    def _execute_sell(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        signal: int,
        force: bool = False,
    ) -> None:
        """
        Execute sell operation

        Args:
            timestamp: Trade time
            price: Current price
            coin: Coin symbol
            result: BacktestResult object
            signal: Strategy signal that triggered this trade (overridden to
                ``FORCED_SELL_SIGNAL`` when ``force`` is True)
            force: Whether this is a forced liquidation (default False)
        """
        executed_price = price * (1 - self.slippage)

        sell_quantity = self.position
        sell_value = sell_quantity * executed_price
        commission = sell_value * self.commission_rate
        net_value = sell_value - commission

        self.cash += net_value
        self.position = 0.0
        self.position_value = 0.0

        trade = Trade(
            timestamp=timestamp,
            action=ACTION_SELL,
            price=executed_price,
            quantity=sell_quantity,
            value=net_value,
            coin=coin,
            strategy_signal=FORCED_SELL_SIGNAL if force else signal,
        )
        result.add_trade(trade)

        logger.debug(
            f"   Sell @ ${executed_price:.2f}, "
            f"Quantity: {sell_quantity:.6f}, "
            f"Net proceeds: ${net_value:.2f}"
        )

    def reset(self) -> None:
        """Reset engine state (cash, position, risk-management counters) to initial values."""
        self.cash = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self._entry_bar = -1
        self._trades_today = 0
        self._current_day = None
        self._peak_equity = self.initial_capital
        self._stopped = False
        self._consecutive_losses = 0
        self._loss_cooldown_until = -1
        self._breaker_triggered_at = -1
        logger.info("Backtest engine reset")


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
