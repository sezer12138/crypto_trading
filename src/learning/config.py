"""Explicit, checkpointed settings for BTC/cash research."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class LearningConfig:
    """Model, optimization and execution settings in hourly bars/fraction units."""

    lookback: int = 168
    hidden: int = 32
    depth: int = 6
    dropout: float = 0.1
    horizons: tuple[int, ...] = (4, 12, 24)
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    max_weight: float = 0.8
    commission: float = 0.001
    slippage: float = 0.001
    stop_loss: float = 0.05
    drawdown_limit: float = 0.2
    breaker: float = 0.18
    rebalance_bars: int = 12
    rebalance_threshold: float = 0.1
    entry_margin: float = 0.002
    min_holding_bars: int = 5
    max_daily_entries: int = 6
    pretrain_epochs: int = 6
    policy_epochs: int = 4
    segment_bars: int = 480
    batch_size: int = 16
    learning_rate: float = 0.001
    cdar_penalty: float = 0.5
    mdd_penalty: float = 2.0
    turnover_penalty: float = 0.00002
    auxiliary_weight: float = 0.05
    label_scale: float = 100.0

    def __post_init__(self) -> None:
        """Reject settings that silently change the causal/executable contract."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0 < self.rebalance_threshold < 1 or self.entry_margin < 0:
            raise ValueError("Invalid rebalance threshold or entry margin")
        if (
            min(self.cdar_penalty, self.mdd_penalty, self.turnover_penalty, self.auxiliary_weight)
            < 0
            or self.label_scale <= 0
        ):
            raise ValueError("Loss weights must be nonnegative and label scale positive")
        if self.horizons != (4, 12, 24):
            raise ValueError("This first version requires 4/12/24-hour horizons")
        if not 0 < self.train_ratio < self.train_ratio + self.validation_ratio < 1:
            raise ValueError("Invalid chronological split ratios")
        if self.hidden < 1 or not 1 <= self.depth <= 6 or self.lookback < 127:
            raise ValueError("Require positive hidden size, depth <=6 and lookback >=127")
        if not 0 < self.max_weight <= 1 or not 0 < self.breaker < self.drawdown_limit < 1:
            raise ValueError("Invalid exposure or drawdown limits")
        if not 0 <= self.dropout < 1 or not 0 < self.stop_loss < 1:
            raise ValueError("Invalid dropout or stop loss")
        if any(v < 0 or v >= 1 for v in (self.commission, self.slippage)):
            raise ValueError("Costs must be fractions between zero and one")
        if min(self.rebalance_bars, self.segment_bars, self.batch_size) < 1:
            raise ValueError("Bar and batch counts must be positive")
        if min(self.pretrain_epochs, self.policy_epochs, self.min_holding_bars) < 0:
            raise ValueError("Epoch and holding counts cannot be negative")
        if self.max_daily_entries < 1 or self.learning_rate <= 0:
            raise ValueError("Entry cap and learning rate must be positive")
