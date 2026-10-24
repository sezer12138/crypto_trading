"""Execution-aware learned strategy and immutable checkpoint loading."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from learning.config import LearningConfig
from learning.data import causal_features, transform_features, validate_candles
from learning.execution import policy_order
from learning.model import TCNPolicy
from strategies._base import PortfolioState, TradeOrder, TradingStrategy


class EncodedStrategy(TradingStrategy):
    """Trade aligned causal embeddings using the engine's actual filled account."""

    def __init__(
        self,
        model: object,
        encoded: torch.Tensor,
        quantiles: torch.Tensor,
        volatility: np.ndarray,
        config: LearningConfig,
    ) -> None:
        super().__init__("TCNPolicy")
        self.model, self.encoded, self.quantiles = model, encoded, quantiles
        self.volatility, self.config = volatility, config
        self.peak = 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """The event/order interface emits only actual required adjustments."""
        if len(df) != len(self.encoded):
            raise ValueError("Embeddings and backtest candles must align")
        self.peak = 0.0
        df["signal"] = 0
        return df

    def generate_order(
        self,
        bar: pd.Series,
        previous_bar: Optional[pd.Series],
        portfolio: PortfolioState,
        bar_index: int,
    ) -> Optional[TradeOrder]:
        """Size coin quantity at close; the engine executes it at the next open."""
        close = float(bar.close)
        self.peak = max(self.peak, portfolio.cash + portfolio.quantity * close)
        with torch.no_grad():
            reference = self.encoded
            order = policy_order(
                self.model,
                reference[bar_index],
                self.quantiles[bar_index],
                reference.new_tensor(portfolio.cash),
                reference.new_tensor(portfolio.quantity),
                close,
                reference.new_tensor(self.peak),
                float(self.volatility[bar_index]),
                bar_index,
                self.config,
            )
        if order is None:
            return None
        return TradeOrder(
            order.signal, None if order.liquidate else float(order.quantity), order.force
        )


class LearnedStrategy(EncodedStrategy):
    """Reloadable BTC strategy using checkpoint normalization and past-only context."""

    def __init__(
        self,
        model: TCNPolicy,
        mean: np.ndarray,
        scale: np.ndarray,
        config: LearningConfig,
        context: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__(model.eval(), torch.empty(0), torch.empty(0), np.empty(0), config)
        self.mean, self.scale, self.context = mean, scale, context

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Warm features from earlier history without trading or scoring that history."""
        history = df
        offset = 0
        if self.context is not None and len(self.context):
            if self.context.index[-1] >= df.index[0]:
                raise ValueError("Context must end before evaluation candles")
            history = pd.concat((self.context, df))
            offset = len(self.context)
        validate_candles(history)
        self.first_valid_index = max(0, 168 + self.config.lookback - 1 - offset)
        raw = causal_features(history)
        features = transform_features(raw, self.mean, self.scale)
        with torch.no_grad():
            encoded, quantiles = self.model.encode(torch.from_numpy(features.T.copy())[None])
        self.encoded = encoded[0, offset:]
        self.quantiles = quantiles[0, offset:]
        self.volatility = raw.volatility_24.fillna(0).to_numpy()[offset:]
        return super().generate_signals(df)

    def generate_order(
        self,
        bar: pd.Series,
        previous_bar: Optional[pd.Series],
        portfolio: PortfolioState,
        bar_index: int,
    ) -> Optional[TradeOrder]:
        """Suppress cold-start orders until both features and encoder are warm."""
        if bar_index < self.first_valid_index:
            return None
        return super().generate_order(bar, previous_bar, portfolio, bar_index)


def load_checkpoint(path: Path, context: Optional[pd.DataFrame] = None) -> LearnedStrategy:
    """Load a locally generated weights-only checkpoint without fitting any data."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = LearningConfig(**payload["config"])
    model = TCNPolicy(payload["feature_count"], config)
    model.load_state_dict(payload["state_dict"])
    return LearnedStrategy(
        model, np.asarray(payload["mean"]), np.asarray(payload["scale"]), config, context
    )
