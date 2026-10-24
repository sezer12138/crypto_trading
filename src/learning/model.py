"""Small causal TCN with ordered quantile and state-conditioned allocation heads."""

import torch
from torch import nn
from torch.nn import functional as F

from learning.config import LearningConfig


class CausalBlock(nn.Module):
    """Residual block whose convolutions see only current and earlier bars."""

    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(hidden, hidden, 2, dilation=dilation)
        self.second = nn.Conv1d(hidden, hidden, 2, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve sequence length using left-only padding."""
        y = self.dropout(F.gelu(self.first(F.pad(x, (self.dilation, 0)))))
        y = self.dropout(F.gelu(self.second(F.pad(y, (self.dilation, 0)))))
        return x + y


class TCNPolicy(nn.Module):
    """Forecast return quantiles and choose a long-only target BTC weight."""

    def __init__(self, feature_count: int, config: LearningConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Conv1d(feature_count, config.hidden, 1)
        self.blocks = nn.Sequential(
            *[CausalBlock(config.hidden, 2**i, config.dropout) for i in range(config.depth)]
        )
        self.quantile_head = nn.Linear(config.hidden, len(config.horizons) * 3)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden + 6, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1)
        )
        nn.init.zeros_(self.quantile_head.bias)
        nn.init.normal_(self.quantile_head.weight, std=0.001)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume [batch, features, time]; return time-aligned embeddings/quantiles."""
        encoded = self.blocks(self.project(features)).transpose(1, 2)
        raw = self.quantile_head(encoded).reshape(*encoded.shape[:-1], 3, 3)
        median = raw[..., 0] / self.config.label_scale
        lower = median - F.softplus(raw[..., 1]) / self.config.label_scale
        upper = median + F.softplus(raw[..., 2]) / self.config.label_scale
        return encoded, torch.stack((lower, median, upper), dim=-1)

    def allocate(
        self, encoded: torch.Tensor, quantiles: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """State columns are filled exposure, global drawdown and 24h volatility."""
        prediction = quantiles[..., -1, :] * self.config.label_scale
        inputs = torch.cat((encoded, prediction, state), dim=-1)
        return torch.sigmoid(self.policy_head(inputs).squeeze(-1)) * self.config.max_weight


def quantile_loss(
    predictions: torch.Tensor, labels: torch.Tensor, scale: float = 100.0
) -> torch.Tensor:
    """Pinball loss for 10/50/90% next-open executable return labels."""
    error = (labels.unsqueeze(-1) - predictions) * scale
    levels = predictions.new_tensor([0.1, 0.5, 0.9])
    return torch.maximum(levels * error, (levels - 1) * error).mean()
