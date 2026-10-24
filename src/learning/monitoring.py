"""TensorBoard diagnostics that do not participate in checkpoint selection."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from learning.config import LearningConfig
from learning.execution import PathResult
from learning.model import TCNPolicy

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def create_writer(
    directory: Path, seed: int, config: LearningConfig, model: TCNPolicy
) -> "SummaryWriter":
    """Create an isolated seed run with reproducible settings and architecture text."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "Install optional dependencies: pip install -r requirements-learning.txt; "
            "or disable logging with --no-tensorboard"
        ) from exc
    writer = SummaryWriter(str(directory / "tensorboard" / f"seed_{seed}"), flush_secs=10)
    writer.add_text(
        "experiment/config", "```json\n" + json.dumps(asdict(config), indent=2) + "\n```"
    )
    writer.add_text("experiment/model", "```text\n" + str(model) + "\n```")
    writer.add_scalar("experiment/parameter_count", sum(p.numel() for p in model.parameters()), 0)
    return writer


def loss_components(
    path: PathResult, auxiliary: torch.Tensor, config: LearningConfig
) -> dict[str, float]:
    """Report detached weighted per-bar terms in the existing policy objective."""
    with torch.no_grad():
        equity = torch.cat((path.equity.new_ones(1), path.equity.detach()))
        drawdown = 1 - equity / torch.cummax(equity, dim=0).values
        eta = torch.quantile(drawdown, 0.95)
        cdar = eta + torch.relu(drawdown - eta).mean() / 0.05
        n = len(path.equity)
        return {
            "growth_loss": float(-torch.log(equity[-1]) / n),
            "cdar_loss": float(config.cdar_penalty * cdar / n),
            "mdd_loss": float(
                config.mdd_penalty * torch.relu(drawdown.max() - config.drawdown_limit).square() / n
            ),
            "turnover_loss": float(config.turnover_penalty * path.turnover.detach() / n),
            "auxiliary_loss": float(config.auxiliary_weight * auxiliary.detach() / n),
        }


def log_model(writer: Optional["SummaryWriter"], model: TCNPolicy, epoch: int) -> None:
    """Log compact head distributions and overall parameter norm once per epoch."""
    if writer is None:
        return
    with torch.no_grad():
        norm = torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))
        writer.add_scalar("model/parameter_norm", float(norm), epoch)
        for name, parameter in model.named_parameters():
            if name.startswith(("quantile_head", "policy_head")):
                writer.add_histogram(f"model/{name}", parameter.detach(), epoch)
