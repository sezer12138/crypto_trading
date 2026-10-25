"""Reproducible quantile pretraining, net-growth tuning and frozen holdout evaluation."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from learning.config import LearningConfig
from learning.data import PreparedData, prepare_data
from learning.execution import make_engine, path_objective, simulate_path
from learning.model import TCNPolicy, quantile_loss
from learning.monitoring import create_writer, gradient_norms, log_model, loss_components
from learning.report import research_metrics, export_trajectory, write_report
from learning.strategy import EncodedStrategy
from optimization.data import fingerprint_file, load_ohlcv
from strategies import get_strategy
from strategies._base import TradingStrategy
from backtest import BacktestResult

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class BenchmarkStrategy(TradingStrategy):
    """Buy once at the first executable open, or remain entirely in cash."""

    def __init__(self, buy: bool) -> None:
        super().__init__("BuyHold" if buy else "Cash")
        self.buy = buy

    def generate_signals(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Emit one entry event; canonical engine liquidates the last close."""
        df["signal"] = 0
        if self.buy:
            df.iloc[0, df.columns.get_loc("signal")] = 1
        return df


def selection_score(metrics: dict[str, Any], limit: float) -> tuple[bool, float, float]:
    """Select on validation only, preferring feasible return then smaller costs."""
    feasible = metrics["max_drawdown_pct"] <= limit * 100
    return feasible, metrics["net_return_pct"], -metrics["total_cost"]


class ValidationSchedule:
    """Apply patience to a maximizing scalar or lexicographic validation score.

    Feasibility remains the leading policy selection criterion. Negate quantile
    loss to use this same maximizing interface during pretraining.
    """

    def __init__(self, config: LearningConfig, minimum_epochs: int) -> None:
        self.config = config
        self.minimum_epochs = minimum_epochs
        self.best = None
        self.bad_epochs = 0
        self.lr_bad_epochs = 0

    def step(self, score: Any, epoch: int, optimizer: torch.optim.Optimizer) -> bool:
        """Update after validation and return whether the stage should stop."""
        if self.best is None or score > self.best:
            self.best = score
            self.bad_epochs = 0
            self.lr_bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.lr_bad_epochs += 1
            if self.config.lr_patience and self.lr_bad_epochs >= self.config.lr_patience:
                for group in optimizer.param_groups:
                    group["lr"] = max(self.config.min_lr, group["lr"] * self.config.lr_factor)
                self.lr_bad_epochs = 0
        return bool(
            self.config.early_stopping_patience
            and epoch >= self.minimum_epochs
            and self.bad_epochs >= self.config.early_stopping_patience
        )


def training_segments(indices: np.ndarray, segment_bars: int) -> list[tuple[int, int]]:
    """Split chronological eligible runs without traversing gap-purged rows."""
    segments = []
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if not len(run):
            continue
        end = int(run[-1]) + 1
        for start in range(int(run[0]), end, segment_bars):
            right = min(start + segment_bars, end)
            if right - start >= 48:
                segments.append((start, right))
    return segments


def adoption_status(metrics: dict[str, Any], limit: float) -> str:
    """Reject cash, losses and drawdown breaches as evidence of a useful policy."""
    return (
        "eligible_for_further_research"
        if metrics["net_return_pct"] > 0
        and metrics["max_drawdown_pct"] <= limit * 100
        and metrics["total_trades"] > 0
        else "rejected"
    )


def encode_partition(
    model: TCNPolicy, data: PreparedData, start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warm the causal receptive field from already observed history."""
    context_start = max(0, start - model.config.lookback + 1)
    x = torch.from_numpy(data.features[context_start:end].T.copy())[None]
    x = x.to(next(model.parameters()).device)
    encoded, quantiles = model.encode(x)
    offset = start - context_start
    return encoded[0, offset:], quantiles[0, offset:]


def evaluate(
    model: TCNPolicy,
    data: PreparedData,
    start: int,
    end: int,
    config: LearningConfig,
    cost_multiplier: float = 1.0,
) -> tuple[BacktestResult, torch.Tensor, torch.Tensor]:
    """Evaluate one continuous account with authoritative next-open fills."""
    model.eval()
    with torch.no_grad():
        encoded, quantiles = encode_partition(model, data, start, end)
    strategy = EncodedStrategy(model, encoded, quantiles, data.volatility[start:end], config)
    result = make_engine(config, cost_multiplier=cost_multiplier).run_backtest(
        data.frame.iloc[start:end], strategy
    )
    return result, encoded, quantiles


def pretrain(
    model: TCNPolicy,
    data: PreparedData,
    config: LearningConfig,
    seed: int,
    directory: Path,
    writer: Optional["SummaryWriter"] = None,
) -> list[dict[str, Any]]:
    """Optimize executable return quantiles on purged training windows only."""
    if config.pretrain_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS pretraining requested but MPS is unavailable")
    device = torch.device(config.pretrain_device)
    model.to(device)
    indices = data.train_indices[:: config.pretrain_stride]
    windows = np.stack([data.features[i - config.lookback + 1 : i + 1].T for i in indices])
    dataset = TensorDataset(
        torch.from_numpy(windows.copy()), torch.from_numpy(data.labels[indices])
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []
    step = 0
    schedule = ValidationSchedule(config, config.min_pretrain_epochs)
    best_state = None
    best_loss = float("inf")
    best_epoch = None
    needs_validation = bool(
        writer is not None
        or config.restore_best_pretrain
        or config.lr_patience
        or config.early_stopping_patience
    )
    for epoch in range(config.pretrain_epochs):
        started = perf_counter()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, predictions = model.encode(x)
            loss = quantile_loss(predictions[:, -1], y, config.label_scale)
            loss.backward()
            heads = gradient_norms(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if writer is not None:
                writer.add_scalar("pretrain/train_loss_step", losses[-1], step)
                writer.add_scalar("pretrain/lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("pretrain/grad_norm", float(grad_norm), step)
                for name, value in heads.items():
                    writer.add_scalar(f"pretrain/{name}_grad_norm", value, step)
        entry = {
            "stage": "pretrain",
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "optimizer_updates": step,
            "epoch_updates": len(losses),
            "lr": optimizer.param_groups[0]["lr"],
        }
        stop = False
        if needs_validation:
            validation = validation_losses(model, data, config, include_policy=False)
            entry["val_loss"] = validation["quantile_loss"]
            if not np.isfinite(entry["val_loss"]):
                raise RuntimeError("Nonfinite pretraining validation loss")
            if entry["val_loss"] < best_loss:
                best_loss, best_epoch = entry["val_loss"], epoch + 1
                if config.restore_best_pretrain:
                    best_state = copy.deepcopy(model.state_dict())
            stop = schedule.step(-entry["val_loss"], epoch + 1, optimizer)
        entry["next_lr"] = optimizer.param_groups[0]["lr"]
        entry["epoch_seconds"] = perf_counter() - started
        entry["stop_reason"] = (
            "validation_patience"
            if stop
            else "epoch_budget" if epoch + 1 == config.pretrain_epochs else None
        )
        if writer is not None:
            writer.add_scalar("pretrain/train_loss", entry["loss"], epoch + 1)
            writer.add_scalar("pretrain/val_loss", entry["val_loss"], epoch + 1)
            writer.add_scalar("pretrain/epoch_seconds", entry["epoch_seconds"], epoch + 1)
            writer.add_scalar("pretrain/optimizer_updates", step, epoch + 1)
            writer.flush()
        history.append(entry)
        (directory / f"seed_{seed}_history.json").write_text(json.dumps(history, indent=2))
        print(
            f"Seed {seed} pretrain {epoch+1}/{config.pretrain_epochs}: loss={entry['loss']:.6f}",
            flush=True,
        )
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        history[-1]["restored_pretrain_epoch"] = best_epoch
        (directory / f"seed_{seed}_history.json").write_text(json.dumps(history, indent=2))
    model.cpu()
    return history


def validation_losses(
    model: TCNPolicy, data: PreparedData, config: LearningConfig, include_policy: bool = True
) -> dict[str, float]:
    """Measure dropout-free losses with labels contained strictly within validation.

    The policy loss uses a continuous cash-reset account over the purged prefix.
    It is diagnostic only; authoritative full-partition returns select checkpoints.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            start = int(data.validation_indices[0])
            end = int(data.validation_indices[-1]) + 1
            encoded, predictions = encode_partition(model, data, start, end)
            auxiliary = quantile_loss(
                predictions,
                torch.from_numpy(data.labels[start:end]).to(predictions.device),
                config.label_scale,
            )
            values = {"quantile_loss": float(auxiliary)}
            if include_policy:
                path = simulate_path(
                    model,
                    encoded,
                    predictions,
                    data.frame.iloc[start:end],
                    data.volatility[start:end],
                    config,
                )
                values.update(loss_components(path, auxiliary, config))
                values["loss"] = float(
                    path_objective(path, config)
                    + config.auxiliary_weight * auxiliary / (end - start)
                )
            return values
    finally:
        model.train(was_training)


def train_seed(
    data: PreparedData,
    config: LearningConfig,
    seed: int,
    directory: Path,
    tensorboard: bool = True,
) -> dict[str, Any]:
    """Own one seed's event writer and close it even when training fails."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TCNPolicy(data.features.shape[1], config)
    writer = create_writer(directory, seed, config, model) if tensorboard else None
    try:
        return _train_seed(model, data, config, seed, directory, writer)
    finally:
        if writer is not None:
            writer.close()


def _train_seed(
    model: TCNPolicy,
    data: PreparedData,
    config: LearningConfig,
    seed: int,
    directory: Path,
    writer: Optional["SummaryWriter"],
) -> dict[str, Any]:
    """Select a single validation checkpoint before opening the holdout account."""
    history = pretrain(model, data, config, seed, directory, writer)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate / 3, weight_decay=1e-4
    )
    best_score = None
    selected = None
    selected_metrics = None
    selected_epoch = 0
    rng = np.random.default_rng(seed)
    step = 0
    schedule = ValidationSchedule(config, config.min_policy_epochs)
    # Epoch zero is a predefined candidate, not a holdout-driven fallback.
    for epoch in range(config.policy_epochs + 1):
        started = perf_counter()
        if epoch:
            losses = []
            components = []
            segments = training_segments(data.train_indices, config.segment_bars)
            if not segments:
                raise ValueError("No eligible policy segments contain at least 48 bars")
            zero_gradients = 0
            zero_updates = 0
            rng.shuffle(segments)
            for left, right in segments:
                model.train()
                optimizer.zero_grad()
                encoded, predictions = encode_partition(model, data, left, right)
                path = simulate_path(
                    model,
                    encoded,
                    predictions,
                    data.frame.iloc[left:right],
                    data.volatility[left:right],
                    config,
                )
                auxiliary = quantile_loss(
                    predictions, torch.from_numpy(data.labels[left:right]), config.label_scale
                )
                loss = path_objective(path, config) + config.auxiliary_weight * auxiliary / (
                    right - left
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                heads = gradient_norms(model)
                before = [p.detach().clone() for p in model.policy_head.parameters()]
                zero_gradients += int(heads["policy_head"] == 0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                zero_updates += int(
                    all(
                        torch.equal(old, p.detach())
                        for old, p in zip(before, model.policy_head.parameters())
                    )
                )
                losses.append(float(loss.detach()))
                step += 1
                if writer is not None:
                    components.append(loss_components(path, auxiliary, config))
                    writer.add_scalar("policy/train_loss_step", losses[-1], step)
                    writer.add_scalar("policy/lr", optimizer.param_groups[0]["lr"], step)
                    writer.add_scalar("policy/grad_norm", float(grad_norm), step)
                    writer.add_scalar("policy/segment_bars", right - left, step)
                    for name, value in heads.items():
                        writer.add_scalar(f"policy/{name}_grad_norm", value, step)
            entry = {
                "stage": "policy",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "optimizer_updates": step,
                "epoch_updates": len(losses),
                "lr": optimizer.param_groups[0]["lr"],
                "policy_zero_gradient_fraction": zero_gradients / len(losses),
                "policy_zero_update_fraction": zero_updates / len(losses),
            }
            if writer is not None:
                writer.add_scalar("policy/train_loss", entry["loss"], epoch)
                for key in (
                    "policy_zero_gradient_fraction",
                    "policy_zero_update_fraction",
                    "optimizer_updates",
                ):
                    writer.add_scalar(f"policy/{key}", entry[key], epoch)
                for key in components[0]:
                    writer.add_scalar(
                        f"policy/train_{key}", float(np.mean([c[key] for c in components])), epoch
                    )
            history.append(entry)
        validation, _, _ = evaluate(model, data, data.train_end, data.validation_end, config)
        metrics = research_metrics(validation)
        entry = {"stage": "validation", "epoch": epoch, "metrics": metrics}
        if writer is not None or config.early_stopping_patience or config.lr_patience:
            diagnostic = validation_losses(model, data, config)
            entry["losses"] = diagnostic
        if writer is not None:
            writer.add_scalar("policy/val_loss", diagnostic["loss"], epoch)
            writer.add_scalar("policy/val_quantile_loss", diagnostic["quantile_loss"], epoch)
            for key in ("growth_loss", "cdar_loss", "mdd_loss", "turnover_loss", "auxiliary_loss"):
                writer.add_scalar(f"policy/val_{key}", diagnostic[key], epoch)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(f"validation/{key}", value, epoch)
            writer.add_scalar("policy/epoch_seconds", perf_counter() - started, epoch)
            log_model(writer, model, epoch)
            writer.flush()
        history.append(entry)
        (directory / f"seed_{seed}_history.json").write_text(json.dumps(history, indent=2))
        print(
            f"Seed {seed} policy {epoch}/{config.policy_epochs}: validation "
            f"return={metrics['net_return_pct']:.2f}%, DD={metrics['max_drawdown_pct']:.2f}%",
            flush=True,
        )
        score = selection_score(metrics, config.drawdown_limit)
        if best_score is None or score > best_score:
            best_score, selected = score, copy.deepcopy(model.state_dict())
            selected_metrics, selected_epoch = metrics, epoch
        stop = schedule.step(score, epoch, optimizer)
        entry["optimizer_updates"] = step
        entry["next_lr"] = optimizer.param_groups[0]["lr"]
        entry["epoch_seconds"] = perf_counter() - started
        entry["stop_reason"] = (
            "validation_patience"
            if stop
            else "epoch_budget" if epoch == config.policy_epochs else None
        )
        (directory / f"seed_{seed}_history.json").write_text(json.dumps(history, indent=2))
        if stop:
            break
    training_summary = {
        "pretrain_updates": sum(
            row.get("epoch_updates", 0) for row in history if row["stage"] == "pretrain"
        ),
        "policy_updates": step,
        "pretrain_epochs": sum(row["stage"] == "pretrain" for row in history),
        "policy_epochs": epoch,
        "policy_stop_reason": entry["stop_reason"],
        "pretrain_stop_reason": next(
            (row["stop_reason"] for row in reversed(history) if row["stage"] == "pretrain"),
            "zero_epoch_budget",
        ),
    }
    model.load_state_dict(selected)
    if writer is not None:
        writer.add_scalar("selection/selected_epoch", selected_epoch, config.policy_epochs)
    checkpoint = f"seed_{seed}.pt"
    torch.save(
        {
            "state_dict": selected,
            "config": asdict(config),
            "feature_count": data.features.shape[1],
            "feature_names": data.feature_names,
            "mean": data.mean.tolist(),
            "scale": data.scale.tolist(),
            "seed": seed,
            "selected_epoch": selected_epoch,
            "selection_partition": "validation",
            "training_summary": training_summary,
        },
        directory / checkpoint,
    )
    holdout, encoded, quantiles = evaluate(model, data, data.test_start, len(data.frame), config)
    stress, _, _ = evaluate(model, data, data.test_start, len(data.frame), config, 2)
    export_trajectory(holdout, directory, f"seed_{seed}_holdout")
    export_trajectory(stress, directory, f"seed_{seed}_costs_doubled")
    with torch.no_grad():
        replay = simulate_path(
            model,
            encoded,
            quantiles,
            data.frame.iloc[data.test_start :],
            data.volatility[data.test_start :],
            config,
        )
    actual = holdout.equity_curve.to_numpy() / 10000
    error = float(np.max(np.abs(actual - replay.equity.numpy())))
    audit = {
        "seed": seed,
        "selected_epoch": selected_epoch,
        "validation": selected_metrics,
        "holdout": research_metrics(holdout),
        "costs_doubled": research_metrics(stress),
        "max_replay_equity_error_fraction": error,
        "history": history,
        "training_summary": training_summary,
    }
    (directory / f"seed_{seed}_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    if error > 5e-5:
        raise RuntimeError(f"Training/canonical execution mismatch: {error}")
    print(
        f"Seed {seed} frozen holdout: return={research_metrics(holdout)['net_return_pct']:.2f}%, "
        f"DD={research_metrics(holdout)['max_drawdown_pct']:.2f}%, replay error={error:.8f}",
        flush=True,
    )
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "selected_epoch": selected_epoch,
        "selection_partition": "validation",
        "validation": selected_metrics,
        "validation_status": adoption_status(selected_metrics, config.drawdown_limit),
        "holdout": research_metrics(holdout),
        "costs_doubled": research_metrics(stress),
        "holdout_status": adoption_status(research_metrics(holdout), config.drawdown_limit),
        "max_replay_equity_error_fraction": error,
        "history": history,
        "training_summary": training_summary,
    }


def run_experiment(
    data_path: Path,
    output: Path,
    config: LearningConfig,
    seeds: list[int],
    tensorboard: bool = True,
    split_indices: tuple[int, int] | None = None,
    allow_training_gaps: bool = False,
) -> dict[str, Any]:
    """Run predeclared seeds, never rank or retune using the final holdout."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique predeclared seeds")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; experiments cannot be overwritten")
    frame = load_ohlcv(data_path)
    preparation = {}
    if split_indices is not None:
        preparation["split_indices"] = split_indices
    if allow_training_gaps:
        preparation["allow_training_gaps"] = True
    data = prepare_data(frame, config, **preparation)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    report = {
        "primary_seed": seeds[0],
        "config": asdict(config),
        "torch_version": str(torch.__version__),
        "tensorboard": {"enabled": tensorboard, "log_dir": "tensorboard" if tensorboard else None},
        "validation_type": "retrospective_chronological_holdout",
        "data": {
            "path": str(data_path.resolve()),
            "sha256": fingerprint_file(data_path),
            "bars": len(frame),
            "train_end_index": data.train_end,
            "validation_end_index": data.validation_end,
            "test_start_index": data.test_start,
            "train_range": [str(frame.index[0]), str(frame.index[data.train_end - 1])],
            "validation_range": [
                str(frame.index[data.train_end]),
                str(frame.index[data.validation_end - 1]),
            ],
            "test_range": [str(frame.index[data.test_start]), str(frame.index[-1])],
            "feature_names": data.feature_names,
        },
        "seeds": [],
        "baselines": {},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for seed in seeds:
        try:
            report["seeds"].append(train_seed(data, config, seed, output, tensorboard))
        except Exception as exc:
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "status": "failed",
                        "completed_seeds": [r["seed"] for r in report["seeds"]],
                    },
                    indent=2,
                )
            )
            raise
        # Keep each completed seed's audit available if a later seed fails.
        (output / "progress.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    strategies = {
        "cash": BenchmarkStrategy(False),
        "buy_hold_capped": BenchmarkStrategy(True),
        "ma_cross": get_strategy("ma_cross"),
        "momentum_generic": get_strategy("momentum"),
    }
    for name, strategy in strategies.items():
        result = make_engine(config).run_backtest(frame.iloc[data.test_start :], strategy)
        report["baselines"][name] = research_metrics(result)
        export_trajectory(result, output, name)
    from dataclasses import replace

    uncapped = make_engine(replace(config, max_weight=1.0)).run_backtest(
        frame.iloc[data.test_start :], BenchmarkStrategy(True)
    )
    report["baselines"]["buy_hold_uncapped_same_risk"] = research_metrics(uncapped)
    export_trajectory(uncapped, output, "buy_hold_uncapped_same_risk")
    primary = report["seeds"][0]
    best_baseline = max(m["net_return_pct"] for m in report["baselines"].values())
    report["interpretation"] = (
        f"Primary seed {seeds[0]}: holdout net return {primary['holdout']['net_return_pct']:.2f}%, "
        f"maximum drawdown {primary['holdout']['max_drawdown_pct']:.2f}%. "
        f"Validation research gate: {primary['validation_status']}. "
        f"Holdout research gate: {primary['holdout_status']}. "
        f"Highest contextual baseline return: {best_baseline:.2f}%. "
        "All seed outcomes are reported; no seed was selected on holdout. "
        "This experiment does not establish prospective profitability. No profile was activated."
    )
    write_report(report, output)
    return report
