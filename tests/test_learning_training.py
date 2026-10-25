"""Regression tests for bounded optimization and gap-safe policy segments."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")


def test_training_segments_never_bridge_removed_training_indices():
    from learning.train import training_segments

    indices = np.r_[np.arange(335, 455), np.arange(800, 918)]
    assert training_segments(indices, 60) == [(335, 395), (395, 455), (800, 860), (860, 918)]
    assert training_segments(np.arange(10, 45), 60) == []


def test_lexicographic_schedule_respects_feasibility_and_minimum_epochs():
    from learning.config import LearningConfig
    from learning.train import ValidationSchedule

    config = replace(LearningConfig(), early_stopping_patience=2, lr_patience=1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    schedule = ValidationSchedule(config, minimum_epochs=4)
    assert not schedule.step((False, 100.0, -1.0), 0, optimizer)
    assert not schedule.step((True, 1.0, -1.0), 1, optimizer)
    assert not schedule.step((False, 200.0, -1.0), 2, optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    assert not schedule.step((True, 0.5, -1.0), 3, optimizer)
    assert schedule.step((True, 0.5, -1.0), 4, optimizer)
    assert schedule.best == (True, 1.0, -1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pretrain_stride": 0},
        {"early_stopping_patience": -1},
        {"lr_patience": -1},
        {"lr_factor": 1.0},
        {"min_lr": 0.0},
        {"cpu_threads": 0},
        {"min_pretrain_epochs": 7},
        {"min_policy_epochs": 5},
    ],
)
def test_training_controls_reject_invalid_budgets(overrides):
    from learning.config import LearningConfig

    with pytest.raises(ValueError):
        LearningConfig(**overrides)


def test_head_gradient_norms_include_absent_policy_gradients():
    from learning.config import LearningConfig
    from learning.model import TCNPolicy
    from learning.monitoring import gradient_norms

    model = TCNPolicy(44, LearningConfig(hidden=4, depth=2))
    _, quantiles = model.encode(torch.ones(1, 44, 168))
    quantiles.sum().backward()
    values = gradient_norms(model)
    assert values["quantile_head"] > 0
    assert values["policy_head"] == 0
    assert values["encoder"] > 0


def test_pretrain_schedule_without_events_restores_best_weights_and_counts_updates(
    tmp_path, monkeypatch
):
    """Catch schedules silently disabled without TensorBoard or number-only restoration."""
    from tests.test_learning import market
    from learning.config import LearningConfig
    from learning.data import prepare_data
    from learning.model import TCNPolicy
    import learning.train as training

    config = LearningConfig(
        hidden=4,
        depth=2,
        pretrain_epochs=8,
        pretrain_stride=24,
        early_stopping_patience=2,
        min_pretrain_epochs=4,
        restore_best_pretrain=True,
        lr_patience=1,
    )
    data = prepare_data(market(), config)
    model = TCNPolicy(44, config)
    snapshots = []

    def controlled_validation(model, data, config, include_policy=True):
        snapshots.append(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        )
        return {"quantile_loss": [0.5, 0.4, 0.6, 0.7][len(snapshots) - 1]}

    monkeypatch.setattr(training, "validation_losses", controlled_validation)
    history = training.pretrain(model, data, config, 7, tmp_path, writer=None)
    assert len(history) == 4
    assert history[-1]["stop_reason"] == "validation_patience"
    assert history[-1]["restored_pretrain_epoch"] == 2
    assert history[-1]["optimizer_updates"] == 8
    assert history[-1]["next_lr"] == pytest.approx(0.00025)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name], atol=0, rtol=0)


def test_invalid_pretraining_device_is_rejected():
    from learning.config import LearningConfig

    with pytest.raises(ValueError):
        LearningConfig(pretrain_device="cuda")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires actual MPS hardware")
def test_mps_pretrain_uses_device_and_returns_cpu_weights(tmp_path):
    """Catch device mismatch in validation or a device-leaking policy handoff."""
    from tests.test_learning import market
    from learning.config import LearningConfig
    from learning.data import prepare_data
    from learning.model import TCNPolicy
    from learning.train import pretrain

    config = LearningConfig(
        hidden=4, depth=2, pretrain_epochs=1, pretrain_device="mps", restore_best_pretrain=True
    )
    data = prepare_data(market(), config)
    model = TCNPolicy(44, config)
    before = model.project.weight.detach().clone()
    torch.use_deterministic_algorithms(True)
    history = pretrain(model, data, config, 7, tmp_path)
    assert next(model.parameters()).device.type == "cpu"
    assert not torch.equal(before, model.project.weight.detach())
    assert np.isfinite(history[0]["val_loss"])


def test_disabled_schedule_keeps_legacy_small_learning_rates_valid():
    """Catch new inactive LR-floor settings rejecting older checkpoint configs."""
    from learning.config import LearningConfig

    config = LearningConfig(learning_rate=1e-7)
    assert config.learning_rate == 1e-7
    with pytest.raises(ValueError):
        LearningConfig(learning_rate=1e-7, lr_patience=1)
