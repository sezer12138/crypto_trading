"""Behavioral tests for causal learning and executable trading paths."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def market(n=1400):
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) * 1.001,
            "low": np.minimum(opening, close) * 0.999,
            "close": close,
            "volume": rng.uniform(500, 1500, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def test_features_are_prefix_invariant_and_scaling_is_train_only():
    from learning.data import prepare_data, causal_features
    from learning.config import LearningConfig

    frame = market()
    config = LearningConfig()
    original = prepare_data(frame, config)
    changed = frame.copy()
    changed.iloc[1000:, :4] *= 2
    other = prepare_data(changed, config)
    np.testing.assert_allclose(
        causal_features(frame).iloc[:1000], causal_features(changed).iloc[:1000], equal_nan=True
    )
    np.testing.assert_array_equal(original.mean, other.mean)
    np.testing.assert_array_equal(original.scale, other.scale)


def test_labels_use_next_open_and_do_not_cross_partition_boundaries():
    from learning.data import prepare_data
    from learning.config import LearningConfig

    frame = market()
    data = prepare_data(frame, LearningConfig())
    t = data.train_indices[0]
    assert data.labels[t, 0] == pytest.approx(frame.close.iloc[t + 4] / frame.open.iloc[t + 1] - 1)
    assert data.train_indices[-1] + 24 < data.train_end
    assert data.validation_indices[-1] + 24 < data.validation_end
    assert data.test_start == data.validation_end


@pytest.mark.parametrize("bad", ["gap", "zero", "ohlc", "volume", "short"])
def test_invalid_candles_are_rejected(bad):
    from learning.data import prepare_data
    from learning.config import LearningConfig

    frame = market()
    if bad == "gap":
        frame = frame.drop(frame.index[300])
    elif bad == "zero":
        frame.iloc[300, 0] = 0
    elif bad == "ohlc":
        frame.iloc[300, 1] = 1
    elif bad == "volume":
        frame.iloc[300, 4] = -1
    else:
        frame = frame.iloc[:200]
    with pytest.raises(ValueError):
        prepare_data(frame, LearningConfig())


def test_tcn_predictions_are_causal_ordered_and_bounded():
    torch = pytest.importorskip("torch")
    from learning.model import TCNPolicy
    from learning.config import LearningConfig

    torch.manual_seed(4)
    model = TCNPolicy(20, LearningConfig(hidden=8)).eval()
    x = torch.randn(1, 20, 300)
    with torch.no_grad():
        encoded, q = model.encode(x)
        changed = x.clone()
        changed[:, :, 200:] += 100
        _, q2 = model.encode(changed)
        torch.testing.assert_close(q[:, :200], q2[:, :200])
        assert torch.all(q[..., 0] <= q[..., 1])
        assert torch.all(q[..., 1] <= q[..., 2])
        target = model.allocate(encoded[0], q[0], torch.zeros(300, 3))
        assert torch.all((target >= 0) & (target <= 0.8))


class FixedPolicy:
    """Test policy with a genuine trainable allocation parameter."""

    def __init__(self, torch, weight=0.6):
        self.weight = torch.tensor(weight, requires_grad=True)

    def allocate(self, encoded, quantiles, state):
        return self.weight


@pytest.mark.parametrize("scenario", ["flat", "gap_stop", "partial", "breaker"])
def test_differentiable_replay_matches_canonical_engine(scenario):
    torch = pytest.importorskip("torch")
    from learning.execution import simulate_path, make_engine, path_objective
    from learning.strategy import EncodedStrategy
    from learning.config import LearningConfig

    torch = pytest.importorskip("torch")
    config = LearningConfig(hidden=8)
    frame = market(70)
    if scenario == "flat":
        frame.loc[:, ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    if scenario == "gap_stop":
        frame.iloc[20:, :4] *= 0.8
    if scenario == "breaker":
        config = LearningConfig(hidden=8, stop_loss=0.8)
        frame.iloc[20:, :4] *= 0.5
    if scenario == "partial":
        frame.iloc[25:, :4] *= 1.4
    model = FixedPolicy(torch)
    z = torch.zeros(len(frame), 8)
    q = torch.ones(len(frame), 3, 3) * 0.02
    q[36:, :, 1] = -0.02
    vol = np.zeros(len(frame), dtype=np.float32)
    simulated = simulate_path(model, z, q, frame, vol, config)
    strategy = EncodedStrategy(model, z, q, vol, config)
    actual = make_engine(config, initial_capital=1).run_backtest(frame, strategy)
    np.testing.assert_allclose(
        simulated.equity.detach().numpy(), actual.equity_curve.values, rtol=2e-5, atol=2e-6
    )
    if scenario == "flat":
        assert simulated.equity[-1] < 1
        loss = path_objective(simulated, config)
        loss.backward()
        assert model.weight.grad is not None and torch.isfinite(model.weight.grad)


def test_risk_exits_bypass_holding_and_last_signal_cannot_buy():
    torch = pytest.importorskip("torch")
    from learning.execution import make_engine
    from learning.strategy import EncodedStrategy
    from learning.config import LearningConfig

    torch = pytest.importorskip("torch")
    frame = market(49)
    frame.loc[:, ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    frame.iloc[2, 2] = 90
    config = LearningConfig(min_holding_bars=20)
    q = torch.ones(49, 3, 3) * 0.02
    strategy = EncodedStrategy(FixedPolicy(torch), torch.zeros(49, 8), q, np.zeros(49), config)
    result = make_engine(config).run_backtest(frame, strategy)
    assert result.trades[1].timestamp == frame.index[2]
    assert result.trades[1].strategy_signal == -2
    q[:, :, 1] = -0.02
    q[-1, :, 1] = 0.02
    result = make_engine(config).run_backtest(
        frame, EncodedStrategy(FixedPolicy(torch), torch.zeros(49, 8), q, np.zeros(49), config)
    )
    assert not result.trades


def test_training_exports_reloadable_frozen_results(tmp_path):
    torch = pytest.importorskip("torch")
    from learning.train import run_experiment

    torch = pytest.importorskip("torch")
    from learning.config import LearningConfig
    from learning.execution import make_engine
    from learning.strategy import load_checkpoint
    import json

    torch.set_num_threads(1)
    frame = market(1800)
    path = tmp_path / "candles.csv"
    frame.to_csv(path, index_label="timestamp")
    config = LearningConfig(
        hidden=4, depth=2, pretrain_epochs=1, policy_epochs=1, segment_bars=120, batch_size=8
    )
    report = run_experiment(path, tmp_path / "run", config, [7])
    assert (tmp_path / "run" / "report.html").exists()
    assert report["data"]["sha256"]
    result = report["seeds"][0]
    assert result["selection_partition"] == "validation"
    checkpoint = tmp_path / "run" / result["checkpoint"]
    assert checkpoint.exists()
    start = report["data"]["test_start_index"]
    strategy = load_checkpoint(checkpoint, frame.iloc[:start])
    actual = make_engine(config).run_backtest(frame.iloc[start:], strategy)
    assert actual.equity_curve.iloc[-1] == pytest.approx(
        result["holdout"]["ending_equity"], rel=1e-5
    )
    assert "costs_doubled" in result
    saved = json.loads((tmp_path / "run" / "report.json").read_text())
    assert saved["primary_seed"] == 7


def test_selection_does_not_reward_infeasible_return_or_cash_as_adoption():
    pytest.importorskip("torch")
    from learning.train import selection_score, adoption_status

    good = {"net_return_pct": 5, "max_drawdown_pct": 10, "total_cost": 1, "total_trades": 2}
    bad = dict(good, net_return_pct=100, max_drawdown_pct=25)
    cash = dict(good, net_return_pct=0, total_trades=0)
    assert selection_score(good, 0.2) > selection_score(bad, 0.2)
    assert adoption_status(cash, 0.2) == "rejected"


@pytest.mark.parametrize(
    "values",
    [
        {"commission": float("nan")},
        {"slippage": float("inf")},
        {"drawdown_limit": float("nan")},
        {"rebalance_threshold": -1},
        {"entry_margin": -1},
        {"max_daily_entries": 0},
    ],
)
def test_invalid_learning_settings_fail_explicitly(values):
    from learning.config import LearningConfig

    with pytest.raises(ValueError):
        LearningConfig(**values)


def test_zero_target_requests_complete_exit_without_rounding_dust():
    torch = pytest.importorskip("torch")
    from learning.strategy import EncodedStrategy
    from learning.config import LearningConfig
    from strategies._base import PortfolioState

    model = FixedPolicy(torch)
    z = torch.zeros(25, 8)
    q = torch.ones(25, 3, 3) * 0.02
    q[12, :, 1] = -0.02
    strategy = EncodedStrategy(model, z, q, np.zeros(25), LearningConfig())
    bar = pd.Series({"close": 62890.0})
    state = PortfolioState(4301.8, 0.07941942662000656, 4962.35, 1)
    order = strategy.generate_order(bar, None, state, 12)
    assert order.signal == -1
    assert order.quantity is None


def test_daily_action_cap_counts_sells_before_allowing_more_buys():
    torch = pytest.importorskip("torch")
    from learning.execution import simulate_path, make_engine
    from learning.strategy import EncodedStrategy
    from learning.config import LearningConfig

    frame = market(50)
    frame.loc[:, ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    config = LearningConfig(rebalance_bars=1, min_holding_bars=0, max_daily_entries=2)
    z = torch.zeros(50, 8)
    q = torch.ones(50, 3, 3) * 0.02
    q[1::2, :, 1] = -0.02
    model = FixedPolicy(torch)
    simulated = simulate_path(model, z, q, frame, np.zeros(50), config)
    actual = make_engine(config, initial_capital=1).run_backtest(
        frame, EncodedStrategy(model, z, q, np.zeros(50), config)
    )
    assert len([t for t in actual.trades if t.action == "buy" and t.timestamp.day == 1]) == 1
    np.testing.assert_allclose(
        simulated.equity.detach().numpy(), actual.equity_curve.values, atol=2e-6, rtol=2e-5
    )


def test_checkpoint_inference_rejects_gaps_across_context_boundary():
    torch = pytest.importorskip("torch")
    from learning.model import TCNPolicy
    from learning.strategy import LearnedStrategy
    from learning.config import LearningConfig
    from learning.data import causal_features

    frame = market(500)
    count = len(causal_features(frame).columns)
    config = LearningConfig(hidden=4)
    model = TCNPolicy(count * 2, config)
    strategy = LearnedStrategy(model, np.zeros(count), np.ones(count), config, frame.iloc[:250])
    with pytest.raises(ValueError, match="continuous hourly"):
        strategy.generate_signals(frame.iloc[251:])


def test_checkpoint_cold_start_cannot_trade_without_feature_warmup():
    torch = pytest.importorskip("torch")
    from learning.model import TCNPolicy
    from learning.strategy import LearnedStrategy
    from learning.config import LearningConfig
    from learning.data import causal_features
    from strategies._base import PortfolioState

    frame = market(50)
    count = len(causal_features(frame).columns)
    config = LearningConfig(hidden=4)
    model = TCNPolicy(count * 2, config)
    with torch.no_grad():
        model.quantile_head.bias.view(3, 3)[:, 0] = 10
    strategy = LearnedStrategy(model, np.zeros(count), np.ones(count), config)
    df = strategy.generate_signals(frame)
    assert strategy.generate_order(df.iloc[0], None, PortfolioState(10000, 0, 0, 0), 0) is None


def test_failed_experiment_keeps_data_manifest_and_failure_evidence(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import json
    import learning.train as training
    from learning.config import LearningConfig

    frame = market()
    source = tmp_path / "data.csv"
    frame.to_csv(source, index_label="timestamp")

    def fail_seed(*args):
        raise RuntimeError("Injected replay mismatch")

    monkeypatch.setattr(training, "train_seed", fail_seed)
    directory = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="Injected replay mismatch"):
        training.run_experiment(source, directory, LearningConfig(), [7])
    manifest = json.loads((directory / "manifest.json").read_text())
    failure = json.loads((directory / "failure.json").read_text())
    assert manifest["data"]["sha256"]
    assert manifest["data"]["test_start_index"] == 1120
    assert failure["seed"] == 7
    assert failure["error_type"] == "RuntimeError"
    assert "Injected replay mismatch" in failure["message"]


def test_tensorboard_records_real_training_and_validation(tmp_path):
    """Catch missing events, wrong learning rates, or absent purged validation loss."""
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from learning.train import run_experiment
    from learning.config import LearningConfig

    source = tmp_path / "market.csv"
    market().to_csv(source, index_label="timestamp")
    directory = tmp_path / "experiment"
    config = LearningConfig(hidden=4, depth=2, pretrain_epochs=1, policy_epochs=1)
    result = run_experiment(source, directory, config, [7])
    events = EventAccumulator(str(directory / "tensorboard" / "seed_7")).Reload()
    expected = {
        "pretrain/train_loss_step",
        "pretrain/train_loss",
        "pretrain/val_loss",
        "pretrain/lr",
        "pretrain/grad_norm",
        "policy/train_loss_step",
        "policy/train_loss",
        "policy/val_loss",
        "policy/val_quantile_loss",
        "policy/lr",
        "policy/grad_norm",
        "policy/train_growth_loss",
        "policy/train_cdar_loss",
        "policy/train_mdd_loss",
        "policy/train_turnover_loss",
        "policy/train_auxiliary_loss",
        "validation/net_return_pct",
        "validation/max_drawdown_pct",
        "selection/selected_epoch",
    }
    assert expected <= set(events.Tags()["scalars"])
    for tag in expected:
        assert all(np.isfinite(event.value) for event in events.Scalars(tag))
    assert events.Scalars("pretrain/lr")[0].value == pytest.approx(0.001)
    assert events.Scalars("policy/lr")[0].value == pytest.approx(0.001 / 3)
    assert [event.step for event in events.Scalars("policy/val_loss")] == [0, 1]
    assert events.Scalars("validation/net_return_pct")[-1].value == pytest.approx(
        result["seeds"][0]["history"][-1]["metrics"]["net_return_pct"]
    )
    assert "model/quantile_head.weight" in events.Tags()["histograms"]
    assert "experiment/config/text_summary" in events.Tags()["tensors"]
    assert result["tensorboard"]["enabled"] is True


def test_tensorboard_can_be_disabled(tmp_path):
    """Catch event creation when logging is explicitly disabled."""
    pytest.importorskip("torch")
    from learning.train import run_experiment
    from learning.config import LearningConfig

    source = tmp_path / "market.csv"
    market().to_csv(source, index_label="timestamp")
    directory = tmp_path / "experiment"
    report = run_experiment(
        source,
        directory,
        LearningConfig(hidden=4, depth=2, pretrain_epochs=1, policy_epochs=1),
        [7],
        tensorboard=False,
    )
    assert not (directory / "tensorboard").exists()
    assert report["tensorboard"]["enabled"] is False


def test_validation_loss_uses_purged_labels_and_restores_training_mode():
    """Catch use of holdout labels or dropout while measuring validation loss."""
    torch = pytest.importorskip("torch")
    from learning.config import LearningConfig
    from learning.data import prepare_data
    from learning.model import TCNPolicy
    from learning.train import validation_losses

    config = LearningConfig(hidden=4, depth=2)
    frame = market()
    data = prepare_data(frame, config)
    model = TCNPolicy(44, config).train()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    values = validation_losses(model, data, config, include_policy=False)
    labels = data.labels[840:1096].astype(float)
    predictions = np.array([-np.log(2) / 100, 0, np.log(2) / 100])
    error = (labels[..., None] - predictions) * 100
    expected = np.maximum(
        np.array([0.1, 0.5, 0.9]) * error, np.array([-0.9, -0.5, -0.1]) * error
    ).mean()
    assert values["quantile_loss"] == pytest.approx(expected, rel=1e-6)
    assert model.training
    changed = frame.copy()
    changed.iloc[data.test_start :, :4] *= 2
    assert validation_losses(model, prepare_data(changed, config), config) == pytest.approx(
        validation_losses(model, data, config)
    )


def test_tensorboard_logging_does_not_change_selected_weights(tmp_path):
    """Catch diagnostics consuming training RNG or altering model/optimizer state."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from learning.config import LearningConfig
    from learning.train import run_experiment

    source = tmp_path / "market.csv"
    market().to_csv(source, index_label="timestamp")
    config = LearningConfig(hidden=4, depth=2, pretrain_epochs=1, policy_epochs=1)
    runs = [tmp_path / "enabled", tmp_path / "disabled"]
    reports = [
        run_experiment(source, path, config, [7], tensorboard=enabled)
        for path, enabled in zip(runs, [True, False])
    ]
    states = [torch.load(path / "seed_7.pt", weights_only=True) for path in runs]
    assert states[0]["selected_epoch"] == states[1]["selected_epoch"]
    for name, value in states[0]["state_dict"].items():
        torch.testing.assert_close(value, states[1]["state_dict"][name], rtol=0, atol=0)
    assert reports[0]["seeds"][0]["holdout"] == reports[1]["seeds"][0]["holdout"]
