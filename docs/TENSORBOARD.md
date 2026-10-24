# TensorBoard training monitoring

The BTC/cash TCN trainer writes TensorBoard events by default. Each experiment keeps its own logs and each seed has a separate run:

```text
results/<experiment>/
  tensorboard/
    seed_7/events.out.tfevents.*
    seed_19/events.out.tfevents.*
    seed_42/events.out.tfevents.*
```

Logging does not change the optimizer, objective, random seed, execution rules or checkpoint ranking. Diagnostic validation uses `eval()` and `no_grad()`, restores the model's previous mode, and never uses holdout labels. Events are flushed after every completed epoch; the writer closes on normal completion and exceptions.

## Train and open the dashboard

For a new environment, use a clean virtual environment rather than inheriting unrelated TensorFlow installations. TensorFlow is not required for this PyTorch pipeline.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-learning.txt

.venv/bin/python scripts/train_tcn.py \
  --data data/historical/btc_1h_730d.csv \
  --output results/tcn_btc_tensorboard_new \
  --seeds 7 19 42 \
  --pretrain-epochs 6 --policy-epochs 4
```

While training runs, open another terminal:

```bash
.venv/bin/python -m tensorboard.main \
  --logdir results/tcn_btc_tensorboard_new/tensorboard \
  --host 127.0.0.1 --port 6006
```

Open [the local dashboard](http://127.0.0.1:6006). Select runs in the left sidebar to compare seeds. Use the Scalars tab for losses and optimization metrics, Histograms/Distributions for head parameters, and Text for configuration and architecture. Reduce smoothing to zero when inspecting exact recorded loss values. The dashboard can refresh while training is still running.

Use `--logdir results` to compare multiple experiments. Choose a different port if 6006 is occupied. Each training run requires a new empty output directory. Previously completed experiments have no new validation-loss or gradient events; those values cannot be recovered from their saved checkpoints/history alone.

Disable event logging explicitly:

```bash
.venv/bin/python scripts/train_tcn.py \
  --data data/historical/btc_1h_730d.csv \
  --output results/tcn_btc_no_events --seeds 7 --no-tensorboard
```

From Python, pass `tensorboard=False` to `run_experiment`. The logging option is experiment metadata rather than part of `LearningConfig` or the inference checkpoint. Disabling it skips the extra diagnostic validation passes and the TensorBoard import; ordinary canonical validation and checkpoint selection still run.

## Recorded scalar tags

| Tag | Horizontal axis | Meaning |
|---|---|---|
| `pretrain/train_loss_step` | Pretraining optimizer update, starting at 1 | Pinball loss of the current shuffled batch before the optimizer update |
| `pretrain/train_loss` | Pretraining epoch, starting at 1 | Existing unweighted mean of batch losses |
| `pretrain/val_loss` | Pretraining epoch | Pinball loss on all label-safe validation closes, with dropout disabled |
| `pretrain/lr` | Pretraining update | Actual optimizer LR; default 0.001 |
| `pretrain/grad_norm` | Pretraining update | Global L2 gradient norm **before** clipping to 1.0 |
| `pretrain/epoch_seconds` | Pretraining epoch | Elapsed epoch time including diagnostic validation |
| `policy/train_loss_step` | Policy optimizer update, starting at 1 | Total loss of the current chronological training segment |
| `policy/train_loss` | Policy epoch, starting at 1 | Existing unweighted mean of segment losses |
| `policy/train_growth_loss` | Policy epoch | Mean weighted per-bar negative log-growth term |
| `policy/train_cdar_loss` | Policy epoch | Mean weighted per-bar CDaR term |
| `policy/train_mdd_loss` | Policy epoch | Mean weighted per-bar squared drawdown-excess term |
| `policy/train_turnover_loss` | Policy epoch | Mean weighted per-bar executed-turnover term |
| `policy/train_auxiliary_loss` | Policy epoch | Mean weighted per-bar auxiliary pinball term |
| `policy/val_loss` | Policy candidate epoch, **including 0** | Continuous-account policy objective plus auxiliary term on the purged validation prefix |
| `policy/val_quantile_loss` | Policy candidate epoch | Unweighted pinball loss over label-safe validation closes |
| `policy/val_growth_loss`, `val_cdar_loss`, `val_mdd_loss`, `val_turnover_loss`, `val_auxiliary_loss` | Policy candidate epoch | Weighted per-bar validation components; these sum to `policy/val_loss` within floating-point precision |
| `policy/lr` | Policy update | Actual optimizer LR; default 0.001 / 3 |
| `policy/grad_norm` | Policy update | Global L2 gradient norm before clipping to 1.0 |
| `policy/segment_bars` | Policy update | Number of bars in this training segment, including shorter retained tails |
| `policy/epoch_seconds` | Policy candidate epoch | Elapsed time including full canonical validation and diagnostic validation |
| `validation/net_return_pct` | Policy candidate epoch | Authoritative full-validation net account return after costs |
| `validation/max_drawdown_pct` | Policy candidate epoch | Authoritative full-validation maximum drawdown, expressed as a positive percentage |
| `validation/daily_sharpe`, `calmar`, `cagr_pct` | Policy candidate epoch | Research metrics; undefined Calmar values are omitted |
| `validation/total_trades`, `total_cost`, `ending_equity` | Policy candidate epoch | Executed fills, total commission plus slippage costs, and final account equity |
| `validation/max_underwater_hours`, `underwater_at_end` | Policy candidate epoch | Longest observed underwater duration and final underwater indicator |
| `model/parameter_norm` | Policy candidate epoch | Global L2 norm of model parameters |
| `experiment/parameter_count` | 0 | Number of trainable model parameters |
| `selection/selected_epoch` | Final policy epoch | Epoch chosen by the original validation return/risk score |

Pretraining and policy update counters are separate. `grad_norm > 1` is possible because the recorded norm is measured before clipping. There is no LR scheduler in this version: constant LR lines within each stage are expected, and policy training starts with a new optimizer.

`model/quantile_head.*` and `model/policy_head.*` histograms are recorded at policy candidate epochs. `experiment/config` and `experiment/model` text entries show the full configuration and architecture. The exact dataset fingerprint and split boundaries remain in `manifest.json`.

## How validation loss is computed

For the 4/12/24-hour targets, diagnostic validation ends at `validation_end - 24` (exclusive). Thus its last label uses a future close still inside validation. Earlier feature context can warm the causal encoder; no holdout price is used in these labels.

For the completed 17,512-bar BTC snapshot:

```text
Full canonical validation: [10507, 14009) -> 3502 bars
Loss diagnostic prefix:   [10507, 13985) -> 3478 bars
```

Pretraining validation loss is the mean scaled quantile pinball loss over this prefix. It uses every safe validation close; training uses 12-bar-strided windows, shuffled batches and dropout. Therefore the two losses do not have identical sampling/noise conventions. The training epoch mean retains the existing equal weighting of batches, including a smaller last batch.

Policy validation loss starts with unit cash, replays the prefix continuously, applies costs and risk controls, and liquidates at its final close. It uses the same weighted per-bar objective as a training segment:

$$
L=\frac{-\log V_n+0.5\,CDaR_{0.95}
+2\,\max(MDD-0.20,0)^2+0.00002\,TO+0.05\,L_Q}{n}.
$$

Training resets its account for each 480-bar segment and averages the segment objectives equally. Validation uses one longer continuous account with dropout disabled. **These losses are useful diagnostics but are not directly equivalent account experiments.** In particular, the `1/n` scaling can make the longer validation objective numerically smaller even without better trading performance. Inspect the component curves alongside actual net return, drawdown, costs and fill counts.

The total objective can be negative. The MDD-excess term can remain zero below 20% drawdown, and discrete order gates can leave policy gradients inactive on some segments. A falling loss alone does not establish profitability.

Checkpoint selection still prefers feasible full-validation drawdown, then higher net return, then lower costs. Added validation losses do not rank checkpoints or trigger early stopping. Holdout/stress outcomes remain in the frozen report rather than live training curves.

## Historical reports and implementation

The October 7 v1 artifacts remain unchanged and contain the earlier report/history schema. New reports add a top-level `tensorboard` object with `enabled` and the relative `log_dir`. With logging enabled, new history entries also include pretraining `val_loss` and policy-validation `losses` dictionaries. The event directory lives inside the experiment and is ignored with the other result artifacts.

Source: [monitoring helpers](../src/learning/monitoring.py), [training integration](../src/learning/train.py), [CLI](../scripts/train_tcn.py). See [the complete training pipeline](TCN_TRAINING_PIPELINE.md) for the input tensors, exact objective and execution rules. The writer uses the [official PyTorch TensorBoard API](https://docs.pytorch.org/docs/stable/tensorboard).

### Local validation environment note

The existing project `.venv` inherits system packages, including an incompatible TensorFlow 2.10/protobuf installation. For the October 10 verification, TensorBoard 2.21 was installed inside `.venv`, and its `tensorboard.compat.notf` marker was added locally to select TensorBoard's built-in TensorFlow-free compatibility layer. This is an environment-only adjustment, not repository code or a required installation step. An inherited What-If plugin still reports a TensorFlow import error at server startup and is skipped; scalar/text/histogram event writing and the live scalar HTTP endpoint were verified. Prefer the clean virtual environment command above for a fresh installation.
