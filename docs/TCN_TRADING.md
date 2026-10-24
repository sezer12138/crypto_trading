# BTC/cash TCN research

This optional subsystem forecasts executable return distributions and allocates BTC versus cash. It is a research model, not an activated production strategy. No existing default strategy is changed.

For exact inputs, feature formulas, tensor layouts, parameter counts, losses and artifact schemas, see [the detailed training pipeline](TCN_TRAINING_PIPELINE.md).

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-learning.txt
.venv/bin/python scripts/train_tcn.py \
  --data data/historical/btc_1h_730d.csv \
  --output results/tcn_btc_v1 \
  --seeds 7 19 42 --pretrain-epochs 6 --policy-epochs 4
.venv/bin/python -m pytest tests/ -q
```

The CSV must contain timestamp/open/high/low/close/volume, unique continuous hourly timestamps, physically valid candles and finite values. Missing intervals are rejected rather than filled with invented prices. The trainer refuses to overwrite a nonempty experiment directory. Dependencies remain optional for the rest of the repository.

## TensorBoard monitoring

New training runs log losses, purged validation losses, LR, pre-clipping gradient norms, loss components, head histograms and canonical validation metrics under `<output>/tensorboard/seed_<seed>`. Start `.venv/bin/python -m tensorboard.main --logdir results --host 127.0.0.1 --port 6006` and open http://127.0.0.1:6006. Pass `--no-tensorboard` to disable logging. See [the detailed monitoring guide](TENSORBOARD.md) for interpretation and commands. Older experiments cannot recover missing live-training events retrospectively.

## Model and training

The input contains 22 causal OHLCV-derived features plus 22 availability masks. Six residual TCN blocks use two left-padded kernel-2 convolutions each and dilations 1/2/4/8/16/32: a 127-bar receptive field, within the 168-bar input context. An ordered quantile head predicts 10/50/90% returns for 4/12/24 hours. Labels enter at the next open and exit at the horizon close.

The policy head receives the encoded history, 24h return quantiles, actual filled exposure, drawdown from the account peak and 24h volatility. It produces a BTC weight in [0, 0.8]. Cash earns zero yield. The policy uses a median-return entry gate covering an approximate full round trip plus a 0.2% safety margin. It throttles volatility and exposure at global drawdowns of 10% and 15%.

Pretraining uses training-only, 12-bar-strided windows and pinball loss. Policy tuning uses consecutive, disjoint 480-bar training segments in seeded shuffled order, each with its own initially flat account. The objective combines net log growth, 95% conditional drawdown-at-risk, maximum-drawdown excess, executed turnover, and an auxiliary quantile loss. Quantile scaling is 100. Discrete gates and action selection use detached branches: gradients optimize quantities and accounting along the realized branch, not alternative discrete paths. Stop-price threshold selection is detached. This is a piecewise differentiable trading approximation, not full reinforcement learning.

## Execution and validation

Evaluation uses the repository BacktestEngine with next-open orders, commission 0.1%, adverse slippage 0.1%, 80% exposure cap, five-bar holding minimum, six daily counted fills (entries are blocked at the cap; exits remain allowed), 5% entry stop, disabled loss cooldown, and an 18% equity breaker that permanently stops the evaluation account. Risk exits bypass holding restrictions. The breaker is a trigger, not a guaranteed maximum drawdown; gaps and costs can exceed it. All-cash and rejected models are kept as evidence.

The training replay shares the policy decision function and reproduces canonical quantity orders, complete zero-target liquidations, daily fill counters, cash/capacity constraints, costs, gaps, pending exits, protective stops and final liquidation. Every seed's final holdout checks both full equity paths. A maximum difference exceeding 0.005% of starting capital aborts the run. This verifies the observed trajectory only; behavioral tests cover synthetic corner cases as well.

Partitions are fixed at chronological 60/20/20. Scalers fit only the training partition. Training and validation forecast labels crossing partition ends are purged by 24 bars. Evaluation may use earlier feature history for warmup; it never uses future candles. A predeclared checkpoint candidate is evaluated at policy epoch zero and after every policy epoch. The selected checkpoint maximizes validation net return subject to <=20% drawdown, then prefers lower costs. If no candidate is feasible, the best candidate remains research evidence and is rejected. Holdout is evaluated only after checkpoint selection; holdout results cannot select a seed or trigger tuning. The first supplied seed is the declared primary seed.

Three predefined seeds provide training robustness, not independent market samples. Doubled-cost replay changes execution costs without retraining or adjusting the stored predictive gate. Cash, capped buy-and-hold, MA crossover, generic momentum and uncapped buy-and-hold with the same risk rules are contextual baselines. Buy-and-hold here includes stop/breaker protection; it can liquidate early and is explicitly labeled as risk-controlled.

Metrics are computed independently of existing engine metrics: standard daily arithmetic Sharpe with zero risk-free rate, elapsed-time CAGR, positive peak-to-trough maximum drawdown including initial capital, Calmar, cost-bearing fills, and longest time underwater. End-of-sample underwater periods are censored: no future recovery is assumed.

Cached historical data previously used in strategy optimization makes this retrospective validation, even when this training code respects its holdout. Two years are insufficient for a 24-month-training plus rolling-test schedule. This first experiment uses a single historical split; it does not establish cross-regime or prospective profitability. New unseen data and paper trading remain necessary before considering production use.

## Artifacts and replay

Each ignored `results/` experiment contains weights-only-safe `.pt` checkpoints with model config, train-only scaler and feature schema; data hash and exact partition timestamps in `report.json`; an initial manifest, per-epoch histories, seed audits and explicit failure records; per-seed base and doubled-cost equity/fill exports; baseline trajectories; and a self-contained `report.html`. No checkpoint is automatically registered in `get_strategy()`.

```python
from pathlib import Path
from learning.strategy import load_checkpoint
from learning.execution import make_engine
from optimization.data import load_ohlcv

frame = load_ohlcv(Path('data/historical/btc_1h_730d.csv'))
start = int(len(frame) * 0.8)
strategy = load_checkpoint(Path('results/tcn_btc_v1/seed_7.pt'), context=frame.iloc[:start])
result = make_engine(strategy.config).run_backtest(frame.iloc[start:], strategy, coin='BTC')
```

Only load checkpoints you generated or trust. `weights_only=True` restricts checkpoint deserialization. Loaded inference validates hourly candle geometry and context continuity, uses its stored scaler, never refits historical data, and suppresses cold-start orders until historical features and the encoder are warm.

## References

- Bai, Kolter and Koltun: https://arxiv.org/abs/1803.01271
- Lim et al., Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
- Zhang, Zohren and Roberts, Deep Learning for Portfolio Optimization: https://arxiv.org/abs/2005.13665
- Chekhlov, Uryasev and Zabarankin, Portfolio Optimization with Drawdown Constraints: https://www.math.columbia.edu/~chekhlov/ChekhlovUryasevZabarankin--2003.pdf

The small TCN/auxiliary-quantile/policy combination is this project's design, not a reproduction of the full TFT architecture or a claim that these papers validate BTC profitability.

## First experiment (2026-10-07)

The 17,512-bar cached BTC dataset spans 2024-10-07 through 2026-10-07. The final 3,503-bar historical holdout spans 2026-05-14 through 2026-10-07. Primary seed 7 was declared in advance. All three seeds trained for six quantile epochs and four policy epochs, selecting only by validation.

| Seed / benchmark | Holdout net return | Maximum drawdown | Doubled-cost return |
|---|---:|---:|---:|
| TCN 7 (primary) | -5.67% | 8.03% | -7.25% |
| TCN 19 | -0.41% | 1.51% | -0.68% |
| TCN 42 | -0.26% | 1.02% | -0.90% |
| Generic momentum, same risk settings | +17.90% | 6.37% | Not evaluated |
| Cash | 0.00% | 0.00% | 0.00% |

All learned policies failed the holdout positive-net-return gate. Low drawdown alone is not evidence of a useful model. The primary model underperformed momentum in both return and drawdown. Keep these checkpoints for research; they were not activated.

The initial run aborted on an execution replay mismatch. Regression fixes for zero-target rounding dust and daily sell-count accounting preceded the final run; no hyperparameters or seeds were changed based on holdout performance. The failed run and original log remain under `results/tcn_btc_v1_execution_mismatch/`. The corrected full-path replay errors were at most approximately 1e-7 of initial capital. Tests: 271 passed, including 26 learning tests; 43 existing repository warnings.
