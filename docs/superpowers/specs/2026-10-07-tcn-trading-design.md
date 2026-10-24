# BTC TCN trading research design

Approved scope: implement and train the previously presented BTC/cash first version. Long-only, hourly closed candles, no leverage. Retrospective validation only; the cached history was previously used in strategy research.

Use causal normalized OHLCV features, a small shared temporal convolutional network, ordered return quantiles at 4/12/24 hours, and a policy conditioned on actual exposure, global drawdown and volatility. Labels enter at the next open and exit at the horizon close. The BTC allocation is capped at 80%. Six residual blocks with kernel 2 and dilations 1/2/4/8/16/32 have a 127-bar receptive field; 168 historical bars provide burn-in.

Pretrain quantiles then optimize net log growth, conditional drawdown at risk, maximum-drawdown excess and turnover over chronological trading segments. Discrete trade and stop decisions use piecewise differentiable branches; replay in the canonical BacktestEngine is authoritative. Training and inference share the same policy sizing and risk settings. Record any replay discrepancy.

Risk: 5% entry stop, 5-bar minimum holding, 6 daily counted fills (cap blocks buys only), disabled loss cooldown, 18% liquidation breaker without reset (20% research acceptance limit). Reduce allocation ceilings at 10% and 15% global drawdown. Rebalance at most every 12 bars with a 10 percentage-point exposure threshold. Costs: commission and slippage each 0.1% per side. Predictive entry gate uses 24h median exceeding a full round-trip cost approximation plus 0.2% safety margin. Protective exits bypass ordinary holding gates.

Chronological split 60% training / 20% validation / 20% final holdout. Purge training and validation labels crossing partition ends. Standardizer fit on training only. Past feature context may cross boundaries; future labels cannot. Select the single checkpoint on validation return subject to drawdown <=20%; ties prefer lower turnover. If all candidates are infeasible, export research evidence but reject adoption. Evaluate the frozen model on holdout exactly once per predefined seed, and costs doubled without retuning. Three predeclared seeds are robustness experiments, never selected using holdout. Export every result, including failures and all-cash policies.

Compare cash, capped buy-and-hold, MA crossover and generic momentum under identical risk/cost settings; additionally report uncapped buy-and-hold for context. Report standard daily arithmetic-return Sharpe, net return, CAGR, MDD, Calmar, trades, costs and recovery. Keep existing engine metric definitions unchanged and explicitly compute research metrics separately. No deployment or production profile activation.

Artifacts: isolated ignored results directory with checkpoint, feature scaler, config, data hash, split timestamps, epochs, trajectories, baseline results, HTML and JSON reports. Reusable optional PyTorch dependencies, CLI training and checkpoint replay documented in English.
