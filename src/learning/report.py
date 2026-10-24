"""Independent research metrics and a self-contained evidence report."""

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import BacktestResult


def research_metrics(result: BacktestResult) -> dict[str, Any]:
    """Use positive drawdown and standard daily arithmetic Sharpe, including first day."""
    equity = result.equity_curve
    initial = float(result.initial_capital)
    values = np.r_[initial, equity.to_numpy(dtype=float)]
    peaks = np.maximum.accumulate(values)
    drawdowns = 1 - values / peaks
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1 / 24)
    growth = values[-1] / initial
    cagr = growth ** (365 / days) - 1
    daily = equity.resample("D").last().pct_change()
    daily.iloc[0] = equity.resample("D").last().iloc[0] / initial - 1
    deviation = float(daily.std())
    sharpe = float(daily.mean() / deviation * np.sqrt(365)) if deviation > 0 else 0.0
    costs = sum(t.commission + t.slippage_cost for t in result.trades)
    longest = current = 0
    for dd in drawdowns:
        current = current + 1 if dd > 1e-9 else 0
        longest = max(longest, current)
    mdd = float(drawdowns.max())
    return {
        "net_return_pct": (growth - 1) * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": mdd * 100,
        "daily_sharpe": sharpe,
        "calmar": cagr / mdd if mdd > 0 else None,
        "total_trades": len(result.trades),
        "total_cost": costs,
        "ending_equity": float(values[-1]),
        "max_underwater_hours": longest,
        "underwater_at_end": bool(drawdowns[-1] > 1e-9),
    }


def export_trajectory(result: BacktestResult, directory: Path, name: str) -> None:
    """Persist equity and full cost-bearing fills for audit and plotting."""
    result.equity_curve.rename("equity").to_csv(directory / f"{name}_equity.csv")
    (directory / f"{name}_trades.json").write_text(
        json.dumps([t.to_dict() for t in result.trades], default=str, indent=2)
    )


def write_report(report: dict[str, Any], directory: Path) -> None:
    """Render primary-seed results with all seed/stress evidence retained in JSON."""
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = []
    for seed in report["seeds"]:
        rows.append((f"TCN seed {seed['seed']}", seed["holdout"]))
        rows.append((f"TCN seed {seed['seed']} / doubled costs", seed["costs_doubled"]))
    rows.extend(report["baselines"].items())
    body = "".join(
        "<tr><td>"
        + html.escape(name)
        + "</td>"
        + "".join(
            f"<td>{metrics[key]:.2f}</td>"
            for key in (
                "net_return_pct",
                "max_drawdown_pct",
                "daily_sharpe",
                "total_trades",
                "total_cost",
            )
        )
        + "</tr>"
        for name, metrics in rows
    )
    series = []
    for name in [f"seed_{report['primary_seed']}_holdout", "buy_hold_capped", "ma_cross"]:
        path = directory / f"{name}_equity.csv"
        if path.exists():
            values = pd.read_csv(path).equity.to_numpy()
            series.append((name, values[:: max(1, len(values) // 600)]))
    svg = ""
    if series:
        low = min(float(v.min()) for _, v in series)
        high = max(float(v.max()) for _, v in series)
        span = max(high - low, 1)
        for (name, values), color in zip(series, ["#58a6ff", "#3fb950", "#f2cc60"]):
            points = " ".join(
                f"{20+i*940/max(len(values)-1,1):.1f},{280-(v-low)*250/span:.1f}"
                for i, v in enumerate(values)
            )
            svg += f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        svg = (
            '<svg viewBox="0 0 980 310" role="img" aria-label="Holdout equity curves">'
            + svg
            + "</svg>"
        )
    source = html.escape(report["data"]["path"])
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>BTC TCN research validation</title><style>
body{{font:16px system-ui;background:#0d1117;color:#c9d1d9;max-width:1100px;margin:40px auto;padding:20px}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #30363d}}
td:first-child,th:first-child{{text-align:left}}a{{color:#58a6ff}}svg{{width:100%;background:#161b22}}pre{{white-space:pre-wrap}}
</style><h1>BTC TCN research validation</h1>
<p>Historical holdout on previously researched data. No deployment or production adoption.</p>
<p>Primary seed: {report['primary_seed']} (declared before evaluation). Train / validation / holdout: 60 / 20 / 20.</p>
<p>Data: {source}<br>Holdout: {html.escape(report['data']['test_range'][0])} — {html.escape(report['data']['test_range'][1])}</p>
{svg}<p>Blue: primary TCN. Green: capped buy-and-hold. Yellow: MA crossover. All curves start with 10,000 capital.</p>
<table><thead><tr><th>Strategy</th><th>Net return %</th><th>Max DD %</th><th>Daily Sharpe</th><th>Fills</th><th>Costs</th></tr></thead><tbody>{body}</tbody></table>
<h2>Interpretation</h2><pre>{html.escape(report['interpretation'])}</pre>
<p>Costs include commission and slippage. Drawdown is positive peak-to-trough loss. Sharpe uses arithmetic daily returns and zero cash yield.
Training optimizes reset chronological segments; holdout runs one continuous account. Discrete execution branches are piecewise differentiable.
The breaker is a liquidation trigger, not a guarantee. Intrabar protective-stop rules match the repository engine; the equity breaker observes opens and closes.</p>
<p><a href="report.json">Full configuration, validation selection, all seeds and replay evidence</a></p></html>"""
    (directory / "report.html").write_text(content)
