#!/usr/bin/env python3
"""Summarize a completed frozen study without selecting on test outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory: Path) -> Path:
    """Create readable seed ranges, paired comparisons and a static research chart."""
    report = json.loads((directory / "report.json").read_text())
    rows = []
    for name, case in report["cases"].items():
        training = [seed["training_summary"] for seed in case["seeds"]]
        historical = [seed["holdout"] for seed in case["seeds"]]
        recent = [seed["metrics"] for seed in case["recent_tail"]]
        validation = [seed["validation"] for seed in case["seeds"]]
        forecast = [
            [row for row in seed["history"] if row["stage"] == "pretrain"] for seed in case["seeds"]
        ]
        gradients = [
            row["policy_zero_gradient_fraction"]
            for seed in case["seeds"]
            for row in seed["history"]
            if row["stage"] == "policy"
        ]
        rows.append(
            {
                "case": name,
                "parameters": 27978 if case["config"]["hidden"] == 32 else 419082,
                "validation_return": [m["net_return_pct"] for m in validation],
                "historical_return": [m["net_return_pct"] for m in historical],
                "historical_mdd": [m["max_drawdown_pct"] for m in historical],
                "recent_return": [m["net_return_pct"] for m in recent],
                "recent_mdd": [m["max_drawdown_pct"] for m in recent],
                "recent_fills": [m["total_trades"] for m in recent],
                "pretrain_updates": [t["pretrain_updates"] for t in training],
                "policy_updates": [t["policy_updates"] for t in training],
                "pretrain_epochs": [t["pretrain_epochs"] for t in training],
                "policy_epochs": [t["policy_epochs"] for t in training],
                "best_pretrain_epoch": [
                    min(rows, key=lambda row: row["val_loss"])["epoch"] for rows in forecast
                ],
                "best_pretrain_val_loss": [
                    min(row["val_loss"] for row in rows) for rows in forecast
                ],
                "last_pretrain_val_loss": [rows[-1]["val_loss"] for rows in forecast],
                "last_pretrain_train_loss": [rows[-1]["loss"] for rows in forecast],
                "policy_zero_gradient_fraction_mean": float(np.mean(gradients)),
            }
        )

    def span(values: list[float], decimals: int = 2) -> str:
        return f"{np.median(values):.{decimals}f} [{min(values):.{decimals}f}, {max(values):.{decimals}f}]"

    text = "# Completed BTC Capacity and History Study\n\n"
    text += f"Validation-selected recipe: **{report['selected_case']}**. Primary seed: **{report['primary_seed']}**.\n\n"
    text += "Every cell ran seeds 7, 19 and 42. Values below show median [minimum, maximum] across the three seeds. "
    text += "These are seed ranges on one market path, not confidence intervals or independent market replications.\n\n"
    text += "| Case | Validation return % | Historical return % | Historical MDD % | Recent return % | Recent MDD % |\n|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += (
            "| "
            + row["case"]
            + " | "
            + " | ".join(
                span(row[key])
                for key in (
                    "validation_return",
                    "historical_return",
                    "historical_mdd",
                    "recent_return",
                    "recent_mdd",
                )
            )
            + " |\n"
        )
    text += "\n## Actual training budgets and gradient activity\n\n"
    text += "| Case | Parameters | Pretrain epochs | Pretrain updates | Policy epochs | Policy updates | Mean zero-policy-gradient fraction |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += f"| {row['case']} | {row['parameters']:,} | " + " | ".join(
            span(row[key], 0)
            for key in ("pretrain_epochs", "pretrain_updates", "policy_epochs", "policy_updates")
        )
        text += f" | {row['policy_zero_gradient_fraction_mean']:.1%} |\n"
    pairs = [
        ("Optimization recipe, width 32", "short_h32_extended", "short_h32_baseline"),
        ("Optimization recipe, width 128", "short_h128_extended", "short_h128_baseline"),
        ("Capacity, short history", "short_h128_extended", "short_h32_extended"),
        ("Capacity, long history", "long_h128_extended", "long_h32_extended"),
        ("Added history, width 32", "long_h32_extended", "short_h32_extended"),
        ("Added history, width 128", "long_h128_extended", "short_h128_extended"),
    ]
    text += "\n## Forecast generalization diagnostics\n\n"
    text += "| Case | Best validation epoch | Best validation loss | Last validation loss | Last training loss |\n|---|---:|---:|---:|---:|\n"
    for row in rows:
        text += "| " + row["case"] + " | " + span(row["best_pretrain_epoch"], 0) + " | "
        text += (
            " | ".join(
                span(row[key], 4)
                for key in (
                    "best_pretrain_val_loss",
                    "last_pretrain_val_loss",
                    "last_pretrain_train_loss",
                )
            )
            + " |\n"
        )
    mapping = {row["case"]: row for row in rows}
    text += "\n## Paired historical comparisons\n\nDifferences are return percentage points for matched seeds; historical outcomes do not rank recipes.\n\n"
    text += "| Comparison | Median paired return difference (pp) | Seeds with improved return |\n|---|---:|---:|\n"
    for label, left, right in pairs:
        change = np.array(mapping[left]["historical_return"]) - mapping[right]["historical_return"]
        text += f"| {label} | {np.median(change):+.2f} | {int((change>0).sum())}/3 |\n"
    text += "\n## Interpretation limits\n\n"
    text += "- The 2026 historical evaluation reuses previously researched dates. It is not a blind test.\n"
    text += "- The 404-hour recent tail was opened after all recipes and checkpoints were frozen; it is too short to establish sustained profitability.\n"
    text += "- An extended budget includes LR decay and best-pretraining restoration; the comparison is not an isolated epoch-count effect.\n"
    text += "- Longer-history cells receive more updates per epoch and therefore more compute.\n"
    text += "- Smaller drawdown can reflect cash holdings or permanent breaker halts rather than a better allocation policy.\n"
    text += "- Hard trading gates can suppress policy gradients; parameter count and updates alone do not resolve this limitation.\n"
    text += "- Recent baselines compute indicators in the evaluation frame, while the learned model receives causal prehistory; this warmup asymmetry remains.\n"
    text += (
        "- Selection uses one common validation year, not multiple expanding-window refit folds.\n"
    )
    text += "- Historical and recent periods start separate fresh-cash accounts; their returns are not one continuous live account.\n"
    text += "- Costs, stops and the 18% breaker match across cells. Results do not activate any production strategy.\n"
    (directory / "summary.md").write_text(text)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["case"].replace("_", "\n") for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ["validation_return", "historical_return", "recent_return"],
        ["2025 selection validation", "2026 researched history", "404-hour frozen recent tail"],
    ):
        values = np.array([row[key] for row in rows])
        medians = np.median(values, axis=1)
        axis.bar(
            np.arange(len(rows)),
            medians,
            color=["#64748b", "#94a3b8", "#0284c7", "#38bdf8", "#059669", "#34d399"],
        )
        for seed in range(values.shape[1]):
            axis.scatter(
                np.arange(len(rows)) + (seed - 1) * 0.08,
                values[:, seed],
                s=22,
                color="#172554",
                zorder=3,
            )
        axis.axhline(0, color="#475569", linewidth=0.8)
        axis.set_xticks(np.arange(len(rows)), labels, fontsize=7)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Net return (%)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("BTC TCN study: bars = seed medians; dots = individual seeds", fontsize=13)
    figure.savefig(directory / "study_returns.png", dpi=170)
    plt.close(figure)
    return directory / "summary.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(summarize(args.directory).resolve())


if __name__ == "__main__":
    main()
