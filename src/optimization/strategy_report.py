"""Auditable JSON, validated profiles, and an English optimization summary."""

import html
import json
from pathlib import Path
from typing import Any


def write_results(output: Path, evidence: dict[str, Any]) -> None:
    """Write a partial/final checkpoint, keeping rejected candidates out of profiles."""
    output.mkdir(parents=True, exist_ok=True)
    results = evidence["strategies"]
    (output / "results.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
    )
    profiles = {name: record["parameters"] for name, record in results.items() if record["adopted"]}
    (output / "validated_profiles.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "identity": evidence["identity"],
                "profiles": profiles,
                "evidence_path": str((output / "results.json").resolve()),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    rows = []
    for name, record in results.items():
        baseline, candidate = record["baseline_holdout"], record["candidate_holdout"]
        reasons = ", ".join(key for key, passed in record["adoption_checks"].items() if not passed)
        rows.append(f"""<tr><td>{html.escape(name)}</td>
        <td>{record['baseline_full_sample']['total_return_pct']:.2f}%</td><td>{record['candidate_full_sample']['total_return_pct']:.2f}%</td><td>{record['baseline_training_score']:.2f}</td><td>{record['training_score']:.2f}</td>
        <td>{baseline['total_return_pct']:.2f}%</td><td>{candidate['total_return_pct']:.2f}%</td>
        <td>{candidate['max_drawdown_pct']:.2f}%</td><td>{candidate['total_round_trips']}</td>
        <td>{'Accepted' if record['adopted'] else 'Rejected'}</td><td>{html.escape(reasons)}</td>
        <td><pre>{html.escape(json.dumps(record['parameters'], indent=2))}</pre></td></tr>""")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Strategy Parameter Optimization</title>
    <style>body{{font:15px system-ui;background:#111827;color:#e5e7eb;padding:28px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border:1px solid #374151;text-align:left}}th{{background:#1f2937}}pre{{font-size:12px;white-space:pre-wrap}}a{{color:#60a5fa}}</style>
    <h1>Strategy Parameter Optimization</h1><p>{html.escape(evidence['identity']['coin'].upper())} / {html.escape(evidence['identity']['interval'])} | {evidence['trials_per_strategy']} trials per strategy | {len(results)} completed strategies</p>
    <p>Three chronological development partitions select one candidate per strategy. The final 20% holdout evaluates that frozen candidate against the default, without retrying selection. All returns include commission and slippage.</p>
    <p>This dataset was analyzed before this search. Holdout results are retrospective validation, not new prospective evidence. No parameter setting guarantees positive future returns.</p>
    <p>Holdout: {html.escape(evidence['holdout']['start'])} to {html.escape(evidence['holdout']['end'])}.</p>
    <p>* Full-sample values include training data, are descriptive only, and do not decide adoption.</p><p>Score = median annual return − 0.5 × return standard deviation − drawdown penalty − missing-position penalty.</p>
    <table><thead><tr><th>Strategy</th><th>Default Full Sample*</th><th>Candidate Full Sample*</th><th>Default Train Score</th><th>Selected Train Score</th><th>Default Holdout</th><th>Candidate Holdout</th><th>Holdout Drawdown</th><th>Completed Positions</th><th>Adoption</th><th>Failed Checks</th><th>Candidate Parameters</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <p>Grid and Martingale include position sizing in their search. Compare risk and exposure alongside account returns. Each partition starts with fresh cash and independent inventory; indicator warmup occurs inside that partition.</p>
    <p><a href="results.json">Complete trials and metrics</a> | <a href="validated_profiles.json">Accepted runtime profiles</a></p></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
