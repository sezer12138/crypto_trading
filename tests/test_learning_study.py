"""Study-level temporal isolation and immutable evidence tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_study_case_grid_separates_capacity_budget_and_history():
    from learning.study import default_cases

    cases = default_cases(pretrain_device="cpu")
    assert len(cases) == 6
    assert len({case.name for case in cases}) == 6
    assert {(case.config.hidden, case.training_start) for case in cases} >= {
        (32, "2018-01-01"),
        (128, "2018-01-01"),
        (32, "2023-01-01"),
        (128, "2023-01-01"),
    }
    extended = [case for case in cases if case.config.pretrain_epochs == 50]
    assert len(extended) == 4
    assert all(case.config.policy_epochs == 30 for case in extended)


def test_variant_selection_ignores_historical_and_recent_test_performance():
    from learning.study import choose_case

    good = {
        "validation": {"net_return_pct": 4, "max_drawdown_pct": 12, "total_cost": 5},
        "holdout": {"net_return_pct": -100},
        "recent_tail": {"net_return_pct": -100},
    }
    poor = {
        "validation": {"net_return_pct": 2, "max_drawdown_pct": 8, "total_cost": 1},
        "holdout": {"net_return_pct": 100},
        "recent_tail": {"net_return_pct": 100},
    }
    assert choose_case({"good": {"seeds": [good]}, "poor": {"seeds": [poor]}}, 0.2) == "good"


def test_small_real_study_freezes_every_checkpoint_before_recent_evaluation(tmp_path):
    torch = pytest.importorskip("torch")
    import json
    from learning.config import LearningConfig
    from learning.study import StudyCase, run_study

    n = 2400
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    prices = 100 * np.exp(np.arange(n) * 0.0001 + 0.001 * np.sin(np.arange(n)))
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": 100.0,
        },
        index=index,
    )
    source = tmp_path / "prices.csv"
    frame.to_csv(source, index_label="timestamp")
    config = LearningConfig(hidden=4, depth=2, pretrain_epochs=1, policy_epochs=1, cpu_threads=1)
    cases = [StudyCase("short", str(index[200]), config), StudyCase("long", str(index[0]), config)]
    output = tmp_path / "study"
    report = run_study(
        source,
        output,
        cases,
        [7],
        str(index[1500]),
        str(index[1900]),
        str(index[2200]),
        tensorboard=False,
    )
    frozen = json.loads((output / "frozen.json").read_text())
    assert frozen["selection_partition"] == "validation"
    assert len(frozen["checkpoints"]) == 2
    assert (output / "report.md").exists()
    assert (output / "report.html").exists()
    for cell in report["cases"].values():
        assert cell["data"]["test_range"][-1] == str(index[2199])
        assert cell["recent_tail"][0]["range"][0] == str(index[2200])
        assert cell["recent_tail"][0]["range"][-1] == str(index[-1])
    with pytest.raises(ValueError, match="empty|resume|exists"):
        run_study(source, output, cases, [7], str(index[1500]), str(index[1900]), str(index[2200]))
    resumed = run_study(
        source,
        output,
        cases,
        [7],
        str(index[1500]),
        str(index[1900]),
        str(index[2200]),
        tensorboard=False,
        resume=True,
    )
    assert resumed["selected_case"] == report["selected_case"]


def test_completed_cell_evidence_cannot_be_modified_on_resume(tmp_path):
    """Catch silently trusting a modified completed checkpoint or result report."""
    from learning.study import verify_completion
    import json
    import hashlib

    cell = tmp_path / "cell"
    cell.mkdir()
    checkpoint = cell / "seed_7.pt"
    checkpoint.write_bytes(b"original checkpoint")
    report = cell / "report.json"
    report.write_text('{"complete": true}')
    evidence = {
        "seed_7.pt": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "report.json": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    (cell / "completion.json").write_text(json.dumps(evidence))
    verify_completion(cell)
    checkpoint.write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="changed|fingerprint"):
        verify_completion(cell)


def test_resume_fingerprint_changes_when_canonical_execution_changes(tmp_path):
    """Catch mixing old completed cells with a changed canonical execution engine."""
    from learning.study import _source_hashes

    root = tmp_path / "repository"
    learning = root / "src" / "learning"
    learning.mkdir(parents=True)
    (learning / "train.py").write_text("training = 'unchanged'\n")
    (root / "src" / "backtest.py").write_text("commission = 0.001\n")
    before = _source_hashes(root)
    (root / "src" / "backtest.py").write_text("commission = 0.01\n")
    assert before != _source_hashes(root)
