"""Week 4 Wednesday tests: the quality gate. They need no data, so they also run in CI.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.quality_gate import gate_failures, preconditions
from src.evaluate import code_fingerprint

THRESHOLDS = {"min_auc": 0.70, "max_cost": 0.70, "max_auc_drop": 0.02,
              "allow_failing_model_tests": 0}


def metrics(auc: float = 0.745, cost: float = 0.60, failed: int = 0) -> dict:
    """A gate_metrics.json record with only the fields the gate reads."""
    return {"valid_auc": auc, "cost_at_threshold": cost, "threshold": 0.1,
            "model_tests": {"passed": 2, "failed": failed, "xfail": 2, "skipped": 0}}


def make_repo(root: Path) -> Path:
    """A tiny repository with code and config, for the fingerprint."""
    (root / "src").mkdir()
    (root / "config").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "config" / "final.yaml").write_text("costs: {}\n")
    (root / "config" / "gate.yaml").write_text("min_auc: 0.7\n")
    return root


# ---------------------------------------------------------------- provided code (passes now)

def test_fingerprint_follows_the_code(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    before = code_fingerprint(root)
    assert before == code_fingerprint(root)
    (root / "src" / "a.py").write_text("x = 2\n")
    assert code_fingerprint(root) != before


def test_fingerprint_ignores_gate_settings_and_line_endings(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    before = code_fingerprint(root)
    (root / "config" / "gate.yaml").write_text("min_auc: 0.9\n")
    assert code_fingerprint(root) == before, "changing a threshold does not change the model"
    (root / "src" / "a.py").write_bytes(b"x = 1\r\n")
    assert code_fingerprint(root) == before, "Windows line endings give the same fingerprint"


def test_stale_metrics_and_unset_thresholds_are_reported(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    record = {**metrics(), "code_fingerprint": "something-else"}
    problems = preconditions(record, {**THRESHOLDS, "max_cost": None}, root)
    assert any("max_cost" in p for p in problems)
    assert any("stale" in p for p in problems)
    assert preconditions({**metrics(), "code_fingerprint": code_fingerprint(root)},
                         THRESHOLDS, root) == []


# ---------------------------------------------------------------- gate_failures

def test_a_good_model_passes() -> None:
    assert gate_failures(metrics(), baseline_auc=0.745, thresholds=THRESHOLDS) == []


@pytest.mark.parametrize("record, baseline, word", [
    (metrics(auc=0.65), 0.66, "0.650"),    # below the floor (and not worse than baseline)
    (metrics(cost=0.85), 0.745, "0.850"),  # too expensive
    (metrics(auc=0.71), 0.745, "0.745"),   # above the floor, but 0.035 below the registered
    (metrics(failed=1), 0.745, "1"),       # a model test fails
])
def test_each_rule_fails_on_its_own(record: dict, baseline: float, word: str) -> None:
    failures = gate_failures(record, baseline_auc=baseline, thresholds=THRESHOLDS)
    assert len(failures) == 1, f"exactly one rule is broken, got {failures}"
    assert word in failures[0], "say the number in the reason"


def test_every_broken_rule_is_reported() -> None:
    failures = gate_failures(metrics(auc=0.60, cost=0.90, failed=2), baseline_auc=0.745,
                             thresholds=THRESHOLDS)
    assert len(failures) == 4, "do not stop at the first failure"


def test_limits_are_inclusive() -> None:
    """Exactly at a limit passes: the floor is the lowest acceptable value."""
    exact = metrics(auc=0.70, cost=0.70)
    assert gate_failures(exact, baseline_auc=0.715, thresholds=THRESHOLDS) == []


def test_xfail_model_tests_do_not_block() -> None:
    record = metrics()
    record["model_tests"] = {"passed": 1, "failed": 0, "xfail": 3, "skipped": 0}
    assert gate_failures(record, baseline_auc=0.745, thresholds=THRESHOLDS) == []
