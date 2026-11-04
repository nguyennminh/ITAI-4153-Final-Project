"""The quality gate: block any change that makes the model worse than the team accepts.

Run it with ``python run.py gate``. Cloud Build runs it on every pull request, as the last
step of cloudbuild.yaml. It exits with code 0 (pass) or 1 (fail); a non-zero exit turns the
build red, and branch protection then blocks the merge.

It reads three committed files, never the data (the CI server has none):

    reports/gate_metrics.json   python run.py evaluate: the model this commit would ship
    reports/final_model.json    Week 3: the model that is registered now (the baseline)
    config/gate.yaml            your thresholds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from src.evaluate import GATE_METRICS, code_fingerprint
from src.experiment import load_week3_config

GATE_CONFIG = Path("config/gate.yaml")
BASELINE = Path("reports/final_model.json")
THRESHOLDS = ("min_auc", "max_cost", "max_auc_drop", "allow_failing_model_tests")


# TODO(student)
def gate_failures(
    metrics: dict[str, Any], baseline_auc: float, thresholds: dict[str, Any]
) -> list[str]:
    """Every reason the candidate model must not be merged; an empty list means it passes.

    Check, and add one plain-language reason per broken rule:

    1. ``metrics["valid_auc"]`` is below ``thresholds["min_auc"]``;
    2. ``metrics["cost_at_threshold"]`` is above ``thresholds["max_cost"]``;
    3. ``metrics["valid_auc"]`` is below ``baseline_auc - thresholds["max_auc_drop"]``
       (worse than the registered model by more than the allowed drop);
    4. ``metrics["model_tests"]["failed"]`` is above
       ``thresholds["allow_failing_model_tests"]``.

    Each reason names the number and the limit, for example
    ``"validation AUC 0.651 is below the floor 0.70"``, so whoever reads the build log knows
    what to fix without opening any code.

    Args:
        metrics: reports/gate_metrics.json, as a dict.
        baseline_auc: the registered model's validation AUC (reports/final_model.json).
        thresholds: config/gate.yaml, as a dict (no nulls: ``main`` checks that first).

    Returns:
        The reasons, in the order above. Check every rule: do not stop at the first failure.
    """
    raise NotImplementedError("TODO(student): one reason per broken rule")


# PROVIDED
def load_inputs(root: Path) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Read the three committed files, with a clear message if one is missing."""
    missing = [p for p in (GATE_METRICS, BASELINE, GATE_CONFIG) if not (root / p).exists()]
    if missing:
        raise SystemExit(f"GATE FAILED: missing {[str(p) for p in missing]}. Run python run.py "
                         "evaluate (and python run.py final in Week 3), then commit them.")
    metrics = json.loads((root / GATE_METRICS).read_text())
    baseline_auc = float(json.loads((root / BASELINE).read_text())["valid"]["valid_auc"])
    thresholds = yaml.safe_load((root / GATE_CONFIG).read_text()) or {}
    return metrics, baseline_auc, thresholds


# PROVIDED
def preconditions(metrics: dict[str, Any], thresholds: dict[str, Any], root: Path) -> list[str]:
    """Problems that make the gate meaningless: unset thresholds, or stale metrics."""
    problems = [f"threshold '{name}' is not set in config/gate.yaml"
                for name in THRESHOLDS if thresholds.get(name) is None]
    current = code_fingerprint(root)
    if metrics.get("code_fingerprint") != current:
        problems.append("stale metrics: the code or config changed after reports/gate_metrics.json "
                        f"was written ({metrics.get('code_fingerprint')} != {current}). Run "
                        "python run.py evaluate and commit the new file.")
    return problems


# PROVIDED
def main() -> None:
    root = load_week3_config()["root"]
    metrics, baseline_auc, thresholds = load_inputs(root)
    print(f"Candidate: validation AUC {metrics['valid_auc']:.3f}, cost "
          f"{metrics['cost_at_threshold']:.3f} at threshold {metrics['threshold']:.3f}, model "
          f"tests {metrics['model_tests']}; registered model AUC {baseline_auc:.3f}")
    problems = preconditions(metrics, thresholds, root)
    if not problems:
        problems = gate_failures(metrics, baseline_auc, thresholds)
    if problems:
        print("\nGATE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nGATE PASSED")


if __name__ == "__main__":
    main()
