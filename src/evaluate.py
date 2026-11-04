"""Measure the model THIS commit would ship, and write the metrics file the quality gate reads.

# PROVIDED: this whole module is finished.

Run it with ``python run.py evaluate`` before you commit a change to code or config. It:

1. builds the model from the code and config/final.yaml exactly as ``python run.py final``
   does (calibrated), WITHOUT registering it,
2. scores the validation split: AUC with its 95% interval, PR-AUC, KS, and the expected cost
   at the threshold of your registered model,
3. reads the model-test results from reports/test_report.json,
4. writes reports/gate_metrics.json with a **fingerprint** of the code and config.

The CI server has no data, so it cannot measure the model itself. It reads this committed
file instead, and recomputes the fingerprint from the code in the pull request: if the code
changed after the metrics were measured, the gate fails ("stale metrics"). So every change
to the code must come with fresh metrics, and nobody can merge numbers that belong to
different code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.calibration import calibrate
from src.cost import decision_summary
from src.evaluation import validation_metrics
from src.experiment import load_training_data, load_week3_config
from src.models import make_model
from src.tracking import git_commit
from src.tune import model_spec

GATE_METRICS = Path("reports/gate_metrics.json")
TEST_REPORT = Path("reports/test_report.json")
# What the fingerprint covers: every source file and config, except the gate's own settings
# (changing a threshold does not change the model).
FINGERPRINT_GLOBS = ("src/**/*.py", "config/*.yaml", "config/*.csv")
NOT_FINGERPRINTED = {"config/gate.yaml"}


# PROVIDED
def code_fingerprint(root: Path) -> str:
    """A short hash of the code and config that build the model.

    Line endings are normalized, so Windows and Linux checkouts give the same hash.
    """
    digest = hashlib.sha256()
    files = sorted({p for pattern in FINGERPRINT_GLOBS for p in root.glob(pattern)
                    if p.relative_to(root).as_posix() not in NOT_FINGERPRINTED
                    and "__pycache__" not in p.parts})
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


# PROVIDED
def registered_threshold(config: dict[str, Any]) -> float:
    """The threshold stored with the registered model (Week 3), or the Week 3 choice."""
    outputs = config["week3"]["outputs"]
    for key in ("final_model", "threshold"):
        if outputs[key].exists():
            return float(json.loads(outputs[key].read_text())["threshold"])
    raise FileNotFoundError("no threshold yet: run python run.py final (Week 3)")


# PROVIDED
def model_test_counts(root: Path) -> dict[str, int]:
    """Outcome counts of the model tests from reports/test_report.json."""
    path = root / TEST_REPORT
    if not path.exists():
        raise FileNotFoundError("reports/test_report.json not found: python run.py testreport")
    counts = json.loads(path.read_text())["categories"]["model"]["counts"]
    return {key: int(counts.get(key, 0)) for key in ("passed", "failed", "xfail", "skipped")}


# PROVIDED
def evaluate_candidate(config: dict[str, Any]) -> dict[str, Any]:
    """Build, fit, and score the model the current code would ship."""
    final, seed = config["week3"]["final"], config["seed"]
    spec = model_spec(config)
    data = load_training_data(config)
    model = calibrate(make_model(spec, data.numeric, data.categorical, config),
                      final["calibration"], config["week3"]["cv_folds"], seed)
    model.fit(data.X_train, data.y_train)
    scores = model.predict_proba(data.X_valid)[:, 1]
    threshold = registered_threshold(config)
    root = config["root"]
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "code_fingerprint": code_fingerprint(root),
        "data_version": data.data_version,
        "model": spec,
        "n_features": len(data.features),
        **validation_metrics(data.y_valid, scores, config["week3"]["n_bootstrap"], seed),
        "threshold": threshold,
        **decision_summary(data.y_valid, scores, threshold, final["costs"]),
        "model_tests": model_test_counts(root),
    }


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.evaluate``."""
    config = load_week3_config()
    record = evaluate_candidate(config)
    path = config["root"] / GATE_METRICS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    tests = record["model_tests"]
    print(f"Validation AUC {record['valid_auc']:.3f} [{record['valid_auc_ci_low']:.3f}, "
          f"{record['valid_auc_ci_high']:.3f}], cost {record['cost_at_threshold']:.3f} per "
          f"applicant at threshold {record['threshold']:.3f}")
    print(f"Model tests: {tests['passed']} passed, {tests['failed']} failed, "
          f"{tests['xfail']} known failures, {tests['skipped']} skipped")
    print(f"Code fingerprint {record['code_fingerprint']}\nWrote {path}  (commit it)")


if __name__ == "__main__":
    main()
