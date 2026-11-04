"""Week 4 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` or ``--stage wed`` (what that checkpoint
needs), or ``python run.py check`` (everything the Week 4 submission needs). It runs your
tests itself, so it always reflects the code as it is now. Passing these checks shows your
tests exist and run; it does not grade how good they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

import yaml

from scripts.check_week2 import CheckFailed, check_sample, count_commit_authors
from scripts.quality_gate import GATE_CONFIG, THRESHOLDS, gate_failures, load_inputs, preconditions
from scripts.test_report import CATEGORIES, REPORT_JSON, build_report
from src.evaluate import GATE_METRICS
from src.experiment import load_week3_config
from src.utils import DEFAULT_CONFIG

FINDINGS = "docs/test_findings.md"
CI_REPORT = "docs/ci_report.md"
CLOUDBUILD = "cloudbuild.yaml"
STEP_COMMANDS = {"lint": "ruff", "test": "pytest", "gate": "quality_gate"}
MIN_TESTS = {"unit": 5, "data": 5, "model": 4}
STUB = "TODO(student)"


# DO NOT EDIT
def _report(config: dict[str, Any]) -> dict:
    if "_report" not in config:
        config["_report"] = build_report(config["root"])
    return config["_report"]


# DO NOT EDIT
def _tests(config: dict[str, Any], category: str) -> list[dict[str, str]]:
    tests = _report(config)["categories"][category]["tests"]
    if len(tests) < MIN_TESTS[category]:
        raise CheckFailed(f"{len(tests)} {category} tests in {CATEGORIES[category]}, "
                          f"need at least {MIN_TESTS[category]}")
    stubs = [t["name"] for t in tests if STUB in t["message"]]
    if stubs:
        raise CheckFailed(f"{len(stubs)} test(s) not written yet: {stubs[:3]}")
    return tests


# DO NOT EDIT
def _summary(tests: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for t in tests:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))


# DO NOT EDIT
def _documented(config: dict[str, Any], names: list[str]) -> list[str]:
    """The names in ``names`` that docs/test_findings.md does not mention."""
    path = config["root"] / FINDINGS
    text = path.read_text() if path.exists() else ""
    return [n for n in names if n not in text]


# DO NOT EDIT
def check_unit(config: dict[str, Any]) -> str:
    tests = _tests(config, "unit")
    failed = [t["name"] for t in tests if t["outcome"] == "failed"]
    if failed:
        raise CheckFailed(f"failing: {failed[:3]}")
    return _summary(tests)


# DO NOT EDIT
def check_data(config: dict[str, Any]) -> str:
    tests = _tests(config, "data")
    if any(t["outcome"] == "skipped" for t in tests):
        raise CheckFailed("data tests were skipped: is data/features/features.parquet here?")
    failed = [t["name"] for t in tests if t["outcome"] == "failed"]
    if failed:
        raise CheckFailed(f"failing: {failed[:3]}; fix the data or document and fix the test")
    return _summary(tests)


# DO NOT EDIT
def check_model_ran(config: dict[str, Any]) -> str:
    tests = _tests(config, "model")
    if all(t["outcome"] == "skipped" for t in tests):
        raise CheckFailed("every model test was skipped: is the data on this machine?")
    unexplained = _documented(config, [t["name"] for t in tests
                                       if t["outcome"] in ("failed", "xfail")])
    if unexplained:
        raise CheckFailed(f"failing tests not explained in {FINDINGS}: {unexplained[:3]}")
    return _summary(tests)


# DO NOT EDIT
def check_model_decided(config: dict[str, Any]) -> str:
    tests = _tests(config, "model")
    failed = [t["name"] for t in tests if t["outcome"] == "failed"]
    if failed:
        raise CheckFailed(f"still failing: {failed[:3]}. Fix the model, or mark the test "
                          "xfail(strict=True) with its finding in " + FINDINGS)
    return _summary(tests)


# DO NOT EDIT
def check_findings(config: dict[str, Any]) -> str:
    path = config["root"] / FINDINGS
    if not path.exists():
        raise CheckFailed(f"{FINDINGS} not found; copy docs/test_findings_template.md")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{FINDINGS} still contains TODO")
    return f"{len(text.split())} words"


# DO NOT EDIT
def check_report_file(config: dict[str, Any]) -> str:
    path = config["root"] / REPORT_JSON
    if not path.exists():
        raise CheckFailed("reports/test_report.json not found; python run.py testreport")
    report = json.loads(path.read_text())
    total = sum(len(c["tests"]) for c in report["categories"].values())
    return f"{total} tests, commit {report['git_commit']}"


# DO NOT EDIT
def check_cloudbuild(config: dict[str, Any]) -> str:
    path = config["root"] / CLOUDBUILD
    if not path.exists():
        raise CheckFailed(f"{CLOUDBUILD} not found in the repository root")
    steps = (yaml.safe_load(path.read_text()) or {}).get("steps") or []
    by_id = {step.get("id"): " ".join([str(step.get("entrypoint", "")),
                                       *map(str, step.get("args", []))]) for step in steps}
    ids = list(by_id)
    missing = [i for i in ("install", *STEP_COMMANDS) if i not in ids]
    if missing:
        raise CheckFailed(f"steps missing: {missing} (found {ids})")
    wrong = [i for i, word in STEP_COMMANDS.items() if word not in by_id[i]]
    if wrong:
        raise CheckFailed(f"steps that do not run their tool: {wrong}")
    order = [ids.index(i) for i in ("install", *STEP_COMMANDS)]
    if order != sorted(order):
        raise CheckFailed("steps must run in the order install, lint, test, gate")
    return " -> ".join(ids)


# DO NOT EDIT
def check_gate_config(config: dict[str, Any]) -> str:
    path = config["root"] / GATE_CONFIG
    if not path.exists():
        raise CheckFailed(f"{GATE_CONFIG} not found")
    thresholds = yaml.safe_load(path.read_text()) or {}
    unset = [name for name in THRESHOLDS if thresholds.get(name) is None]
    if unset:
        raise CheckFailed(f"thresholds not set: {unset}")
    return ", ".join(f"{name}={thresholds[name]}" for name in THRESHOLDS)


# DO NOT EDIT
def check_gate(config: dict[str, Any]) -> str:
    root = config["root"]
    if not (root / GATE_METRICS).exists():
        raise CheckFailed("reports/gate_metrics.json not found; python run.py evaluate")
    metrics, baseline_auc, thresholds = load_inputs(root)
    problems = preconditions(metrics, thresholds, root)
    if problems:
        raise CheckFailed(problems[0])
    failures = gate_failures(metrics, baseline_auc, thresholds)
    if failures:
        raise CheckFailed(f"the gate fails on your own model: {failures[0]}")
    return (f"passes: AUC {metrics['valid_auc']:.3f} (registered {baseline_auc:.3f}), cost "
            f"{metrics['cost_at_threshold']:.3f}")


# DO NOT EDIT
def check_ci_report(config: dict[str, Any]) -> str:
    path = config["root"] / CI_REPORT
    if not path.exists():
        raise CheckFailed(f"{CI_REPORT} not found; copy docs/ci_report_template.md")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{CI_REPORT} still contains TODO")
    unexplained = [name for name in THRESHOLDS if name not in text]
    if unexplained:
        raise CheckFailed(f"thresholds not justified in {CI_REPORT}: {unexplained}")
    if "console.cloud.google.com/cloud-build" not in text:
        raise CheckFailed(f"{CI_REPORT} needs the link to a Cloud Build run")
    return f"{len(text.split())} words, every threshold justified"


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("sample", check_sample),
    ("unit tests", check_unit),
    ("data tests", check_data),
    ("model tests run", check_model_ran),
    ("test findings", check_findings),
]
WEDNESDAY: list[Check] = [
    ("cloudbuild.yaml", check_cloudbuild),
    ("gate thresholds", check_gate_config),
    ("quality gate", check_gate),
]
SUBMISSION: list[Check] = MONDAY + [
    ("model decided", check_model_decided),
    ("test report", check_report_file),
] + WEDNESDAY + [
    ("ci report", check_ci_report),
]
STAGES = {"mon": MONDAY, "wed": WEDNESDAY, "all": SUBMISSION}


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 4 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=list(STAGES), default="all",
                        help="mon / wed: that checkpoint only; all: the full submission")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    checks = STAGES[args.stage]

    if args.stage != "wed":
        print("Running your unit, data, and model tests (about 20 seconds) ...\n", flush=True)
    failed = 0
    print(f"{'check':<18}{'result':<7}details")
    print("-" * 90)
    for name, check in checks:
        try:
            status, detail = "PASS", check(config)
        except CheckFailed as err:
            status, detail = "FAIL", str(err)
            failed += 1
        except Exception as err:  # a crash is a failure too, with the reason shown
            status, detail = "FAIL", f"{type(err).__name__}: {err}"
            failed += 1
        print(f"{name:<18}{status:<7}{detail}")
    print(f"{'git authors':<18}{'INFO':<7}{count_commit_authors(config['root'])}")
    print("-" * 90)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
