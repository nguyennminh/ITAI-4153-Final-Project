"""Week 7 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` (what the Monday checkpoint needs), or
``python run.py check`` (Monday's part of the submission; Wednesday's zip replaces this
script with the full Week 7 checks). It calls your
LIVE service at the URL in docs/deployment.md (Week 5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.check_week2 import CheckFailed, count_commit_authors
from scripts.check_week5 import check_live
from src.experiment import load_week3_config
from src.utils import DEFAULT_CONFIG

RETRAIN = Path("reports/retrain")


# DO NOT EDIT
def _pytest(config: dict[str, Any], *paths: str) -> str:
    result = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "-p",
                             "no:cacheprovider"], cwd=config["root"], capture_output=True,
                            text=True)
    last = (result.stdout.strip().splitlines() or ["no output"])[-1]
    if result.returncode != 0:
        raise CheckFailed(f"{' '.join(paths)}: {last}")
    return last.strip("= ")


# DO NOT EDIT
def _json(config: dict[str, Any], rel: Path, hint: str) -> dict[str, Any]:
    path = config["root"] / rel
    if not path.exists():
        raise CheckFailed(f"{rel} not found; {hint}")
    return json.loads(path.read_text())


# DO NOT EDIT
def check_mon_tests(config: dict[str, Any]) -> str:
    return _pytest(config, "tests/test_week7_mon.py")


# DO NOT EDIT
def check_outcomes(config: dict[str, Any]) -> str:
    record = _json(config, RETRAIN / "outcomes.json", "python run.py outcomes")
    if record.get("source") != "cloud_logging":
        raise CheckFailed("outcomes were not computed from the Cloud Logging download (Week 6)")
    batches = record["batches"]
    if not {"1", "2"} <= set(batches):
        raise CheckFailed(f"outcomes for batches {sorted(batches)}; batches 1 and 2 needed")
    costs = ", ".join(f"batch {b} {r['real']['cost']:.3f}" for b, r in batches.items())
    return f"cost per applicant: validation {record['validation']['cost']:.3f}, {costs}"


# DO NOT EDIT
def check_challenger(config: dict[str, Any]) -> str:
    record = _json(config, RETRAIN / "challenger.json", "python run.py retrain")
    version = str(record["registered"]["version"])
    try:
        from mlflow import MlflowClient

        from src.tracking import setup_mlflow

        setup_mlflow(config)
        name = config["week3"]["final"]["registry"]["name"]
        registered = MlflowClient().get_model_version(name, version)
    except Exception as err:  # noqa: BLE001  (no MLflow database, or no such version)
        raise CheckFailed(f"challenger version {version} is not in the MLflow registry "
                          f"({type(err).__name__})") from err
    return (f"version {version} ({registered.status}), trained on "
            f"{record['trained_on']['total']:,} applicants")


# DO NOT EDIT
def check_decision(config: dict[str, Any]) -> str:
    decision = _json(config, RETRAIN / "decision.json", "python run.py champion")
    if decision.get("decision") not in ("promote", "keep") or not decision.get("reasons"):
        raise CheckFailed("decision.json has no decision with reasons")
    if "model_tests" not in decision or "gate_failures" not in decision:
        raise CheckFailed("decision.json was not made with the gate: python run.py champion")
    gate = "gate passed" if not decision["gate_failures"] else \
        f"gate failed ({len(decision['gate_failures'])})"
    return f"{decision['decision'].upper()} (v{decision['challenger_version']} vs " \
           f"v{decision['champion_version']}), {gate}"


# DO NOT EDIT
def check_serving(config: dict[str, Any]) -> str:
    """The live service runs the champion that models/ was exported from."""
    return check_live(config)


# DO NOT EDIT
def _doc(config: dict[str, Any], rel: str, min_words: int, needed: tuple[str, ...]) -> str:
    path = config["root"] / rel
    if not path.exists():
        raise CheckFailed(f"{rel} not found; copy its template")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{rel} still contains TODO")
    words = len(text.split())
    if words < min_words:
        raise CheckFailed(f"{rel}: {words} words; answer every question")
    absent = [n for n in needed if n.lower() not in text.lower()]
    if absent:
        raise CheckFailed(f"{rel} does not mention {absent}")
    return f"{words} words"


# DO NOT EDIT
def check_decision_record(config: dict[str, Any]) -> str:
    return _doc(config, "docs/decision_record.md", 350, ("gate", "recent", "validation",
                                                          "rollback"))


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("mon tests", check_mon_tests),
    ("outcomes", check_outcomes),
    ("challenger", check_challenger),
    ("decision", check_decision),
    ("live service", check_serving),
]
SUBMISSION: list[Check] = MONDAY + [("decision record", check_decision_record)]
STAGES = {"mon": MONDAY, "all": SUBMISSION}


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 7 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=list(STAGES), default="all",
                        help="mon: the checkpoint only; all: Monday's part of the submission")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    os.chdir(config["root"])

    failed = 0
    print(f"{'check':<16}{'result':<7}details")
    print("-" * 90)
    for name, check in STAGES[args.stage]:
        try:
            status, detail = "PASS", check(config)
        except CheckFailed as err:
            status, detail = "FAIL", str(err)
            failed += 1
        except Exception as err:  # a crash is a failure too, with the reason shown
            status, detail = "FAIL", f"{type(err).__name__}: {err}"
            failed += 1
        print(f"{name:<16}{status:<7}{detail}")
    print(f"{'git authors':<16}{'INFO':<7}{count_commit_authors(config['root'])}")
    print("-" * 90)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
