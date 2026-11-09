"""Week 5 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` (what the Monday checkpoint needs) or
``python run.py check`` (Monday plus the design note). It starts your API on your real
exported model (no server needed) and sends it requests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.check_week2 import CheckFailed, check_sample, count_commit_authors
from src.experiment import load_week3_config
from src.utils import DEFAULT_CONFIG

MODELS = Path("models")
DESIGN = "docs/api_design.md"
BUNDLE = ("model.joblib", "model_info.json", "feature_store.parquet")


# DO NOT EDIT
def check_bundle(config: dict[str, Any]) -> str:
    root = config["root"]
    missing = [name for name in BUNDLE if not (root / MODELS / name).exists()]
    if missing:
        raise CheckFailed(f"models/ is missing {missing}; python run.py export")
    info = json.loads((root / MODELS / "model_info.json").read_text())
    final = json.loads(config["week3"]["outputs"]["final_model"].read_text())
    if str(final["registered"]["version"]) != info["version"]:
        raise CheckFailed(f"models/ holds version {info['version']} but version "
                          f"{final['registered']['version']} is registered; export again")
    pins = (root / "requirements-api.txt")
    if not pins.exists():
        raise CheckFailed("requirements-api.txt not found; python run.py export")
    return (f"version {info['version']}, {info['n_features']} features, "
            f"{info['n_applicants']:,} applicants")


# DO NOT EDIT
def check_api_tests(config: dict[str, Any]) -> str:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/api", "-q", "-p",
                             "no:cacheprovider"], cwd=config["root"], capture_output=True,
                            text=True)
    last = (result.stdout.strip().splitlines() or ["no output"])[-1]
    if result.returncode != 0:
        raise CheckFailed(f"tests/api: {last}")
    return last.strip("= ")


# DO NOT EDIT
def _client(config: dict[str, Any]):
    with warnings.catch_warnings():  # newer Starlette warns about its own test client
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

    from api.main import app

    os.environ.pop("MODEL_BUCKET", None)
    os.environ["MODEL_DIR"] = str(config["root"] / MODELS)
    return TestClient(app)


# DO NOT EDIT
def check_real_model(config: dict[str, Any]) -> str:
    import pandas as pd

    known = int(pd.read_parquet(config["root"] / MODELS / "feature_store.parquet")
                ["SK_ID_CURR"].iloc[0])
    with _client(config) as client:
        health = client.get("/health").json()
        if health.get("status") != "ok":
            raise CheckFailed(f"/health says {health}")
        ok = client.post("/predict", json={"SK_ID_CURR": known})
        if ok.status_code != 200:
            raise CheckFailed(f"/predict for applicant {known}: HTTP {ok.status_code} "
                              f"{ok.text[:120]}")
        bad = client.post("/predict", json={"SK_ID_CURR": known, "AMT_INCOME_TOTAL": -1})
        unknown = client.post("/predict", json={"SK_ID_CURR": 1})
        info = client.get("/model-info")
    if bad.status_code != 422:
        raise CheckFailed(f"a negative income gave HTTP {bad.status_code}, expected 422")
    if unknown.status_code != 404:
        raise CheckFailed(f"an unknown applicant gave HTTP {unknown.status_code}, expected 404")
    if info.status_code != 200:
        raise CheckFailed(f"/model-info: HTTP {info.status_code}")
    body = ok.json()
    return (f"applicant {known}: probability {body['probability']:.3f} -> {body['decision']}; "
            "invalid 422, unknown 404")


# DO NOT EDIT
def check_design(config: dict[str, Any]) -> str:
    path = config["root"] / DESIGN
    if not path.exists():
        raise CheckFailed(f"{DESIGN} not found; copy docs/api_design_template.md")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{DESIGN} still contains TODO")
    words = len(text.split())
    if words < 150:
        raise CheckFailed(f"{words} words; answer every question")
    return f"{words} words"


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("sample", check_sample),
    ("model bundle", check_bundle),
    ("api tests", check_api_tests),
    ("real model", check_real_model),
]
STAGES = {"mon": MONDAY, "all": MONDAY + [("design note", check_design)]}


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 5 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=list(STAGES), default="all",
                        help="mon: the Monday checkpoint only; all: everything so far")
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
