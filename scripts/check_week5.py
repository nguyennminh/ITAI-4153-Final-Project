"""Week 5 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` or ``--stage wed`` (what that checkpoint
needs), or ``python run.py check`` (everything the Week 5 submission needs). It starts your
API on your real exported model (no server needed), and it calls your LIVE service at the
URL in docs/deployment.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEPLOYMENT = "docs/deployment.md"
API_DOCS = "docs/api.md"
LATENCY = "reports/latency.json"
URL = re.compile(r"https://[a-z0-9.-]+\.run\.app")
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


# DO NOT EDIT
def _instructions(config: dict[str, Any]) -> list[str]:
    path = config["root"] / "Dockerfile"
    if not path.exists():
        raise CheckFailed("Dockerfile not found in the repository root")
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


# DO NOT EDIT
def check_dockerfile(config: dict[str, Any]) -> str:
    lines = _instructions(config)
    base = next((line.split()[1] for line in lines if line.startswith("FROM")), "")
    if "slim" not in base:
        raise CheckFailed(f"base image {base!r} is not a slim image")
    pins = (config["root"] / "requirements-api.txt").read_text()
    wanted = re.search(r"Python (\d+\.\d+)", pins)
    if wanted and f"python:{wanted.group(1)}" not in base:
        raise CheckFailed(f"FROM {base}, but the model was exported with Python "
                          f"{wanted.group(1)}: use python:{wanted.group(1)}-slim")
    if not any("-r requirements-api.txt" in line for line in lines):
        raise CheckFailed("the Dockerfile does not install requirements-api.txt")
    users = [line.split()[1] for line in lines if line.startswith("USER")]
    if not users or users[-1] in ("root", "0"):
        raise CheckFailed("the container must run as a user without root rights (USER)")
    cmd = next((line for line in reversed(lines) if line.startswith(("CMD", "ENTRYPOINT"))), "")
    if "uvicorn" not in cmd or "PORT" not in cmd:
        raise CheckFailed("CMD must start uvicorn on $PORT")
    return f"{base}, user {users[-1]}, pinned requirements"


# DO NOT EDIT
def check_dockerignore(config: dict[str, Any]) -> str:
    path = config["root"] / ".dockerignore"
    if not path.exists():
        raise CheckFailed(".dockerignore not found")
    entries = {line.strip().rstrip("/") for line in path.read_text().splitlines()}
    missing = [e for e in ("data", "models", ".venv") if e not in entries]
    if missing:
        raise CheckFailed(f".dockerignore must exclude {missing}")
    return "data, models, .venv never go into the image"


# DO NOT EDIT
def _service_url(config: dict[str, Any]) -> str:
    path = config["root"] / DEPLOYMENT
    if not path.exists():
        raise CheckFailed(f"{DEPLOYMENT} not found; copy docs/deployment_template.md")
    found = URL.search(path.read_text())
    if not found:
        raise CheckFailed(f"no https://...run.app address in {DEPLOYMENT}")
    return found.group(0)


# DO NOT EDIT
def check_live(config: dict[str, Any]) -> str:
    import httpx
    import pandas as pd

    url = _service_url(config)
    info = json.loads((config["root"] / MODELS / "model_info.json").read_text())
    known = int(pd.read_parquet(config["root"] / MODELS / "feature_store.parquet")
                ["SK_ID_CURR"].iloc[0])
    try:
        with httpx.Client(timeout=90) as client:
            health = client.get(f"{url}/health")
            answer = client.post(f"{url}/predict", json={"SK_ID_CURR": known})
            bad = client.post(f"{url}/predict", json={"SK_ID_CURR": "x"})
    except httpx.HTTPError as err:
        raise CheckFailed(f"{url} did not answer: {err}") from err
    if health.status_code != 200 or health.json().get("status") != "ok":
        raise CheckFailed(f"{url}/health: HTTP {health.status_code} {health.text[:100]}")
    if health.json().get("model_version") != info["version"]:
        raise CheckFailed(f"live model version {health.json().get('model_version')} != "
                          f"exported version {info['version']}: upload the new bundle and "
                          "redeploy")
    if answer.status_code != 200:
        raise CheckFailed(f"{url}/predict: HTTP {answer.status_code} {answer.text[:100]}")
    if bad.status_code != 422:
        raise CheckFailed(f"invalid input gave HTTP {bad.status_code} on the live service")
    return f"{url}: version {info['version']}, {answer.json()['decision']} for {known}"


# DO NOT EDIT
def check_latency(config: dict[str, Any]) -> str:
    path = config["root"] / LATENCY
    if not path.exists():
        raise CheckFailed(f"{LATENCY} not found; python run.py latency --url <your service>")
    record = json.loads(path.read_text())
    if record["url"].rstrip("/") != _service_url(config):
        raise CheckFailed(f"{LATENCY} measured {record['url']}, not the live service")
    if record["n"] < 50 or record["errors"]:
        raise CheckFailed(f"{record['n']} requests, {record['errors']} errors: measure "
                          "at least 50 without errors")
    return (f"p50 {record['p50_ms']:.0f} ms, p95 {record['p95_ms']:.0f} ms, first request "
            f"{record['first_request_ms']:.0f} ms")


# DO NOT EDIT
def check_api_docs(config: dict[str, Any]) -> str:
    path = config["root"] / API_DOCS
    if not path.exists():
        raise CheckFailed(f"{API_DOCS} not found; copy docs/api_template.md")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{API_DOCS} still contains TODO")
    for needed in ("/health", "/model-info", "/predict", "422", "404", "p95", "batch"):
        if needed not in text:
            raise CheckFailed(f"{API_DOCS} does not mention {needed}")
    return f"{len(text.split())} words"


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("sample", check_sample),
    ("model bundle", check_bundle),
    ("api tests", check_api_tests),
    ("real model", check_real_model),
]
WEDNESDAY: list[Check] = [
    ("dockerfile", check_dockerfile),
    (".dockerignore", check_dockerignore),
    ("live service", check_live),
]
SUBMISSION: list[Check] = MONDAY + [("design note", check_design)] + WEDNESDAY + [
    ("latency", check_latency),
    ("api docs", check_api_docs),
]
STAGES = {"mon": MONDAY, "wed": WEDNESDAY, "all": SUBMISSION}


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 5 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=list(STAGES), default="all",
                        help="mon / wed: that checkpoint only; all: the full submission")
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
