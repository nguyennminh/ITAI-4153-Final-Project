"""Week 6 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` (what the Monday checkpoint needs), or
``python run.py check`` (everything Monday's part of the submission needs; Wednesday's zip
replaces this script with the full Week 6 checks). It calls your
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
from scripts.check_week5 import _service_url, check_live
from src.experiment import load_week3_config
from src.utils import DEFAULT_CONFIG

MODELS = Path("models")
PRODUCTION = Path("data/production")
MONITORING = Path("reports/monitoring")
MONITORING_REPORT = "docs/monitoring_report.md"
SENT_BATCHES = (1, 2)
MIN_LOGGED = 950


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
def check_mon_tests(config: dict[str, Any]) -> str:
    return _pytest(config, "tests/api", "tests/test_week6_mon.py")


# DO NOT EDIT
def _batch_ids(config: dict[str, Any], batch: int) -> list[int]:
    path = config["root"] / PRODUCTION / f"batch{batch}.jsonl"
    if not path.exists():
        raise CheckFailed(f"{path.relative_to(config['root'])} not found; python run.py pool")
    return [json.loads(line)["SK_ID_CURR"] for line in path.read_text().splitlines() if line]


# DO NOT EDIT
def check_pool(config: dict[str, Any]) -> str:
    import pandas as pd

    sizes = {b: len(_batch_ids(config, b)) for b in (1, 2, 3)}
    if set(sizes.values()) != {1000}:
        raise CheckFailed(f"batch sizes {sizes}; rerun python run.py pool")
    extension = config["root"] / PRODUCTION / "store_extension.parquet"
    if not extension.exists():
        raise CheckFailed("data/production/store_extension.parquet not found; python run.py pool")
    return f"3 batches of 1,000, {len(pd.read_parquet(extension, columns=['SK_ID_CURR'])):,} " \
           "rows for the feature store"


# DO NOT EDIT
def check_store(config: dict[str, Any]) -> str:
    import pandas as pd

    info = json.loads((config["root"] / MODELS / "model_info.json").read_text())
    ids = set(pd.read_parquet(config["root"] / MODELS / "feature_store.parquet",
                              columns=["SK_ID_CURR"])["SK_ID_CURR"])
    missing = [i for b in (1, 2, 3) for i in _batch_ids(config, b) if i not in ids]
    if missing:
        raise CheckFailed(f"{len(missing)} production applicants are not in models/"
                          "feature_store.parquet: python run.py export (after pool)")
    return f"{info['n_applicants']:,} applicants in the exported feature store"


# DO NOT EDIT
def check_live_store(config: dict[str, Any]) -> str:
    import httpx

    detail = check_live(config)
    url = _service_url(config)
    applicant = _batch_ids(config, 3)[0]  # never sent by you, so it does not pollute the log
    with httpx.Client(timeout=90) as client:
        answer = client.post(f"{url}/predict", json={"SK_ID_CURR": applicant})
    if answer.status_code != 200:
        raise CheckFailed(f"the live service answers HTTP {answer.status_code} for production "
                          f"applicant {applicant}: upload the new bundle and redeploy")
    return f"{detail}; production applicants known"


# DO NOT EDIT
def check_sent(config: dict[str, Any]) -> str:
    url = _service_url(config)
    parts = []
    for batch in SENT_BATCHES:
        path = config["root"] / "reports" / f"sent_batch{batch}.json"
        if not path.exists():
            raise CheckFailed(f"{path.name} not found; python run.py send --batch {batch} "
                              "--url <your service>")
        receipt = json.loads(path.read_text())
        if receipt["url"].rstrip("/") != url:
            raise CheckFailed(f"batch {batch} went to {receipt['url']}, not the live service")
        if receipt["ok"] < receipt["sent"]:
            raise CheckFailed(f"batch {batch}: only {receipt['ok']} of {receipt['sent']} "
                              "answered 200")
        parts.append(f"batch {batch}: {receipt['ok']:,} OK")
    return ", ".join(parts)


# DO NOT EDIT
def check_monitoring(config: dict[str, Any]) -> str:
    path = config["root"] / MONITORING / "summary.json"
    if not path.exists():
        raise CheckFailed(f"{MONITORING}/summary.json not found; python run.py monitor")
    summary = json.loads(path.read_text())
    if summary["source"] != "cloud_logging":
        raise CheckFailed("the report was built from a local file: download the LIVE "
                          "service's log from Cloud Logging (WEEK6.md) and run monitor again")
    batches = summary["batches"]
    for batch in SENT_BATCHES:
        n = batches.get(str(batch), {}).get("requests", 0)
        if n < MIN_LOGGED:
            raise CheckFailed(f"only {n} logged predictions for batch {batch}: was the whole "
                              "log downloaded (--limit, --freshness)?")
    major = {b: rec["features_major"] for b, rec in batches.items()}
    return f"{summary['records']:,} live predictions; features with major drift per batch {major}"


# DO NOT EDIT
def _doc(config: dict[str, Any], rel: str, min_words: int, needed: tuple[str, ...]) -> str:
    path = config["root"] / rel
    template = rel.replace(".md", "_template.md")
    if not path.exists():
        raise CheckFailed(f"{rel} not found; copy {template}")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{rel} still contains TODO")
    words = len(text.split())
    if words < min_words:
        raise CheckFailed(f"{words} words; answer every question")
    lower = text.lower()
    absent = [n for n in needed if n.lower() not in lower]
    if absent:
        raise CheckFailed(f"{rel} does not mention {absent}")
    return f"{words} words"


# DO NOT EDIT
def check_monitoring_report(config: dict[str, Any]) -> str:
    return _doc(config, MONITORING_REPORT, 400, ("PSI", "KS", "batch 2", "alert"))


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("logging tests", check_mon_tests),
    ("production pool", check_pool),
    ("feature store", check_store),
    ("live service", check_live_store),
    ("batches sent", check_sent),
    ("monitoring", check_monitoring),
]
SUBMISSION: list[Check] = MONDAY + [("monitoring rep.", check_monitoring_report)]
STAGES = {"mon": MONDAY, "all": SUBMISSION}


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 6 checks.")
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
