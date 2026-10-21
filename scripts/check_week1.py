"""Week 1 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check`` or ``python -m scripts.check_week1``.
Passing these checks shows your files exist and look plausible. It does not grade them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.make_sample import sample_hash
from src.utils import DEFAULT_CONFIG, load_config

PLACEHOLDER = "TODO(student)"
METRICS_KEYS = (
    "model", "roc_auc", "pr_auc", "ks", "n_train", "n_valid", "valid_default_rate", "seed",
)
DOCS = ("problem.md", "data_card.md")


class CheckFailed(Exception):
    """Raised by a check with a short explanation."""


# DO NOT EDIT
def check_sample_files(config: dict[str, Any]) -> str:
    sample_dir = config["paths"]["sample_dir"]
    missing = [f for f in config["tables"].values() if not (sample_dir / f).exists()]
    if missing:
        raise CheckFailed(f"missing {missing}; run python run.py sample")
    return f"all {len(config['tables'])} tables in {sample_dir.name}/"


# DO NOT EDIT
def _sample_ids(config: dict[str, Any]) -> pd.Series:
    app = config["paths"]["sample_dir"] / config["tables"]["application_train"]
    if not app.exists():
        raise CheckFailed("sampled application_train.csv not found")
    return pd.read_csv(app, usecols=[config["id_col"]])[config["id_col"]]


# DO NOT EDIT
def check_sample_size(config: dict[str, Any]) -> str:
    n = len(_sample_ids(config))
    if n != config["sample_size"]:
        raise CheckFailed(f"{n} applicants, expected {config['sample_size']}")
    return f"{n:,} applicants"


# DO NOT EDIT
def check_sample_hash(config: dict[str, Any]) -> str:
    expected = config.get("expected_sample_hash")
    actual = sample_hash(_sample_ids(config).to_numpy())
    if not expected:
        return f"no expected hash set yet (yours: {actual[:12]}...)"
    if actual != expected:
        raise CheckFailed("sample differs from the instructor's; restore make_sample.py and "
                          "config.yaml, then run python run.py sample --force")
    return "matches the instructor's sample"


# DO NOT EDIT
def check_table_summary(config: dict[str, Any]) -> str:
    path = config["paths"]["reports_dir"] / "table_summary.json"
    if not path.exists():
        raise CheckFailed("reports/table_summary.json not found; run python run.py summary")
    rows = json.loads(path.read_text())
    names = sorted(r.get("table") for r in rows)
    if names != sorted(config["tables"]):
        raise CheckFailed(f"expected the {len(config['tables'])} tables, found {names}")
    bigger = [r["table"] for r in rows if r["mb_after"] > r["mb_before"]]
    if bigger:
        raise CheckFailed(f"mb_after is larger than mb_before for {bigger}")
    return f"{len(rows)} tables, memory shrinks for each"


# DO NOT EDIT
def check_split_file(config: dict[str, Any]) -> str:
    path = config["paths"]["split_file"]
    id_col = config["id_col"]
    if not path.exists():
        raise CheckFailed("config/split_ids.csv not found; restore it from the template")
    split = pd.read_csv(path)
    if list(split.columns) != [id_col, "split"]:
        raise CheckFailed(f"columns should be [{id_col}, split]")
    if split[id_col].duplicated().any():
        raise CheckFailed("duplicate ids")
    if set(split["split"]) != {"train", "valid"}:
        raise CheckFailed("split values must be exactly train and valid")
    if set(split[id_col]) != set(_sample_ids(config)):
        raise CheckFailed("ids do not match the sampled applicants")
    counts = split["split"].value_counts()
    return f"{counts['train']:,} train / {counts['valid']:,} valid"


# DO NOT EDIT
def check_metrics(config: dict[str, Any]) -> str:
    path = config["paths"]["metrics_file"]
    if not path.exists():
        raise CheckFailed("reports/metrics.json not found; run python run.py baseline")
    metrics = json.loads(path.read_text())
    missing = [k for k in METRICS_KEYS if k not in metrics]
    if missing:
        raise CheckFailed(f"missing keys {missing}")
    if not 0.5 <= metrics["roc_auc"] <= 1.0:
        raise CheckFailed(f"roc_auc {metrics['roc_auc']} is outside [0.5, 1]")
    if not 0.0 <= metrics["pr_auc"] <= 1.0:
        raise CheckFailed(f"pr_auc {metrics['pr_auc']} is outside [0, 1]")
    if not 0.0 <= metrics["ks"] <= 1.0:
        raise CheckFailed(f"ks {metrics['ks']} is outside [0, 1]")
    if metrics["n_train"] + metrics["n_valid"] != config["sample_size"]:
        raise CheckFailed("n_train + n_valid does not equal the sample size")
    return (f"roc_auc {metrics['roc_auc']:.3f}, pr_auc {metrics['pr_auc']:.3f}, "
            f"ks {metrics['ks']:.3f}")


# DO NOT EDIT
def check_docs(config: dict[str, Any]) -> str:
    docs_dir = config["root"] / "docs"
    problems = []
    for name in DOCS:
        path = docs_dir / name
        if not path.exists():
            problems.append(f"{name} missing (copy it from the template)")
        elif PLACEHOLDER in path.read_text():
            problems.append(f"{name} still has {PLACEHOLDER} placeholders")
    if problems:
        raise CheckFailed("; ".join(problems))
    return "problem.md and data_card.md filled in"


# DO NOT EDIT
def count_commit_authors(root: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "--format=%ae"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "not a git repository (or no commits yet)"
    authors = {line.strip().lower() for line in out.splitlines() if line.strip()}
    return f"{len(authors)} distinct commit author(s)"


REQUIRED_CHECKS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("sample files", check_sample_files),
    ("sample size", check_sample_size),
    ("sample hash", check_sample_hash),
    ("table summary", check_table_summary),
    ("split file", check_split_file),
    ("metrics.json", check_metrics),
    ("docs", check_docs),
]


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 1 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    failed = 0
    print(f"{'check':<16}{'result':<7}details")
    print("-" * 78)
    for name, check in REQUIRED_CHECKS:
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
    print("-" * 78)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
