"""Week 2 checks (Monday checkpoint). Prints a PASS/FAIL table and exits non-zero on a failure.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check`` or ``python -m scripts.check_week2``.
Passing these checks shows your files exist and look plausible. It does not grade them.
The INFO rows track the carry-over work; they are not required for this checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.make_sample import sample_hash
from src.data import load_application, load_table
from src.features.bureau import BUREAU_BALANCE_FEATURES, aggregate_bureau, aggregate_bureau_balance
from src.features.common import SENTINEL, load_week2_config
from src.features.join import join_features
from src.features.previous import (
    POS_CASH_FEATURES,
    PREVIOUS_FEATURES,
    aggregate_pos_cash,
    aggregate_previous,
)
from src.utils import DEFAULT_CONFIG

REQUIRED_SETS = ("application", "application+bureau+previous")
MIN_LEAKAGE_ROWS = 8
ANSWERS = {"yes", "no"}


class CheckFailed(Exception):
    """Raised by a check with a short explanation."""


# DO NOT EDIT
def check_sample(config: dict[str, Any]) -> str:
    ids = load_application(config)[config["id_col"]]
    if len(ids) != config["sample_size"]:
        raise CheckFailed(f"{len(ids)} applicants, expected {config['sample_size']}")
    expected = config.get("expected_sample_hash")
    if expected and sample_hash(ids.to_numpy()) != expected:
        raise CheckFailed("sample differs from the instructor's; see python run.py check1")
    return f"{len(ids):,} applicants, same sample as the instructor"


# DO NOT EDIT
def check_week1_metrics(config: dict[str, Any]) -> str:
    path = config["paths"]["metrics_file"]
    if not path.exists():
        raise CheckFailed("reports/metrics.json not found; finish Week 1 (python run.py baseline)")
    return f"Week 1 roc_auc {json.loads(path.read_text())['roc_auc']:.3f}"


# DO NOT EDIT
def _unique_ids(df: pd.DataFrame, config: dict[str, Any], table: str) -> None:
    id_col = config["id_col"]
    if id_col not in df.columns:
        raise CheckFailed(f"{table}: {id_col} must be a column, not the index")
    if df[id_col].duplicated().any():
        raise CheckFailed(f"{table}: {int(df[id_col].duplicated().sum())} duplicated ids")


# DO NOT EDIT
def check_previous(config: dict[str, Any]) -> str:
    prev = aggregate_previous(load_table("previous_application", config), config["id_col"])
    _unique_ids(prev, config, "aggregate_previous")
    missing = [c for c in PREVIOUS_FEATURES if c not in prev.columns]
    if missing:
        raise CheckFailed(f"missing columns {missing}")
    for col in ("PREV_APPROVED_RATE", "PREV_REFUSED_RATE"):
        if not prev[col].between(0, 1).all():
            raise CheckFailed(f"{col} has values outside [0, 1]")
    if (prev["PREV_DAYS_TERMINATION_MEAN"] > 0).any():
        raise CheckFailed(f"PREV_DAYS_TERMINATION_MEAN is positive: is {SENTINEL} still in it?")
    return f"{len(prev):,} applicants, {prev.shape[1] - 1} PREV_ features"


# DO NOT EDIT
def check_join(config: dict[str, Any]) -> str:
    id_col = config["id_col"]
    app = load_application(config)
    tables = {
        "bureau": aggregate_bureau(load_table("bureau", config), id_col),
        "previous_application": aggregate_previous(
            load_table("previous_application", config), id_col),
    }
    joined = join_features(app, tables, id_col)
    if len(joined) != len(app):
        raise CheckFailed(f"{len(joined):,} rows for {len(app):,} applicants")
    if joined[id_col].tolist() != app[id_col].tolist():
        raise CheckFailed("applicant ids or their order changed in the join")
    no_bureau = ~app[id_col].isin(tables["bureau"][id_col])
    if joined.loc[no_bureau, "BUREAU_COUNT"].isna().any():
        raise CheckFailed("BUREAU_COUNT is NaN for applicants without bureau records")
    return f"{len(joined):,} rows x {joined.shape[1]} columns, one row per applicant"


# DO NOT EDIT
def check_comparison(config: dict[str, Any]) -> str:
    path = config["week2"]["paths"]["comparison_file"]
    if not path.exists():
        raise CheckFailed("reports/auc_comparison.json not found; run python run.py compare")
    rows = {r["feature_set"]: r for r in json.loads(path.read_text())["results"]}
    missing = [name for name in REQUIRED_SETS if name not in rows]
    if missing:
        raise CheckFailed(f"feature sets missing: {missing}")
    week1 = json.loads(config["paths"]["metrics_file"].read_text())["roc_auc"]
    base = rows["application"]["roc_auc"]
    if abs(base - week1) > 1e-6:
        raise CheckFailed(f"'application' roc_auc {base:.4f} differs from metrics.json "
                          f"{week1:.4f}; rerun python run.py baseline, then compare")
    new = rows["application+bureau+previous"]
    low, high = new["delta_ci"]
    return f"{base:.3f} -> {new['roc_auc']:.3f} (change {new['delta_roc_auc']:+.3f}, " \
           f"95% CI [{low:+.3f}, {high:+.3f}])"


# DO NOT EDIT
def check_leakage_table(config: dict[str, Any]) -> str:
    path: Path = config["week2"]["paths"]["leakage_table"]
    if not path.exists():
        raise CheckFailed("docs/leakage_table.md not found; copy docs/leakage_table_template.md")
    answered = 0
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"[A-Z][A-Z0-9_]+", cells[0]):
            continue
        reason = cells[3]
        if cells[2].lower() in ANSWERS and reason and "TODO" not in reason:
            answered += 1
    if answered < MIN_LEAKAGE_ROWS:
        raise CheckFailed(f"{answered} features answered (Yes/No plus a reason); "
                          f"need at least {MIN_LEAKAGE_ROWS} to start")
    return f"{answered} features answered"


# DO NOT EDIT
def carry_over(run: Callable[[], pd.DataFrame], columns: list[str]) -> str:
    """INFO only: report whether a carry-over aggregation works yet."""
    try:
        out = run()
    except NotImplementedError:
        return "not started"
    except Exception as err:
        return f"error: {type(err).__name__}: {err}"
    missing = [c for c in columns if c not in out.columns]
    if missing:
        return f"runs, but missing columns {missing}"
    return f"runs: {len(out):,} applicants"


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
    ("sample", check_sample),
    ("week 1 baseline", check_week1_metrics),
    ("previous agg", check_previous),
    ("join", check_join),
    ("auc comparison", check_comparison),
    ("leakage table", check_leakage_table),
]


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 2 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_week2_config(args.config)
    id_col = config["id_col"]

    failed = 0
    print(f"{'check':<18}{'result':<7}details")
    print("-" * 80)
    for name, check in REQUIRED_CHECKS:
        try:
            status, detail = "PASS", check(config)
        except CheckFailed as err:
            status, detail = "FAIL", str(err)
            failed += 1
        except NotImplementedError as err:
            status, detail = "FAIL", f"not written yet ({err})"
            failed += 1
        except Exception as err:  # a crash is a failure too, with the reason shown
            status, detail = "FAIL", f"{type(err).__name__}: {err}"
            failed += 1
        print(f"{name:<18}{status:<7}{detail}")

    bb = carry_over(lambda: aggregate_bureau_balance(
        load_table("bureau_balance", config), load_table("bureau", config), id_col),
        BUREAU_BALANCE_FEATURES)
    pos = carry_over(lambda: aggregate_pos_cash(
        load_table("POS_CASH_balance", config), id_col), POS_CASH_FEATURES)
    print(f"{'bureau_balance':<18}{'INFO':<7}carry-over: {bb}")
    print(f"{'POS_CASH':<18}{'INFO':<7}carry-over: {pos}")
    print(f"{'git authors':<18}{'INFO':<7}{count_commit_authors(config['root'])}")
    print("-" * 80)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
