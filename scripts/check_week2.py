"""Week 2 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check`` (everything the Week 2 submission needs) or
``python run.py check --stage wed`` (only what the Wednesday checkpoint needs).
Passing these checks shows your files exist and look plausible. It does not grade them.
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
from src.features.bureau import BUREAU_BALANCE_FEATURES, aggregate_bureau_balance
from src.features.common import load_week2_config
from src.features.payments import (
    CREDIT_CARD_FEATURES,
    INSTALLMENTS_FEATURES,
    aggregate_credit_card,
    aggregate_installments,
)
from src.features.previous import POS_CASH_FEATURES, aggregate_pos_cash
from src.make_features import EXPECTED_FEATURES, build_feature_table, load_tables
from src.utils import DEFAULT_CONFIG
from src.validate import DataValidationError

CORRUPT_DIR = "data/corrupt"
ANSWERS = {"yes", "no"}
FEATURE_NAME = re.compile(r"[A-Z][A-Z0-9_]+")


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
def _aggregate_check(run: Callable[[], pd.DataFrame], columns: list[str],
                     config: dict[str, Any]) -> str:
    out = run()
    id_col = config["id_col"]
    if id_col not in out.columns or out[id_col].duplicated().any():
        raise CheckFailed(f"must return one row per applicant with {id_col} as a column")
    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise CheckFailed(f"missing columns {missing}")
    return f"{len(out):,} applicants"


# DO NOT EDIT
def check_bureau_balance(config: dict[str, Any]) -> str:
    return _aggregate_check(lambda: aggregate_bureau_balance(
        load_table("bureau_balance", config), load_table("bureau", config), config["id_col"]),
        BUREAU_BALANCE_FEATURES, config)


# DO NOT EDIT
def check_pos_cash(config: dict[str, Any]) -> str:
    return _aggregate_check(lambda: aggregate_pos_cash(
        load_table("POS_CASH_balance", config), config["id_col"]), POS_CASH_FEATURES, config)


# DO NOT EDIT
def check_installments(config: dict[str, Any]) -> str:
    return _aggregate_check(lambda: aggregate_installments(
        load_table("installments_payments", config), config["id_col"]),
        INSTALLMENTS_FEATURES, config)


# DO NOT EDIT
def check_credit_card(config: dict[str, Any]) -> str:
    return _aggregate_check(lambda: aggregate_credit_card(
        load_table("credit_card_balance", config), config["id_col"]),
        CREDIT_CARD_FEATURES, config)


# DO NOT EDIT
def check_clean_build(config: dict[str, Any]) -> str:
    try:
        features = build_feature_table(load_tables(config, config["paths"]["sample_dir"]),
                                       config)
    except DataValidationError as err:
        raise CheckFailed(f"validation fails on the clean sample: {err.problems[:3]}") from err
    if len(features) != config["sample_size"]:
        raise CheckFailed(f"{len(features):,} rows, expected {config['sample_size']:,}")
    return f"{len(features):,} rows x {features.shape[1]} columns, validation passes"


# DO NOT EDIT
def check_feature_file(config: dict[str, Any]) -> str:
    path = config["week2"]["paths"]["features_dir"] / "features.parquet"
    if not path.exists():
        raise CheckFailed("data/features/features.parquet not found; run python run.py features")
    table = pd.read_parquet(path)
    missing = [c for c in EXPECTED_FEATURES if c not in table.columns]
    if missing:
        raise CheckFailed(f"missing columns {missing[:5]}; rerun python run.py features")
    report = config["week2"]["paths"]["reports_dir"] / "validation_sample.json"
    if not report.exists() or json.loads(report.read_text())["status"] != "pass":
        raise CheckFailed("reports/validation_sample.json missing or not 'pass'")
    return f"features.parquet {len(table):,} rows, validation_sample.json: pass"


# DO NOT EDIT
def check_corrupt(config: dict[str, Any]) -> str:
    corrupt_dir = config["root"] / CORRUPT_DIR
    if not corrupt_dir.exists():
        raise CheckFailed(f"{CORRUPT_DIR}/ not found (it is created in class)")
    try:
        build_feature_table(load_tables(config, corrupt_dir), config)
    except DataValidationError as err:
        report = config["week2"]["paths"]["reports_dir"] / "validation_corrupt.json"
        if not report.exists():
            raise CheckFailed("run python run.py features --sample-dir data/corrupt to write "
                              "reports/validation_corrupt.json") from err
        return f"fails loudly with {len(err.problems)} problem(s) listed"
    except Exception as err:
        raise CheckFailed(f"crashed with {type(err).__name__} instead of listing the problems "
                          f"({err})") from err
    raise CheckFailed("the corrupted data passed validation")


# DO NOT EDIT
def check_comparison(config: dict[str, Any]) -> str:
    path = config["week2"]["paths"]["comparison_file"]
    if not path.exists():
        raise CheckFailed("reports/auc_comparison.json not found; run python run.py compare")
    rows = {r["feature_set"]: r for r in json.loads(path.read_text())["results"]}
    missing = [name for name in config["week2"]["feature_sets"] if name not in rows]
    if missing:
        raise CheckFailed(f"feature sets missing: {missing}; rerun python run.py compare")
    week1 = json.loads(config["paths"]["metrics_file"].read_text())["roc_auc"]
    if abs(rows["application"]["roc_auc"] - week1) > 1e-6:
        raise CheckFailed("'application' roc_auc differs from metrics.json; rerun baseline, "
                          "then compare")
    last = list(rows.values())[-1]
    low, high = last["delta_ci"]
    return (f"{rows['application']['roc_auc']:.3f} -> {last['roc_auc']:.3f} with everything "
            f"(change {last['delta_roc_auc']:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}])")


# DO NOT EDIT
def _table_rows(path: Path) -> dict[str, list[str]]:
    """Feature name -> its cells, for every markdown table row that starts with a feature."""
    rows = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and FEATURE_NAME.fullmatch(cells[0]):
            rows[cells[0]] = cells
    return rows


# DO NOT EDIT
def check_leakage_table(config: dict[str, Any]) -> str:
    path = config["week2"]["paths"]["leakage_table"]
    if not path.exists():
        raise CheckFailed("docs/leakage_table.md not found; copy docs/leakage_table_template.md")
    rows = _table_rows(path)
    unanswered = [f for f in EXPECTED_FEATURES if f not in rows
                  or rows[f][2].lower() not in ANSWERS or "TODO" in rows[f][3] or not rows[f][3]]
    if unanswered:
        raise CheckFailed(f"{len(unanswered)} feature(s) not answered, e.g. {unanswered[:3]}")
    return f"all {len(EXPECTED_FEATURES)} features answered"


# DO NOT EDIT
def check_feature_dictionary(config: dict[str, Any]) -> str:
    path = config["week2"]["paths"]["feature_dictionary"]
    if not path.exists():
        raise CheckFailed("docs/feature_dictionary.md not found; copy the template")
    rows = _table_rows(path)
    missing = [f for f in EXPECTED_FEATURES if f not in rows
               or any("TODO" in cell or not cell for cell in rows[f][1:4])]
    if missing:
        raise CheckFailed(f"{len(missing)} feature(s) not described, e.g. {missing[:3]}")
    return f"all {len(EXPECTED_FEATURES)} features described"


# DO NOT EDIT
def count_commit_authors(root: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "--format=%ae"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "not a git repository (or no commits yet)"
    authors = {line.strip().lower() for line in out.splitlines() if line.strip()}
    return f"{len(authors)} distinct commit author(s)"


Check = tuple[str, Callable[[dict[str, Any]], str]]
WEDNESDAY: list[Check] = [
    ("sample", check_sample),
    ("bureau_balance", check_bureau_balance),
    ("POS_CASH", check_pos_cash),
    ("installments", check_installments),
    ("credit card", check_credit_card),
    ("clean build", check_clean_build),
    ("feature file", check_feature_file),
    ("corrupt data", check_corrupt),
]
SUBMISSION: list[Check] = WEDNESDAY + [
    ("auc comparison", check_comparison),
    ("leakage table", check_leakage_table),
    ("feature dict", check_feature_dictionary),
]


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 2 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=["wed", "all"], default="all",
                        help="wed: the Wednesday checkpoint only; all: the full submission")
    args = parser.parse_args()
    config = load_week2_config(args.config)
    checks = WEDNESDAY if args.stage == "wed" else SUBMISSION

    failed = 0
    print(f"{'check':<16}{'result':<7}details")
    print("-" * 80)
    for name, check in checks:
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
        print(f"{name:<16}{status:<7}{detail}")
    print(f"{'git authors':<16}{'INFO':<7}{count_commit_authors(config['root'])}")
    print("-" * 80)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
