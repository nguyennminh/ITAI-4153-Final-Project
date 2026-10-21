"""Build the validated feature table, one row per applicant.

Run it with ``python run.py features`` or ``python -m src.make_features``. It:

1. reads the seven sampled tables,
2. validates them (``validate_inputs``),
3. aggregates each child table to one row per applicant,
4. joins everything onto the application table and adds the ratio features,
5. validates the result (``validate_features``), and
6. writes data/features/features.parquet and reports/validation_sample.json.

If any validation fails, it writes the report, prints every problem, writes **no** feature
table, and exits with code 1, so nothing downstream can use bad data by accident.

Options:
    --sample-dir PATH   read the tables from another folder (default: data/sample). The
                        outputs are then named after that folder, for example
                        reports/validation_<folder>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.bureau import (  # noqa: F401  (for you to use)
    BUREAU_BALANCE_FEATURES,
    aggregate_bureau,
    aggregate_bureau_balance,
)
from src.features.common import load_week2_config
from src.features.join import join_features  # noqa: F401
from src.features.payments import (  # noqa: F401
    CREDIT_CARD_FEATURES,
    INSTALLMENTS_FEATURES,
    aggregate_credit_card,
    aggregate_installments,
)
from src.features.previous import (  # noqa: F401
    POS_CASH_FEATURES,
    PREVIOUS_FEATURES,
    aggregate_pos_cash,
    aggregate_previous,
)
from src.features.ratios import RATIO_FEATURES, add_ratio_features  # noqa: F401
from src.utils import DEFAULT_CONFIG
from src.validate import DataValidationError, validate_features, validate_inputs  # noqa: F401

# PROVIDED: the columns aggregate_bureau creates (the other modules list theirs).
BUREAU_FEATURES = [
    "BUREAU_COUNT", "BUREAU_AMT_CREDIT_SUM_MEAN", "BUREAU_AMT_CREDIT_SUM_MAX",
    "BUREAU_AMT_DEBT_MEAN", "BUREAU_AMT_OVERDUE_SUM", "BUREAU_DAYS_OVERDUE_MAX",
    "BUREAU_DAYS_CREDIT_MAX", "BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT",
]
# Every engineered column the feature table must contain.
EXPECTED_FEATURES = (
    BUREAU_FEATURES + PREVIOUS_FEATURES + BUREAU_BALANCE_FEATURES + POS_CASH_FEATURES
    + INSTALLMENTS_FEATURES + CREDIT_CARD_FEATURES + RATIO_FEATURES
)


# PROVIDED
def load_tables(config: dict[str, Any], sample_dir: Path) -> dict[str, pd.DataFrame]:
    """Read all seven tables from ``sample_dir``. A missing file is a validation problem."""
    tables, missing = {}, []
    for name, filename in config["tables"].items():
        path = sample_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            missing.append(f"{name}: file {path} not found")
    if missing:
        raise DataValidationError("input", missing)
    return tables


# TODO(student)
def build_feature_table(tables: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    """Turn the seven raw tables into one validated table with one row per applicant.

    Args:
        tables: Table name -> raw DataFrame, as returned by ``load_tables``.
        config: The Week 2 config (``load_week2_config``).

    Returns:
        The application table's columns (unchanged, including TARGET) plus every column in
        ``EXPECTED_FEATURES``, one row per applicant, in the application table's order.

    Raises:
        DataValidationError: from ``validate_inputs`` before any aggregation, or from
            ``validate_features`` on the result.

    Hints:
        - Follow the six steps in this module's docstring; steps 2 to 5 are one call each
          once the six aggregations are written.
        - Validate the inputs FIRST. If you aggregate first, a renamed column crashes with a
          KeyError and you never see the rest of the problems.
        - Do not fill missing values, scale, or clean DAYS_EMPLOYED here. Those are fitted or
          applied in the model pipeline, on the train split only; doing them here would use
          validation applicants' data and would run twice.
    """

    # 1. Validate inputs before aggregation so all input problems are reported.
    validate_inputs(tables)

    # 2. Get the original application table.
    app = tables["application_train"]

    # 3. Aggregate each child table to one row per applicant.
    feature_tables = {
        "bureau": aggregate_bureau(tables["bureau"]),
        "previous_application": aggregate_previous(
            tables["previous_application"]
        ),
        "bureau_balance": aggregate_bureau_balance(
            tables["bureau_balance"],
            tables["bureau"],
        ),
        "POS_CASH_balance": aggregate_pos_cash(
            tables["POS_CASH_balance"]
        ),
        "installments_payments": aggregate_installments(
            tables["installments_payments"]
        ),
        "credit_card_balance": aggregate_credit_card(
            tables["credit_card_balance"]
        ),
    }

    # 4. Join all aggregated features onto the application table.
    features = join_features(app, feature_tables)

    # 5. Add the four ratio features.
    features = add_ratio_features(features)

    # 6. Validate the completed feature table.
    validate_features(
        features,
        EXPECTED_FEATURES,
        len(app),
        config,
    )

    return features


    raise NotImplementedError("TODO(student): build and validate the feature table")


# PROVIDED
def output_paths(config: dict[str, Any], sample_dir: Path) -> tuple[Path, Path]:
    """Where the feature table and the validation report go for this input folder."""
    name = sample_dir.name
    week2 = config["week2"]["paths"]
    table = week2["features_dir"] / ("features.parquet" if name == "sample"
                                     else f"features_{name}.parquet")
    report = week2["reports_dir"] / f"validation_{name}.json"
    return table, report


# PROVIDED
def write_report(path: Path, record: dict[str, Any]) -> None:
    """Write the validation report as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.make_features [--sample-dir PATH]``."""
    parser = argparse.ArgumentParser(description="Build the validated feature table.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--sample-dir", default=None, help="folder with the 7 tables")
    args = parser.parse_args()

    config = load_week2_config(args.config)
    root = config["root"]
    sample_dir = (root / args.sample_dir).resolve() if args.sample_dir \
        else config["paths"]["sample_dir"]
    table_path, report_path = output_paths(config, sample_dir)
    record: dict[str, Any] = {"sample_dir": str(sample_dir.relative_to(root)
                                                if sample_dir.is_relative_to(root)
                                                else sample_dir)}

    try:
        features = build_feature_table(load_tables(config, sample_dir), config)
    except DataValidationError as err:
        write_report(report_path, {**record, "status": "fail", "stage": err.stage,
                                   "problems": err.problems})
        print(f"\nFAILED: {err}\n\nNo feature table written. Report: {report_path}")
        sys.exit(1)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(table_path, index=False)
    write_report(report_path, {**record, "status": "pass", "stage": None, "problems": [],
                               "rows": len(features), "columns": features.shape[1],
                               "output": str(table_path.relative_to(root))})
    print(f"PASSED: {len(features):,} rows x {features.shape[1]} columns")
    print(f"Wrote {table_path}\nWrote {report_path}")


if __name__ == "__main__":
    main()
