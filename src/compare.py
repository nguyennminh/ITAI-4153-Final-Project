"""Does the credit history help? Re-run the Week 1 baseline on growing feature sets.

# PROVIDED: this whole module is finished. Read it and reuse it.

Run it with ``python run.py compare`` or ``python -m src.compare``. For each feature set in
config/features.yaml it:

1. aggregates the listed child tables (your functions in src/features/),
2. joins them onto the application table (your ``join_features``), adding the ratio
   features when the set lists "ratios",
3. trains YOUR Week 1 pipeline from src/baseline.py on the train split only, and
4. scores the validation split with YOUR ``evaluate``.

Every set uses the same model, the same split, and the same seed, so the only thing that
changes is the features. It writes reports/auc_comparison.json. Your Week 1
reports/metrics.json is not touched.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from src.baseline import build_model, build_preprocessor, clean_application, get_feature_columns
from src.data import load_application, load_table
from src.features.bureau import aggregate_bureau, aggregate_bureau_balance
from src.features.common import load_week2_config
from src.features.join import join_features
from src.features.payments import aggregate_credit_card, aggregate_installments
from src.features.previous import aggregate_pos_cash, aggregate_previous
from src.features.ratios import add_ratio_features
from src.metrics import evaluate
from src.uncertainty import bootstrap_auc_ci, bootstrap_auc_diff_ci
from src.utils import DEFAULT_CONFIG, set_seed

BASELINE_SET = "application"
RATIOS = "ratios"  # not a table: adds the ratio features after the join


# PROVIDED
def build_aggregate(table: str, config: dict[str, Any]) -> pd.DataFrame:
    """Load one child table from data/sample and aggregate it to one row per applicant."""
    id_col = config["id_col"]
    if table == "bureau":
        return aggregate_bureau(load_table("bureau", config), id_col)
    if table == "bureau_balance":
        bureau = load_table("bureau", config)
        return aggregate_bureau_balance(load_table("bureau_balance", config), bureau, id_col)
    if table == "previous_application":
        return aggregate_previous(load_table("previous_application", config), id_col)
    if table == "POS_CASH_balance":
        return aggregate_pos_cash(load_table("POS_CASH_balance", config), id_col)
    if table == "installments_payments":
        return aggregate_installments(load_table("installments_payments", config), id_col)
    if table == "credit_card_balance":
        return aggregate_credit_card(load_table("credit_card_balance", config), id_col)
    raise KeyError(f"No aggregation for table '{table}' yet")


# PROVIDED
def split_frame(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split any one-row-per-applicant table with the fixed split file."""
    id_col = config["id_col"]
    split = pd.read_csv(config["paths"]["split_file"], dtype={id_col: df[id_col].dtype})
    side = df[id_col].map(split.set_index(id_col)["split"])
    if side.isna().any():
        raise ValueError(f"{int(side.isna().sum())} applicants are missing from the split file")
    train = df[side == "train"].reset_index(drop=True)
    valid = df[side == "valid"].reset_index(drop=True)
    return train, valid


# PROVIDED
def score_feature_set(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the Week 1 pipeline on train rows of ``df``; return valid labels, scores, #features."""
    set_seed(config["seed"])
    target = config["target_col"]
    train, valid = split_frame(clean_application(df), config)
    numeric, categorical = get_feature_columns(train, config)
    features = numeric + categorical
    model = build_model(config, build_preprocessor(numeric, categorical))
    model.fit(train[features], train[target])  # fit on train only
    scores = model.predict_proba(valid[features])[:, 1]
    return valid[target].to_numpy(), scores, len(features)


# PROVIDED
def run_comparison(config: dict[str, Any]) -> dict[str, Any]:
    """Score every feature set in features.yaml and write reports/auc_comparison.json."""
    week2 = config["week2"]
    feature_sets: dict[str, list[str]] = week2["feature_sets"]
    if next(iter(feature_sets)) != BASELINE_SET or feature_sets[BASELINE_SET]:
        raise ValueError(f"the first feature set must be '{BASELINE_SET}: []'")
    n_boot, seed = week2["n_bootstrap"], config["seed"]

    app = load_application(config)
    aggregates: dict[str, pd.DataFrame] = {}
    results, baseline_scores = [], None
    for name, tables in feature_sets.items():
        child_tables = [t for t in tables if t != RATIOS]
        for table in child_tables:
            if table not in aggregates:
                aggregates[table] = build_aggregate(table, config)
        joined = join_features(app, {t: aggregates[t] for t in child_tables}, config["id_col"])
        if RATIOS in tables:
            joined = add_ratio_features(joined)
        y_valid, scores, n_features = score_feature_set(joined, config)
        if baseline_scores is None:
            baseline_scores = scores

        row = {"feature_set": name, "tables": tables, "n_features": n_features,
               **evaluate(y_valid, scores)}
        row["roc_auc_ci"] = list(bootstrap_auc_ci(y_valid, scores, n_boot, seed))
        row["delta_roc_auc"] = row["roc_auc"] - results[0]["roc_auc"] if results else 0.0
        row["delta_ci"] = (list(bootstrap_auc_diff_ci(y_valid, scores, baseline_scores,
                                                      n_boot, seed)) if results else [0.0, 0.0])
        results.append(row)
        print(f"  {name:<44} roc_auc {row['roc_auc']:.3f}", flush=True)

    report = {
        "seed": seed,
        "n_bootstrap": n_boot,
        "n_valid": int(len(y_valid)),
        "valid_defaults": int(y_valid.sum()),
        "results": results,
    }
    path = week2["paths"]["comparison_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


# PROVIDED
def print_table(report: dict[str, Any]) -> None:
    """Print the comparison as a table."""
    print(f"\nValidation: {report['n_valid']} applicants, {report['valid_defaults']} defaults. "
          f"Intervals are 95% bootstrap ({report['n_bootstrap']} resamples).\n")
    print(f"{'feature set':<44}{'#feat':>6}{'ROC-AUC':>9}{'95% CI':>16}"
          f"{'change':>9}{'95% CI of change':>20}")
    for r in report["results"]:
        ci = f"[{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}]"
        dci = f"[{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}]"
        print(f"{r['feature_set']:<44}{r['n_features']:>6}{r['roc_auc']:>9.3f}{ci:>16}"
              f"{r['delta_roc_auc']:>+9.3f}{dci:>20}")


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.compare [--config PATH]``."""
    parser = argparse.ArgumentParser(description="Compare the baseline on growing feature sets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    config = load_week2_config(args.config)
    report = run_comparison(config)
    print_table(report)
    print(f"\nWrote {config['week2']['paths']['comparison_file']}")


if __name__ == "__main__":
    main()
