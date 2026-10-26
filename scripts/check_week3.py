"""Week 3 checks. Prints a PASS/FAIL table and exits non-zero if a required check fails.

# DO NOT EDIT: the instructor uses this exact script at checkpoints.

Run it with ``python run.py check --stage mon`` (what the Monday checkpoint needs) or
``python run.py check`` (Monday plus the carry-over). The checks read your MLflow runs,
so run them on the computer that trained the runs.
Passing these checks shows your runs exist and look plausible. It does not grade them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd

from scripts.check_week2 import CheckFailed, check_sample, count_commit_authors
from src.experiment import data_version, feature_table_path, load_week3_config
from src.models import BOOSTING_MODELS
from src.tracking import latest_top_level, load_runs
from src.utils import DEFAULT_CONFIG

BASELINE, TREE, TUNED, TUNING, DEFAULT = (
    "baseline_lr", "tree_unlimited", "tree_tuned", "tree_tuning", "hgb_default")
NOTES = "docs/experiment_notes.md"
MIN_VARIATIONS, TARGET_VARIATIONS = 3, 5
MIN_OVERFIT_GAP = 0.10


# DO NOT EDIT
def _runs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All finished runs, and the newest top-level run of each name (cached per config)."""
    if "_runs" not in config:
        runs = load_runs(config)
        config["_runs"] = (runs, latest_top_level(runs))
    return config["_runs"]


# DO NOT EDIT
def _run(config: dict[str, Any], name: str) -> pd.Series:
    _, top = _runs(config)
    match = top[top["run_name"] == name]
    if match.empty:
        raise CheckFailed(f"no finished run named '{name}'; python run.py train {name}")
    return match.iloc[0]


# DO NOT EDIT
def _boosting(config: dict[str, Any]) -> pd.DataFrame:
    """Boosting runs trained by python run.py train (not tuning trials or the final model)."""
    _, top = _runs(config)
    if top.empty or "params.model" not in top.columns:
        return top
    step = top.get("tags.step", pd.Series(index=top.index, dtype=str))
    boosting = top[top["params.model"].isin(BOOSTING_MODELS) & (step == "train")].copy()
    boosting["tags.settings"] = boosting.get("tags.settings", "").fillna("")
    return boosting


# DO NOT EDIT
def _cv(run: pd.Series) -> float:
    return float(run["metrics.cv_auc_mean"])


# DO NOT EDIT
def check_feature_table(config: dict[str, Any]) -> str:
    path = feature_table_path(config)
    if not path.exists():
        raise CheckFailed("data/features/features.parquet not found; python run.py features")
    table = pd.read_parquet(path)
    if len(table) != config["sample_size"]:
        raise CheckFailed(f"{len(table):,} rows, expected {config['sample_size']:,}")
    split = pd.read_csv(config["paths"]["split_file"])
    config["_data_version"] = data_version(table, split)
    return f"{len(table):,} rows x {table.shape[1]} columns, data_version {config['_data_version']}"


# DO NOT EDIT
def check_tracking(config: dict[str, Any]) -> str:
    runs, top = _runs(config)
    if top.empty:
        raise CheckFailed("no finished MLflow runs yet; python run.py train baseline_lr")
    for tag in ("tags.git_commit", "tags.data_version"):
        if tag not in top.columns or top[tag].isna().any():
            raise CheckFailed(f"some runs have no {tag.split('.')[1]} tag; train them with "
                              "python run.py train")
    current = config.get("_data_version")
    stale = sorted(top.loc[top["tags.data_version"] != current, "run_name"]) if current else []
    if stale:
        raise CheckFailed(f"{len(stale)} run(s) used different data or a different split: "
                          f"{stale[:4]}; retrain them")
    dirty = int(top["tags.git_commit"].str.endswith("-dirty").sum())
    note = f", {dirty} from uncommitted code" if dirty else ""
    return f"{len(top)} named runs ({len(runs)} with child runs), all on the same data{note}"


# DO NOT EDIT
def check_baseline(config: dict[str, Any]) -> str:
    run = _run(config, BASELINE)
    return f"CV AUC {_cv(run):.3f} ± {run['metrics.cv_auc_std']:.3f}"


# DO NOT EDIT
def check_tree_overfit(config: dict[str, Any]) -> str:
    run = _run(config, TREE)
    gap = float(run["metrics.train_auc"]) - _cv(run)
    if gap < MIN_OVERFIT_GAP:
        raise CheckFailed(f"train {run['metrics.train_auc']:.3f} vs CV {_cv(run):.3f}: does "
                          f"{TREE} really have no limits?")
    return f"train AUC {run['metrics.train_auc']:.3f} vs CV AUC {_cv(run):.3f}"


# DO NOT EDIT
def check_tree_tuned(config: dict[str, Any]) -> str:
    runs, _ = _runs(config)
    grid = config["week3"]["tree_grid"]
    expected = len(grid["max_depth"]) * len(grid["min_samples_leaf"])
    parents = runs[(runs["run_name"] == TUNING) & (runs["parent"] == "")]
    if parents.empty:
        raise CheckFailed(f"no '{TUNING}' parent run; python run.py train {TUNED}")
    children = runs[runs.get("tags.mlflow.parentRunId", pd.Series(dtype=str))
                    == parents.iloc[0]["run_id"]]
    if len(children) < expected:
        raise CheckFailed(f"{len(children)} child runs under {TUNING}, expected {expected}")
    run = _run(config, TUNED)
    best = children.loc[children["metrics.cv_auc_mean"].idxmax()]
    if abs(_cv(run) - float(best["metrics.cv_auc_mean"])) > 1e-9:
        raise CheckFailed(f"{TUNED} is not the child with the best CV AUC ({best['run_name']})")
    depth, leaf = run.get("params.max_depth"), run.get("params.min_samples_leaf")
    return f"{len(children)} trees tried; best max_depth={depth}, min_samples_leaf={leaf}: " \
           f"CV AUC {_cv(run):.3f}"


# DO NOT EDIT
def check_boosting_default(config: dict[str, Any]) -> str:
    boosting = _boosting(config)
    default = boosting[boosting["tags.settings"] == ""] if not boosting.empty else boosting
    if default.empty:
        raise CheckFailed(f"no boosting run with default settings; python run.py train {DEFAULT}")
    run = default.iloc[0]
    return f"{run['run_name']}: CV AUC {_cv(run):.3f}, train AUC {run['metrics.train_auc']:.3f}"


# DO NOT EDIT
def _variations(config: dict[str, Any]) -> pd.DataFrame:
    boosting = _boosting(config)
    if boosting.empty:
        return boosting
    return boosting[boosting["tags.settings"] != ""].drop_duplicates("tags.settings")


# DO NOT EDIT
def check_variations(config: dict[str, Any]) -> str:
    variations = _variations(config)
    if len(variations) < MIN_VARIATIONS:
        raise CheckFailed(f"{len(variations)} boosting variation(s) with different settings, "
                          f"need at least {MIN_VARIATIONS}")
    return f"{len(variations)} variations: " + ", ".join(variations["run_name"][:6])


# DO NOT EDIT
def check_best_boosting(config: dict[str, Any]) -> str:
    boosting = _boosting(config)
    if boosting.empty:
        raise CheckFailed("no boosting runs yet")
    best = boosting.loc[boosting["metrics.cv_auc_mean"].idxmax()]
    baseline, tree = _cv(_run(config, BASELINE)), _cv(_run(config, TUNED))
    if _cv(best) <= max(baseline, tree):
        raise CheckFailed(f"best boosting CV AUC {_cv(best):.3f} ({best['run_name']}) does not "
                          f"beat baseline {baseline:.3f} and tuned tree {tree:.3f} yet; look "
                          "at its train AUC")
    return f"{best['run_name']} CV AUC {_cv(best):.3f} > baseline {baseline:.3f}, " \
           f"tuned tree {tree:.3f}"


# DO NOT EDIT
def check_comparable_runs(config: dict[str, Any]) -> str:
    variations = _variations(config)
    if len(variations) < TARGET_VARIATIONS:
        raise CheckFailed(f"{len(variations)} boosting variations, need {TARGET_VARIATIONS} "
                          "(carry-over)")
    return f"{len(variations)} boosting variations on the same data"


# DO NOT EDIT
def check_notes(config: dict[str, Any]) -> str:
    path = config["root"] / NOTES
    if not path.exists():
        raise CheckFailed(f"{NOTES} not found; copy docs/experiment_notes_template.md")
    text = path.read_text()
    if "TODO" in text:
        raise CheckFailed(f"{NOTES} still contains TODO")
    missing = [name for name in _variations(config)["run_name"] if name not in text]
    if missing:
        raise CheckFailed(f"variations not discussed in {NOTES}: {missing[:4]}")
    return f"{len(text.split())} words, every variation discussed"


Check = tuple[str, Callable[[dict[str, Any]], str]]
MONDAY: list[Check] = [
    ("sample", check_sample),
    ("feature table", check_feature_table),
    ("tracking", check_tracking),
    ("baseline", check_baseline),
    ("tree overfit", check_tree_overfit),
    ("tree tuned", check_tree_tuned),
    ("boost default", check_boosting_default),
    ("variations", check_variations),
    ("best boosting", check_best_boosting),
]
CARRY_OVER: list[Check] = [
    ("5+ variations", check_comparable_runs),
    ("experiment notes", check_notes),
]


# DO NOT EDIT
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 3 checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--stage", choices=["mon", "all"], default="all",
                        help="mon: the Monday checkpoint only; all: Monday plus the carry-over")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    checks = MONDAY if args.stage == "mon" else MONDAY + CARRY_OVER

    failed = 0
    print(f"{'check':<18}{'result':<7}details")
    print("-" * 90)
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
        print(f"{name:<18}{status:<7}{detail}")
    print(f"{'git authors':<18}{'INFO':<7}{count_commit_authors(config['root'])}")
    print("-" * 90)
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
