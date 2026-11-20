"""Simulated production: new applicants for your live service, in three batches.

# PROVIDED: this whole module is finished.

Run it with ``python run.py pool`` (``--raw-dir PATH`` if your full download is not in
data/raw). The Home Credit data has no dates, so "production" is
simulated: 3,000 applicants who are NOT in your 5,000-applicant working sample are taken from
your own full download (data/raw), and their features are built by YOUR Week 2 pipeline
(``build_feature_table``, with its validation), exactly as for training. They come in three
batches of 1,000 applicants:

    batch 1   a normal week: new applicants, nothing changed
    batch 2   a week later, something changed upstream (you diagnose what, on Monday)
    batch 3   kept for Week 7

It writes into data/production/ (never committed: it is derived from the Kaggle data):

    store_extension.parquet   the 3,000 applicants' model features: python run.py export adds
                              them to the feature store, so the live service can score them
    batch1.jsonl, batch2.jsonl, batch3.jsonl   the requests to send, one per line
    labels.parquet            who really defaulted. The service never sees it; do not look at
                              it to diagnose drift: in production you would not know yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.make_sample import APPLICANT_CHILD_TABLES, BUREAU_ID, check_raw_files, filter_csv
from src.baseline import clean_application
from src.experiment import load_week3_config
from src.features.ratios import RATIO_FEATURES, add_ratio_features
from src.make_features import build_feature_table, load_tables

PRODUCTION_DIR = Path("data/production")
BATCH_SIZE = 1000
SEED = 2026
BATCH2_DEFAULT_RATE = 0.16
BATCH2_INCOME_FACTOR = 0.7
BATCH2_EXT3_MISSING = 0.4


# PROVIDED
def choose_batches(app_path: Path, exclude: set[int], id_col: str, target_col: str,
                   seed: int = SEED) -> dict[int, np.ndarray]:
    """Three disjoint batches of applicants outside ``exclude`` (the working sample)."""
    labels = pd.read_csv(app_path, usecols=[id_col, target_col])
    labels = labels[~labels[id_col].isin(exclude)].sort_values(id_col)
    rng = np.random.default_rng(seed)

    batch1 = rng.choice(labels[id_col].to_numpy(), BATCH_SIZE, replace=False)
    rest = labels[~labels[id_col].isin(batch1)]
    n_bad = int(BATCH_SIZE * BATCH2_DEFAULT_RATE)
    bad = rng.choice(rest.loc[rest[target_col] == 1, id_col].to_numpy(), n_bad, replace=False)
    good = rng.choice(rest.loc[rest[target_col] == 0, id_col].to_numpy(), BATCH_SIZE - n_bad,
                      replace=False)
    batch2 = np.concatenate([bad, good])
    rest = rest[~rest[id_col].isin(batch2)]
    batch3 = rng.choice(rest[id_col].to_numpy(), BATCH_SIZE, replace=False)
    return {1: np.sort(batch1), 2: np.sort(batch2), 3: np.sort(batch3)}


# PROVIDED
def extract_tables(config: dict[str, Any], ids: np.ndarray, target: Path) -> None:
    """Copy every row of the seven raw tables that belongs to ``ids`` into ``target``."""
    raw_dir, tables, id_col = config["paths"]["raw_dir"], config["tables"], config["id_col"]
    check_raw_files(raw_dir, tables)
    target.mkdir(parents=True, exist_ok=True)
    keep = {str(int(i)) for i in ids}
    chunk = config["chunksize"]

    def run(name: str, key: str, keep_set: set[str], collect: str | None = None) -> set[str]:
        print(f"  filtering {tables[name]} ...", flush=True)
        _, collected = filter_csv(raw_dir / tables[name], target / tables[name], key, keep_set,
                                  chunk, collect)
        return collected

    run("application_train", id_col, keep)
    bureau_ids = run("bureau", id_col, keep, collect=BUREAU_ID)
    run("bureau_balance", BUREAU_ID, bureau_ids)
    for name in APPLICANT_CHILD_TABLES:
        run(name, id_col, keep)


# PROVIDED
def apply_batch2_changes(features: pd.DataFrame, ids: np.ndarray, seed: int = SEED) -> pd.DataFrame:
    """What changed upstream for batch 2's applicants (diagnosing it is your job)."""
    out = features.copy()
    rows = out["SK_ID_CURR"].isin(ids)
    out.loc[rows, "AMT_INCOME_TOTAL"] = out.loc[rows, "AMT_INCOME_TOTAL"] * BATCH2_INCOME_FACTOR
    out = add_ratio_features(out.drop(columns=[c for c in RATIO_FEATURES if c in out.columns]))
    rng = np.random.default_rng(seed + 2)
    lost = rows & (rng.random(len(out)) < BATCH2_EXT3_MISSING)
    out.loc[lost, "EXT_SOURCE_3"] = np.nan
    return out


# PROVIDED
def build_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Build data/production/ (store extension, batch requests, labels)."""
    root, id_col, target_col = config["root"], config["id_col"], config["target_col"]
    out = root / PRODUCTION_DIR
    sample_ids = set(pd.read_csv(config["paths"]["split_file"])[id_col])
    batches = choose_batches(config["paths"]["raw_dir"] / config["tables"]["application_train"],
                             sample_ids, id_col, target_col)
    every = np.concatenate(list(batches.values()))

    raw_copy = out / "raw"
    shutil.rmtree(raw_copy, ignore_errors=True)
    print(f"Extracting {len(every):,} production applicants from {config['paths']['raw_dir']}")
    extract_tables(config, every, raw_copy)
    print("Building their features with your Week 2 pipeline (with validation) ...")
    features = build_feature_table(load_tables(config, raw_copy), config)
    features = apply_batch2_changes(features, batches[2])

    final = json.loads(config["week3"]["outputs"]["final_model"].read_text())
    model_features = final["features"]
    cleaned = clean_application(features)
    cleaned[[id_col, *model_features]].to_parquet(out / "store_extension.parquet", index=False)
    batch_of = {int(i): b for b, ids in batches.items() for i in ids}
    labels = features[[id_col, target_col]].assign(batch=features[id_col].map(batch_of))
    labels.to_parquet(out / "labels.parquet", index=False)
    for b, ids in batches.items():
        lines = [json.dumps({id_col: int(i)}) for i in ids]
        (out / f"batch{b}.jsonl").write_text("\n".join(lines) + "\n")
    shutil.rmtree(raw_copy, ignore_errors=True)
    return {"applicants": len(every), "batches": {b: len(ids) for b, ids in batches.items()},
            "features": len(model_features)}


# PROVIDED
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the simulated production batches.")
    parser.add_argument("--raw-dir", default=None,
                        help="use a full download in another folder instead of data/raw")
    args = parser.parse_args()
    config = load_week3_config()
    if args.raw_dir:
        config["paths"]["raw_dir"] = Path(args.raw_dir).expanduser().resolve()
    print("This reads the large raw tables once and takes a few minutes.")
    result = build_pool(config)
    print(f"\n{result['applicants']:,} production applicants, {result['features']} model "
          f"features each, in batches {result['batches']}")
    print(f"Wrote {PRODUCTION_DIR}/store_extension.parquet, batch1-3.jsonl, labels.parquet")
    print("Next: python run.py export (adds them to the feature store), upload, redeploy.")


if __name__ == "__main__":
    main()
