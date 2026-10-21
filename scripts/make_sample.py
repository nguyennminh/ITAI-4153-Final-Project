"""Build the fixed 5,000-applicant working sample from the full Kaggle download.

# PROVIDED: this whole script is finished. Do not change it: every team must build the
# identical sample, and `python run.py check` compares a hash of it.

Run it with ``python run.py sample`` or
``python -m scripts.make_sample [--force] [--raw-dir PATH]``.

How it works
1. Read only the id and target columns of application_train.csv and pick applicants:
   within each TARGET class, shuffle the sorted ids with a seeded NumPy RandomState and
   take the first n. The sample keeps the full data's default rate.
2. Copy every row of the 7 tables that belongs to a chosen applicant into data/sample/.
   Large files are read in chunks, as plain text, so memory stays low and every value
   is copied exactly as it appears in the original file.
   bureau_balance has no applicant id; it links through SK_ID_BUREAU to bureau rows.
3. Write reports/sample_row_counts.json and reports/sample_hash.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import DEFAULT_CONFIG, load_config

BUREAU_ID = "SK_ID_BUREAU"
# Child tables keyed by the applicant id (bureau is handled first, bureau_balance after it).
APPLICANT_CHILD_TABLES = (
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
)
ROW_COUNTS_FILENAME = "sample_row_counts.json"
HASH_FILENAME = "sample_hash.txt"


# PROVIDED
def sample_hash(ids: list[int] | np.ndarray) -> str:
    """sha256 of the sorted applicant ids joined by newlines."""
    text = "\n".join(str(int(i)) for i in sorted(ids))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# PROVIDED
def check_raw_files(raw_dir: Path, tables: dict[str, str]) -> None:
    """Stop with a clear message if any raw CSV is missing."""
    missing = [raw_dir / f for f in tables.values() if not (raw_dir / f).exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing raw data file(s):\n  {names}\nDownload and unzip the Kaggle data into "
            "data/raw first (see 'Getting the data' in the README)."
        )


# PROVIDED
def read_header(path: Path) -> list[str]:
    """Return a CSV file's column names without reading its rows."""
    return list(pd.read_csv(path, nrows=0).columns)


# PROVIDED
def require_columns(path: Path, columns: list[str]) -> None:
    """Stop with a clear message if a CSV lacks a column the sampler needs."""
    missing = [c for c in columns if c not in read_header(path)]
    if missing:
        raise ValueError(f"{path.name} is missing expected column(s) {missing}.")


# PROVIDED
def choose_applicants(app_path: Path, id_col: str, target_col: str,
                      sample_size: int, seed: int) -> tuple[np.ndarray, float]:
    """Pick ``sample_size`` applicant ids, stratified on the target.

    Returns the sorted chosen ids and the default rate of the full table.
    """
    labels = pd.read_csv(app_path, usecols=[id_col, target_col],
                         dtype={id_col: "int64", target_col: "int64"})
    labels = labels.sort_values(id_col, kind="mergesort").reset_index(drop=True)
    if labels[id_col].duplicated().any():
        raise ValueError(f"{app_path.name} has duplicate {id_col} values.")
    if sample_size > len(labels):
        raise ValueError(f"sample_size {sample_size} exceeds the {len(labels)} applicants.")

    rate = float(labels[target_col].mean())
    n_pos = int(round(sample_size * rate))
    wanted = {1: n_pos, 0: sample_size - n_pos}

    chosen = []
    for cls, n in wanted.items():
        class_ids = labels.loc[labels[target_col] == cls, id_col].to_numpy()
        if n > len(class_ids):
            raise ValueError(f"Need {n} applicants with {target_col}={cls}, have {len(class_ids)}.")
        order = np.random.RandomState(seed).permutation(len(class_ids))
        chosen.append(class_ids[order[:n]])
    return np.sort(np.concatenate(chosen)), rate


# PROVIDED
def filter_csv(src: Path, dst: Path, key_col: str, keep: set[str], chunksize: int,
               collect_col: str | None = None) -> tuple[int, set[str]]:
    """Copy the rows of ``src`` whose ``key_col`` is in ``keep`` into ``dst``, chunk by chunk.

    Every value is read as text (no type guessing, no NA conversion), so it is written
    back exactly as it was. The header is always written, so a table with no matching
    rows still produces a valid, header-only CSV.

    Returns the number of rows written and, if ``collect_col`` is given, the set of that
    column's values among the kept rows.
    """
    pd.DataFrame(columns=read_header(src)).to_csv(dst, index=False)
    n_rows = 0
    collected: set[str] = set()
    reader = pd.read_csv(src, dtype=str, keep_default_na=False, chunksize=chunksize)
    for chunk in reader:
        kept = chunk[chunk[key_col].isin(keep)]
        if kept.empty:
            continue
        kept.to_csv(dst, mode="a", header=False, index=False)
        n_rows += len(kept)
        if collect_col is not None:
            collected.update(kept[collect_col])
    return n_rows, collected


# PROVIDED
def build_sample(config: dict[str, Any], force: bool = False,
                 raw_dir: Path | None = None) -> dict[str, Any]:
    """Build data/sample/ and the sample reports. Returns counts, rate, and hash."""
    raw_dir = Path(raw_dir) if raw_dir else config["paths"]["raw_dir"]
    sample_dir = config["paths"]["sample_dir"]
    reports_dir = config["paths"]["reports_dir"]
    tables = config["tables"]
    id_col, target_col = config["id_col"], config["target_col"]

    if sample_dir.exists() and any(sample_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{sample_dir} already exists. Rebuild it with: python run.py sample --force"
        )
    check_raw_files(raw_dir, tables)
    require_columns(raw_dir / tables["application_train"], [id_col, target_col])
    require_columns(raw_dir / tables["bureau"], [id_col, BUREAU_ID])
    require_columns(raw_dir / tables["bureau_balance"], [BUREAU_ID])
    for name in APPLICANT_CHILD_TABLES:
        require_columns(raw_dir / tables[name], [id_col])

    ids, full_rate = choose_applicants(raw_dir / tables["application_train"], id_col,
                                       target_col, config["sample_size"], config["seed"])
    keep_ids = {str(int(i)) for i in ids}

    # Build into a temporary folder and swap it in at the end, so an interrupted run
    # never leaves a half-written sample behind.
    tmp_dir = sample_dir.with_name(sample_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    chunk = config["chunksize"]
    counts: dict[str, int] = {}

    def run(name: str, key: str, keep: set[str], collect: str | None = None) -> set[str]:
        print(f"  filtering {tables[name]} ...", flush=True)
        counts[name], collected = filter_csv(raw_dir / tables[name], tmp_dir / tables[name],
                                             key, keep, chunk, collect)
        return collected

    run("application_train", id_col, keep_ids)
    if counts["application_train"] != len(ids):
        raise RuntimeError("Sampled application rows do not match the chosen ids.")
    bureau_ids = run("bureau", id_col, keep_ids, collect=BUREAU_ID)
    run("bureau_balance", BUREAU_ID, bureau_ids)
    for name in APPLICANT_CHILD_TABLES:
        run(name, id_col, keep_ids)

    sample_default_rate = float(
        pd.read_csv(tmp_dir / tables["application_train"], usecols=[target_col])[target_col].mean()
    )
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    tmp_dir.rename(sample_dir)

    digest = sample_hash(ids)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ordered = {name: counts[name] for name in tables}
    (reports_dir / ROW_COUNTS_FILENAME).write_text(json.dumps(ordered, indent=2) + "\n")
    (reports_dir / HASH_FILENAME).write_text(digest + "\n")
    return {"row_counts": ordered, "full_default_rate": full_rate,
            "sample_default_rate": sample_default_rate, "hash": digest}


# PROVIDED
def print_report(result: dict[str, Any]) -> None:
    """Print the row counts table, default rates, and hash."""
    print(f"\n{'table':<24}{'rows in sample':>16}")
    print("-" * 40)
    for name, n in result["row_counts"].items():
        print(f"{name:<24}{n:>16,}")
    print(f"\nDefault rate: full data {result['full_default_rate']:.4f}, "
          f"sample {result['sample_default_rate']:.4f}")
    print(f"Sample hash: {result['hash']}")


# PROVIDED
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the fixed working sample.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild an existing sample")
    parser.add_argument("--raw-dir", default=None,
                        help="use a full download in another folder instead of data/raw")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Building a {config['sample_size']:,}-applicant sample (seed {config['seed']}). "
          "This reads the large tables once and can take several minutes.")
    try:
        result = build_sample(config, force=args.force, raw_dir=args.raw_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    print_report(result)
    print(f"\nWrote {config['paths']['sample_dir']} and reports/{ROW_COUNTS_FILENAME}, "
          f"reports/{HASH_FILENAME}")


if __name__ == "__main__":
    main()
