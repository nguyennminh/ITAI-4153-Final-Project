"""Make a corrupted copy of your sample, the way real data goes wrong.

Hand this out in class on Wednesday (the surprise). Students save it as
scripts/corrupt_sample.py and run, from the repository root:

    python -m scripts.corrupt_sample

It reads data/sample/ (their own sample, so nothing from the dataset is shared) and writes
data/corrupt/ with the same seven files, some of them damaged. data/ is already in
.gitignore, so the corrupted copy is never committed. Then:

    python run.py features --sample-dir data/corrupt

must fail, list the problems in plain language, and write no feature table.

The damage is fixed (seeded), so every team gets the same corrupted copy. What it changes is
listed only in the instructor guide; do not tell students before they run it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sample"
TARGET = ROOT / "data" / "corrupt"
SEED = 2026


def corrupt_application(app: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A renamed column, missing credit amounts, impossible values, and duplicated rows."""
    app = app.copy()
    n = len(app)
    app.loc[rng.choice(n, size=250, replace=False), "AMT_CREDIT"] = np.nan
    born = rng.choice(n, size=40, replace=False)
    app.loc[born, "DAYS_BIRTH"] = -app.loc[born, "DAYS_BIRTH"]          # born in the future
    app.loc[rng.choice(n, size=15, replace=False), "AMT_ANNUITY"] = -1500.0
    app = pd.concat([app, app.iloc[rng.choice(n, size=3, replace=False)]], ignore_index=True)
    return app.rename(columns={"AMT_INCOME_TOTAL": "AMT_INCOME"})       # a "schema change"


def corrupt_previous(prev: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A new upstream system writes statuses in capitals: nothing crashes, rates go wrong."""
    prev = prev.copy()
    rows = rng.random(len(prev)) < 0.2
    prev.loc[rows, "NAME_CONTRACT_STATUS"] = prev.loc[rows, "NAME_CONTRACT_STATUS"].str.upper()
    return prev


def corrupt_installments(inst: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Refunds booked as negative payments."""
    inst = inst.copy()
    rows = rng.choice(len(inst), size=30, replace=False)
    inst.loc[rows, "AMT_PAYMENT"] = -inst.loc[rows, "AMT_PAYMENT"].abs() - 1
    return inst


def main() -> None:
    if not (SOURCE / "application_train.csv").exists():
        raise SystemExit("data/sample not found: run python run.py sample first")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    damage = {
        "application_train.csv": corrupt_application,
        "previous_application.csv": corrupt_previous,
        "installments_payments.csv": corrupt_installments,
    }
    for csv in sorted(SOURCE.glob("*.csv")):
        if csv.name in damage:
            damage[csv.name](pd.read_csv(csv), rng).to_csv(TARGET / csv.name, index=False)
        else:
            shutil.copy(csv, TARGET / csv.name)
        print(f"  {csv.name}")
    print(f"Wrote {TARGET}. Now run: python run.py features --sample-dir data/corrupt")


if __name__ == "__main__":
    main()
