"""Synthetic test fixtures. No Kaggle data is needed to run the tests.

# DO NOT EDIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = 365243


# DO NOT EDIT
def write_config(root: Path, **overrides: Any) -> Path:
    """Copy the repo's config.yaml into ``root/config/`` with some values overridden."""
    with (REPO_ROOT / "config" / "config.yaml").open() as f:
        raw = yaml.safe_load(f)
    raw.update(overrides)
    path = root / "config" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


# DO NOT EDIT
def make_application(n: int = 500, n_pos: int = 40, seed: int = 0) -> pd.DataFrame:
    """A small application table with the columns and quirks the project relies on."""
    rng = np.random.default_rng(seed)
    target = np.zeros(n, dtype=int)
    target[rng.choice(n, size=n_pos, replace=False)] = 1
    days_employed = -rng.integers(30, 15000, size=n)
    days_employed[rng.random(n) < 0.15] = SENTINEL
    jobs = ["Laborers", "Core staff", "Managers", "Drivers"]
    occupation = rng.choice(jobs, size=n).astype(object)
    occupation[rng.random(n) < 0.3] = None
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100002, 100002 + n),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M", "XNA"], size=n, p=[0.6, 0.39, 0.01]),
        "AMT_INCOME_TOTAL": np.round(rng.lognormal(11.8, 0.5, size=n) - 15000 * target, 1),
        "AMT_CREDIT": np.round(rng.lognormal(13.0, 0.6, size=n), 1),
        "DAYS_BIRTH": -rng.integers(7500, 25000, size=n),
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": occupation,
    })


# DO NOT EDIT
def make_child_tables(ids: np.ndarray, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Small child tables linked to the applicant ids (bureau_balance via SK_ID_BUREAU)."""
    rng = np.random.default_rng(seed)
    bureau_curr = rng.choice(ids, size=len(ids) * 2)
    bureau = pd.DataFrame({
        "SK_ID_CURR": bureau_curr,
        "SK_ID_BUREAU": np.arange(5000000, 5000000 + len(bureau_curr)),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], size=len(bureau_curr)),
        "AMT_CREDIT_SUM": np.round(rng.lognormal(12, 1, size=len(bureau_curr)), 2),
    })
    bb_ids = np.repeat(bureau["SK_ID_BUREAU"].to_numpy(), 3)
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": bb_ids,
        "MONTHS_BALANCE": np.tile([0, -1, -2], len(bureau)),
        "STATUS": rng.choice(["0", "1", "C", "X"], size=len(bb_ids)),
    })

    def per_applicant(extra: dict[str, Any]) -> pd.DataFrame:
        curr = rng.choice(ids, size=len(ids))
        frame = {"SK_ID_PREV": np.arange(2000000, 2000000 + len(curr)), "SK_ID_CURR": curr}
        frame.update({k: v(len(curr)) for k, v in extra.items()})
        return pd.DataFrame(frame)

    return {
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": per_applicant({
            "AMT_APPLICATION": lambda k: np.round(rng.lognormal(11, 1, size=k), 1),
            "NAME_CONTRACT_STATUS": lambda k: rng.choice(["Approved", "Refused"], size=k),
        }),
        "POS_CASH_balance": per_applicant({
            "MONTHS_BALANCE": lambda k: -rng.integers(1, 40, size=k),
            "SK_DPD": lambda k: rng.integers(0, 3, size=k),
        }),
        "installments_payments": per_applicant({
            "AMT_INSTALMENT": lambda k: np.round(rng.lognormal(9, 1, size=k), 2),
            "AMT_PAYMENT": lambda k: np.where(rng.random(k) < 0.1, np.nan,
                                              np.round(rng.lognormal(9, 1, size=k), 2)),
        }),
        "credit_card_balance": per_applicant({
            "AMT_BALANCE": lambda k: np.round(rng.lognormal(10, 1, size=k), 2),
            "NAME_CONTRACT_STATUS": lambda k: rng.choice(["Active", "Completed"], size=k),
        }),
    }


# DO NOT EDIT
def make_split(app: pd.DataFrame, valid_frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """A stratified train/valid split of the synthetic applicants."""
    rng = np.random.RandomState(seed)
    rows = []
    for _, group in app.groupby("TARGET"):
        ids = group["SK_ID_CURR"].to_numpy()
        n_valid = int(round(len(ids) * valid_frac))
        shuffled = ids[rng.permutation(len(ids))]
        rows += [(i, "valid") for i in shuffled[:n_valid]]
        rows += [(i, "train") for i in shuffled[n_valid:]]
    split = pd.DataFrame(rows, columns=["SK_ID_CURR", "split"])
    return split.sort_values("SK_ID_CURR").reset_index(drop=True)


# DO NOT EDIT
@pytest.fixture
def app_df() -> pd.DataFrame:
    """500 synthetic applicants, 8 percent defaults."""
    return make_application()


# DO NOT EDIT
@pytest.fixture
def project(tmp_path: Path) -> dict[str, Any]:
    """A temporary project with a 500-applicant sample in all 7 tables and a split file."""
    config_path = write_config(tmp_path, sample_size=500)
    config = load_config(config_path)
    sample_dir = config["paths"]["sample_dir"]
    sample_dir.mkdir(parents=True)

    app = make_application()
    tables = {"application_train": app, **make_child_tables(app["SK_ID_CURR"].to_numpy())}
    for name, df in tables.items():
        df.to_csv(sample_dir / config["tables"][name], index=False)
    make_split(app).to_csv(config["paths"]["split_file"], index=False)
    return config


# DO NOT EDIT
def make_raw_tables() -> dict[str, pd.DataFrame]:
    """A tiny raw download: 60 applicants (10 defaults), all values stored as text.

    Includes text that type guessing would change ('NA', '007', '1.50', quoted commas)
    so the tests can prove rows are copied unchanged. credit_card_balance only has rows
    for unknown applicants, so its sample must be header-only.
    """
    n = 60
    ids = np.arange(200000, 200000 + 3 * n, 3)
    target = np.zeros(n, dtype=int)
    target[np.arange(2, n, 6)] = 1  # 10 defaults
    suites = np.array(["Unaccompanied", "Spouse, partner", "", "NA"], dtype=object)
    app = pd.DataFrame({
        "SK_ID_CURR": ids.astype(str),
        "TARGET": target.astype(str),
        "NAME_TYPE_SUITE": suites[np.arange(n) % 4],
        "AMT_INCOME_TOTAL": [f"{100000 + 1500 * i}.0" for i in range(n)],
        "REGION_CODE": ["007"] * n,
        "EXT_SOURCE_1": ["1.50" if i % 2 else "" for i in range(n)],
        "DAYS_EMPLOYED": [str(SENTINEL) if i % 7 == 0 else str(-100 * i) for i in range(n)],
    })
    children = make_child_tables(ids, seed=1)
    tables = {"application_train": app}
    for name, df in children.items():
        tables[name] = df.astype(str)
    cc = tables["credit_card_balance"]
    cc["SK_ID_CURR"] = [str(900000 + i) for i in range(len(cc))]  # nobody in the sample
    return tables


# DO NOT EDIT
@pytest.fixture
def raw_project(tmp_path: Path) -> dict[str, Any]:
    """A temporary project with a tiny raw download in data/raw (sample size 20)."""
    config_path = write_config(tmp_path, sample_size=20, chunksize=7)
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True)
    for name, df in make_raw_tables().items():
        df.to_csv(raw_dir / config["tables"][name], index=False)
    return config
