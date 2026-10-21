"""Hand-made tables for the Week 2 tests. No Kaggle data is needed to run the tests.

# DO NOT EDIT

The small tables are tiny on purpose: you can work out every expected number by hand.
Applicant 1 has history everywhere, 2 has a little, 3 has no bureau records, 4 has only
"C"/"X" monthly statuses, and 5 has no history at all.

The larger generated tables (500 applicants, all seven tables) feed the end-to-end tests of
compare and make_features.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.common import load_week2_config
from tests.conftest import REPO_ROOT, SENTINEL, make_application, make_split, write_config

NAN = np.nan


# DO NOT EDIT
def small_application() -> pd.DataFrame:
    """Five applicants, deliberately not sorted by id."""
    return pd.DataFrame({
        "SK_ID_CURR": [3, 1, 2, 5, 4],
        "TARGET": [0, 1, 0, 0, 1],
        "AMT_INCOME_TOTAL": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


# DO NOT EDIT
def small_bureau() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 2, 4],
        "SK_ID_BUREAU": [11, 12, 13, 21, 41],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed", "Closed"],
        "AMT_CREDIT_SUM": [100.0, 300.0, 200.0, 1000.0, 500.0],
        "AMT_CREDIT_SUM_DEBT": [50.0, NAN, 100.0, 0.0, NAN],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 10.0, 0.0, 0.0],
        "CREDIT_DAY_OVERDUE": [0, 0, 5, 0, 0],
        "DAYS_CREDIT": [-100, -900, -50, -2000, -300],
    })


# DO NOT EDIT
def small_bureau_balance() -> pd.DataFrame:
    """Credit 13 (applicant 1) has no monthly history at all."""
    rows = [
        (11, 0, "0"), (11, -1, "1"), (11, -2, "C"),          # applicant 1
        (12, 0, "X"), (12, -1, "X"),                          # applicant 1
        (21, 0, "0"), (21, -1, "0"), (21, -2, "2"), (21, -3, "5"),  # applicant 2
        (41, 0, "C"), (41, -1, "X"),                          # applicant 4
    ]
    return pd.DataFrame(rows, columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])


# DO NOT EDIT
def small_previous() -> pd.DataFrame:
    """Applications 102 and 103 tie on DAYS_DECISION; 103 is listed first on purpose."""
    return pd.DataFrame({
        "SK_ID_PREV": [101, 103, 102, 201, 202],
        "SK_ID_CURR": [1, 1, 1, 2, 2],
        "NAME_CONTRACT_STATUS": ["Approved", "Canceled", "Refused", "Approved", "Unused offer"],
        "AMT_APPLICATION": [1000.0, 3000.0, 2000.0, 500.0, 700.0],
        "AMT_CREDIT": [900.0, 3000.0, 0.0, 500.0, 700.0],
        "DAYS_DECISION": [-500, -100, -100, -50, -800],
        "DAYS_TERMINATION": [-400, SENTINEL, SENTINEL, -10, -700],
        "DAYS_FIRST_DUE": [-480, SENTINEL, SENTINEL, -40, SENTINEL],
    })


# DO NOT EDIT
def small_pos_cash() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_PREV": [101, 101, 101, 105, 105, 201],
        "SK_ID_CURR": [1, 1, 1, 1, 1, 2],
        "MONTHS_BALANCE": [-3, -2, -1, -2, -1, -1],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed", "Active", "Active", "Active"],
        "SK_DPD": [0, 5, 0, 0, 0, 30],
    })


# DO NOT EDIT
def small_installments() -> pd.DataFrame:
    """Applicant 1: one early, one 10 days late, one underpaid, one never paid.
    Applicant 2: one payment 20 days late and overpaid. Applicant 3: one early payment."""
    return pd.DataFrame({
        "SK_ID_PREV": [101, 101, 101, 105, 201, 301],
        "SK_ID_CURR": [1, 1, 1, 1, 2, 3],
        "NUM_INSTALMENT_NUMBER": [1, 2, 3, 1, 1, 1],
        "DAYS_INSTALMENT": [-100, -70, -40, -10, -30, -50],
        "DAYS_ENTRY_PAYMENT": [-105, -60, -40, NAN, -10, -55],
        "AMT_INSTALMENT": [100.0, 100.0, 100.0, 50.0, 200.0, 10.0],
        "AMT_PAYMENT": [100.0, 100.0, 80.0, NAN, 250.0, 10.0],
    })


# DO NOT EDIT
def small_credit_card() -> pd.DataFrame:
    """Applicant 1: three months on a 1,000 limit (one overpaid). Applicant 2: limit 0."""
    return pd.DataFrame({
        "SK_ID_PREV": [901, 901, 901, 902, 902],
        "SK_ID_CURR": [1, 1, 1, 2, 2],
        "MONTHS_BALANCE": [-3, -2, -1, -2, -1],
        "AMT_BALANCE": [500.0, 900.0, -10.0, 300.0, 200.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0, 1000.0, 0.0, 0.0],
        "AMT_DRAWINGS_CURRENT": [100.0, 0.0, 0.0, 50.0, 0.0],
        "SK_DPD": [0, 3, 0, 0, 0],
    })


# DO NOT EDIT
def small_application_full() -> pd.DataFrame:
    """Five applicants with every column the ratio features and the input schema read."""
    return pd.DataFrame({
        "SK_ID_CURR": [3, 1, 2, 5, 4],
        "TARGET": [0, 1, 0, 0, 1],
        "AMT_INCOME_TOTAL": [100000.0, 200000.0, 50000.0, 80000.0, 120000.0],
        "AMT_CREDIT": [300000.0, 400000.0, 100000.0, 80000.0, 240000.0],
        "AMT_ANNUITY": [15000.0, 20000.0, NAN, 4000.0, 12000.0],
        "AMT_GOODS_PRICE": [270000.0, 400000.0, 90000.0, NAN, 200000.0],
        "DAYS_BIRTH": [-12000, -15000, -9000, -20000, -11000],
        "DAYS_EMPLOYED": [-1000, SENTINEL, -200, -3000, 0],
    })


# DO NOT EDIT
def make_full_application(seed: int = 0) -> pd.DataFrame:
    """The Week 1 synthetic applicants plus AMT_ANNUITY and AMT_GOODS_PRICE."""
    rng = np.random.default_rng(seed)
    app = make_application()
    app["AMT_INCOME_TOTAL"] = app["AMT_INCOME_TOTAL"].clip(lower=20000)
    n = len(app)
    app["AMT_ANNUITY"] = np.where(rng.random(n) < 0.01, NAN,
                                  np.round(app["AMT_CREDIT"] / rng.uniform(10, 30, n), 1))
    app["AMT_GOODS_PRICE"] = np.where(rng.random(n) < 0.01, NAN,
                                      np.round(app["AMT_CREDIT"] * 0.9, 1))
    return app


# DO NOT EDIT
def make_week2_child_tables(ids: np.ndarray, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Larger random child tables with every column the Week 2 aggregations read."""
    rng = np.random.default_rng(seed)
    n = len(ids) * 3
    bureau = pd.DataFrame({
        "SK_ID_CURR": rng.choice(ids, size=n),
        "SK_ID_BUREAU": np.arange(5000000, 5000000 + n),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold"], size=n, p=[0.4, 0.58, 0.02]),
        "AMT_CREDIT_SUM": np.round(rng.lognormal(12, 1, size=n), 2),
        "AMT_CREDIT_SUM_DEBT": np.where(rng.random(n) < 0.2, NAN, rng.lognormal(10, 2, size=n)),
        "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(n) < 0.05, rng.lognormal(8, 1, size=n), 0),
        "CREDIT_DAY_OVERDUE": np.where(rng.random(n) < 0.05, rng.integers(1, 300, size=n), 0),
        "DAYS_CREDIT": -rng.integers(2, 2900, size=n),
    })
    with_history = bureau["SK_ID_BUREAU"].to_numpy()[rng.random(n) < 0.4]
    months = rng.integers(1, 12, size=len(with_history))
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": np.repeat(with_history, months),
        "MONTHS_BALANCE": np.concatenate([-np.arange(m) for m in months]),
        "STATUS": rng.choice(["0", "1", "2", "5", "C", "X"], size=int(months.sum()),
                             p=[0.4, 0.05, 0.02, 0.01, 0.37, 0.15]),
    })
    previous = pd.DataFrame({
        "SK_ID_PREV": np.arange(2000000, 2000000 + n),
        "SK_ID_CURR": rng.choice(ids, size=n),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled", "Unused offer"],
                                           size=n, p=[0.62, 0.18, 0.18, 0.02]),
        "AMT_APPLICATION": np.round(rng.lognormal(11, 1, size=n), 1),
        "AMT_CREDIT": np.round(rng.lognormal(11, 1, size=n), 1),
        "DAYS_DECISION": -rng.integers(2, 2900, size=n),
        "DAYS_TERMINATION": np.where(rng.random(n) < 0.3, SENTINEL,
                                     -rng.integers(1, 2800, size=n)),
    })
    loans = previous[["SK_ID_PREV", "SK_ID_CURR"]].sample(frac=0.5, random_state=seed)
    pos_months = rng.integers(1, 10, size=len(loans))
    k = int(pos_months.sum())
    pos_cash = pd.DataFrame({
        "SK_ID_PREV": np.repeat(loans["SK_ID_PREV"].to_numpy(), pos_months),
        "SK_ID_CURR": np.repeat(loans["SK_ID_CURR"].to_numpy(), pos_months),
        "MONTHS_BALANCE": np.concatenate([-np.arange(1, m + 1) for m in pos_months]),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], size=k, p=[0.9, 0.1]),
        "SK_DPD": np.where(rng.random(k) < 0.08, rng.integers(1, 60, size=k), 0),
    })
    inst_loans = previous[["SK_ID_PREV", "SK_ID_CURR"]].sample(frac=0.6, random_state=seed + 1)
    inst_n = rng.integers(1, 8, size=len(inst_loans))
    k = int(inst_n.sum())
    due = -rng.integers(2, 2900, size=k)
    paid = np.where(rng.random(k) < 0.02, NAN, due + rng.integers(-20, 15, size=k))
    amount_due = np.round(rng.lognormal(9, 1, size=k), 2)
    installments = pd.DataFrame({
        "SK_ID_PREV": np.repeat(inst_loans["SK_ID_PREV"].to_numpy(), inst_n),
        "SK_ID_CURR": np.repeat(inst_loans["SK_ID_CURR"].to_numpy(), inst_n),
        "NUM_INSTALMENT_NUMBER": np.concatenate([np.arange(1, m + 1) for m in inst_n]),
        "DAYS_INSTALMENT": due,
        "DAYS_ENTRY_PAYMENT": paid,
        "AMT_INSTALMENT": amount_due,
        "AMT_PAYMENT": np.where(np.isnan(paid), NAN,
                                np.round(amount_due * rng.choice([1.0, 0.8], size=k,
                                                                 p=[0.9, 0.1]), 2)),
    })
    card_owners = rng.choice(ids, size=len(ids) // 3, replace=False)
    card_months = rng.integers(1, 12, size=len(card_owners))
    k = int(card_months.sum())
    limit = np.where(rng.random(len(card_owners)) < 0.2, 0.0,
                     rng.choice([45000.0, 90000.0, 180000.0], size=len(card_owners)))
    credit_card = pd.DataFrame({
        "SK_ID_PREV": np.repeat(np.arange(3000000, 3000000 + len(card_owners)), card_months),
        "SK_ID_CURR": np.repeat(card_owners, card_months),
        "MONTHS_BALANCE": np.concatenate([-np.arange(1, m + 1) for m in card_months]),
        "AMT_BALANCE": np.round(rng.uniform(0, 90000, size=k), 2),
        "AMT_CREDIT_LIMIT_ACTUAL": np.repeat(limit, card_months),
        "AMT_DRAWINGS_CURRENT": np.where(rng.random(k) < 0.7, 0.0,
                                         np.round(rng.uniform(0, 20000, size=k), 2)),
        "SK_DPD": np.where(rng.random(k) < 0.05, rng.integers(1, 40, size=k), 0),
    })
    return {"bureau": bureau, "bureau_balance": bureau_balance,
            "previous_application": previous, "POS_CASH_balance": pos_cash,
            "installments_payments": installments, "credit_card_balance": credit_card}


# DO NOT EDIT
def make_week2_project(root: Path, n_bootstrap: int = 50) -> dict[str, Any]:
    """A temporary project: 500 synthetic applicants in all seven tables, and a split."""
    config_path = write_config(root, sample_size=500)
    shutil.copy(REPO_ROOT / "config" / "features.yaml", root / "config" / "features.yaml")
    config = load_week2_config(config_path)
    config["week2"]["n_bootstrap"] = n_bootstrap
    sample_dir = config["paths"]["sample_dir"]
    sample_dir.mkdir(parents=True)

    app = make_full_application()
    tables = {"application_train": app,
              **make_week2_child_tables(app["SK_ID_CURR"].to_numpy())}
    for name, df in tables.items():
        df.to_csv(sample_dir / config["tables"][name], index=False)
    make_split(app).to_csv(config["paths"]["split_file"], index=False)
    return config
