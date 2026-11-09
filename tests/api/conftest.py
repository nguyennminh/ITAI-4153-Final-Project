"""A small, made-up bundle for the API tests: no Kaggle data, no MLflow, so they run in CI.

# DO NOT EDIT
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO",
            "ANNUITY_CREDIT_RATIO", "EXT_SOURCE_2", "CODE_GENDER"]
FIRST_ID = 100001
N = 300


# DO NOT EDIT
def make_store(n: int = N, seed: int = 0) -> pd.DataFrame:
    """A feature store like the real one: SK_ID_CURR plus the model's features."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(30_000, 300_000, n).round(0)
    credit = rng.uniform(50_000, 1_000_000, n).round(0)
    annuity = (credit * rng.uniform(0.03, 0.08, n)).round(0)
    goods = (credit * rng.uniform(0.8, 1.0, n)).round(0)
    store = pd.DataFrame({
        "SK_ID_CURR": np.arange(FIRST_ID, FIRST_ID + n),
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "CREDIT_INCOME_RATIO": credit / income, "ANNUITY_INCOME_RATIO": annuity / income,
        "CREDIT_GOODS_RATIO": credit / goods, "ANNUITY_CREDIT_RATIO": annuity / credit,
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "CODE_GENDER": rng.choice(["F", "M"], n),
    })
    store.loc[3, "EXT_SOURCE_2"] = np.nan  # missing values happen in the real store
    return store


# DO NOT EDIT
def make_model(store: pd.DataFrame) -> Pipeline:
    """A small fitted pipeline with the same kind of preprocessing as the real one."""
    rng = np.random.default_rng(1)
    risk = 1 / (1 + np.exp(4 * store["EXT_SOURCE_2"].fillna(0.5) - 2))
    target = (rng.uniform(size=len(store)) < risk * 0.3).astype(int)
    numeric = [f for f in FEATURES if f != "CODE_GENDER"]
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), ["CODE_GENDER"]),
    ])
    model = Pipeline([("preprocess", prep),
                      ("model", HistGradientBoostingClassifier(max_iter=20, random_state=0))])
    return model.fit(store[FEATURES], target)


# DO NOT EDIT
@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder with the three bundle files, as python run.py export writes them."""
    folder = tmp_path_factory.mktemp("bundle")
    store = make_store()
    joblib.dump(make_model(store), folder / "model.joblib")
    store.to_parquet(folder / "feature_store.parquet", index=False)
    info = {"name": "credit-risk", "version": "7", "uri": "models:/credit-risk@staging",
            "threshold": 0.12, "features": FEATURES, "n_features": len(FEATURES),
            "valid_auc": 0.75, "data_version": "abc123", "exported": "2026-11-09T10:00:00+00:00",
            "python": "3.11", "n_applicants": N}
    (folder / "model_info.json").write_text(json.dumps(info))
    return folder


# DO NOT EDIT
@pytest.fixture(scope="session")
def client(bundle_dir: Path) -> Iterator[TestClient]:
    """The real app, started (lifespan included) on the made-up bundle."""
    from api.main import app

    old = os.environ.get("MODEL_DIR"), os.environ.pop("MODEL_BUCKET", None)
    os.environ["MODEL_DIR"] = str(bundle_dir)
    with TestClient(app) as test_client:
        yield test_client
    if old[0] is None:
        os.environ.pop("MODEL_DIR", None)
    else:
        os.environ["MODEL_DIR"] = old[0]
    if old[1] is not None:
        os.environ["MODEL_BUCKET"] = old[1]
