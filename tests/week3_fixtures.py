"""Small hand-made data for the Week 3 tests. No Kaggle data is needed.

# DO NOT EDIT
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin

from src.experiment import load_week3_config
from tests.conftest import REPO_ROOT, SENTINEL, make_split, write_config


# DO NOT EDIT
def make_feature_table(n: int = 400, n_pos: int = 60, seed: int = 0) -> pd.DataFrame:
    """A feature table like features.parquet: SIGNAL predicts TARGET, NOISE does not.

    Includes the quirks the real table has: NaN in a numeric column, a text column with
    missing values, an integer count, and the DAYS_EMPLOYED sentinel.
    """
    rng = np.random.default_rng(seed)
    target = np.zeros(n, dtype=int)
    target[rng.choice(n, size=n_pos, replace=False)] = 1
    noise = rng.normal(size=n)
    noise[rng.random(n) < 0.2] = np.nan
    days_employed = -rng.integers(30, 15000, size=n)
    days_employed[rng.random(n) < 0.15] = SENTINEL
    category = rng.choice(["Laborers", "Managers", "Drivers"], size=n).astype(object)
    category[rng.random(n) < 0.25] = None
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100002, 100002 + n),
        "TARGET": target,
        "SIGNAL": rng.normal(size=n) + 1.5 * target,
        "NOISE": noise,
        "BUREAU_COUNT": rng.integers(0, 6, size=n),
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": category,
    })


# DO NOT EDIT
def make_week3_project(root: Path, n: int = 400) -> dict[str, Any]:
    """A temporary project: a feature table, a split, and a small, fast Week 3 config.

    It has its own MLflow and Optuna databases, so the tests never touch your real runs.
    """
    config_path = write_config(root, sample_size=n)
    shutil.copy(REPO_ROOT / "config" / "features.yaml", root / "config" / "features.yaml")
    with (REPO_ROOT / "config" / "models.yaml").open() as f:
        models = yaml.safe_load(f)
    models.update({
        "cv_folds": 3,
        "n_bootstrap": 20,
        "tree_grid": {"max_depth": [2, 4], "min_samples_leaf": [5, 40]},
    })
    (root / "config" / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
    with (REPO_ROOT / "config" / "final.yaml").open() as f:
        final = yaml.safe_load(f)
    final["optuna"].update({"study": "test-study", "trials_per_run": 2})
    final["model"]["params"] = {"max_iter": 30}
    (root / "config" / "final.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    config = load_week3_config(config_path)

    table = make_feature_table(n)
    features_dir = config["week2"]["paths"]["features_dir"]
    features_dir.mkdir(parents=True)
    table.to_parquet(features_dir / "features.parquet", index=False)
    make_split(table).to_csv(config["paths"]["split_file"], index=False)
    return config


# DO NOT EDIT
class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Records every fit in ``SpyClassifier.fits`` and predicts the training default rate.

    ``fits`` is shared by all copies (``clone`` makes new objects), so a test can see every
    fit that happened inside your function.
    """

    fits: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SpyClassifier:
        y = np.asarray(y)
        SpyClassifier.fits.append({"n": len(y), "n_pos": int(y.sum()),
                                   "ids": set(np.asarray(X["SK_ID"]))})
        self.classes_ = np.array([0, 1])
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # A tiny amount of each row's SIGNAL, so the AUC is defined and not exactly 0.5.
        p = np.clip(self.rate_ + 0.01 * np.tanh(np.asarray(X["SIGNAL"])), 0, 1)
        return np.column_stack([1 - p, p])


# DO NOT EDIT
def spy_data(n: int = 300, n_pos: int = 45) -> tuple[pd.DataFrame, pd.Series]:
    """Rows for SpyClassifier: an id column (to track which rows were fitted) and SIGNAL."""
    table = make_feature_table(n, n_pos)
    X = pd.DataFrame({"SK_ID": table["SK_ID_CURR"], "SIGNAL": table["SIGNAL"]})
    return X, table["TARGET"]
