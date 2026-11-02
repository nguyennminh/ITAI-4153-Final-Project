"""The model under test, and helpers that change an applicant the way real life would.

# PROVIDED: this whole module is finished. Read it and reuse it.

The model tests in tests/model/ test the model you registered in Week 3
(``models:/credit-risk@staging``), through ``predict_proba`` only. They never look inside
the model, so they keep working whatever model you register later (boosting, a tree,
LightGBM, ...). That is what makes them useful in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.experiment import TrainingData, feature_table_path, load_training_data
from src.features.ratios import RATIO_FEATURES, add_ratio_features
from src.models import fit_model, make_model
from src.tracking import setup_mlflow
from src.tune import model_spec

INCOME = "AMT_INCOME_TOTAL"


@dataclass
class ProductionModel:
    """The model under test, the rows to test it on, and where it came from."""

    model: Any
    data: TrainingData
    threshold: float
    source: str


# PROVIDED
def score(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Probability of default for every row of ``X``. The ONLY way the tests call a model."""
    return model.predict_proba(X)[:, 1]


# PROVIDED
def load_production_model(config: dict[str, Any]) -> ProductionModel:
    """Load the model registered in Week 3, with the validation rows it is tested on.

    Uses ``models:/credit-risk@staging`` and the features and threshold recorded in
    reports/final_model.json. If your team has not run ``python run.py final`` yet, it
    trains the model from config/final.yaml instead (uncalibrated, threshold 0.09), so the
    tests can still run; ``source`` says which one you got.

    Raises:
        FileNotFoundError: if the Week 2 feature table is missing (no data on this machine,
            for example in CI). The model tests turn this into a skip.
    """
    if not feature_table_path(config).exists():
        raise FileNotFoundError("data/features/features.parquet not found")
    final_path = config["week3"]["outputs"]["final_model"]
    if final_path.exists():
        record = json.loads(final_path.read_text())
        setup_mlflow(config)
        try:
            model = mlflow.sklearn.load_model(record["registered"]["uri"])
        except Exception:  # noqa: BLE001  (this laptop has no MLflow database: train instead)
            model = None
        if model is not None:
            data = load_training_data(config, record["features"])
            return ProductionModel(model, data, float(record["threshold"]),
                                   f"registered model {record['registered']['uri']}")
    data = load_training_data(config)
    spec = model_spec(config)
    model = fit_model(make_model(spec, data.numeric, data.categorical, config),
                      data.X_train, data.y_train, spec.get("undersample"), config["seed"])
    return ProductionModel(model, data, 0.09, "final.yaml model, trained now (not registered)")


# PROVIDED
def change_income(X: pd.DataFrame, factor: float) -> pd.DataFrame:
    """A copy of ``X`` where every applicant earns ``factor`` times as much, everything else equal.

    "Everything else equal" means what would really be true: the loan amounts stay the same,
    so the ratio features that divide by income (CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO)
    change with it. They are recomputed with YOUR Week 2 ``add_ratio_features``. Changing
    AMT_INCOME_TOTAL alone would describe an impossible applicant.
    """
    out = X.copy()
    out[INCOME] = out[INCOME] * factor
    ratios = [c for c in RATIO_FEATURES if c in out.columns]
    recomputed = add_ratio_features(out.drop(columns=ratios))
    out[ratios] = recomputed[ratios].to_numpy()
    return out
