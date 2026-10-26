"""The models Week 3 compares, built from a short spec in config/models.yaml.

A spec is a small dict, for example::

    {"model": "hist_gradient_boosting", "params": {"learning_rate": 0.05}, "undersample": 3}

``make_model`` turns it into an unfitted scikit-learn Pipeline, and ``fit_model`` fits it,
undersampling the training rows first when the spec asks for that.
"""

from __future__ import annotations

from typing import Any

import numpy as np  # noqa: F401  (for you to use)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401  (for you to use)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier  # noqa: F401  (for you to use)

from src.baseline import build_model, build_preprocessor

# PROVIDED: the model names a spec may use. The check script treats the BOOSTING ones as
# boosting runs, whichever library you pick.
BOOSTING_MODELS = {"hist_gradient_boosting", "lightgbm", "xgboost", "catboost"}
MODEL_NAMES = {"logistic_regression", "decision_tree"} | BOOSTING_MODELS


# PROVIDED
def tree_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Prepare columns for a tree model: numbers as they are, categories as integer codes.

    Trees do not need scaling, and scikit-learn's trees handle NaN themselves, so numbers
    pass through untouched. Each text category becomes an integer code; a missing or unseen
    category becomes -1.
    """
    codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-1)
    return ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", codes, categorical_cols),
    ])


def make_model(
    spec: dict[str, Any], numeric_cols: list[str], categorical_cols: list[str],
    config: dict[str, Any],
) -> Pipeline:
    """Build the unfitted Pipeline described by ``spec``.

    - ``logistic_regression``: YOUR Week 1 pipeline (already written below; read it).
    - ``decision_tree``: ``tree_preprocessor`` followed by a ``DecisionTreeClassifier``.
    - ``hist_gradient_boosting``: ``tree_preprocessor`` followed by a
      ``HistGradientBoostingClassifier``.

    The tree and boosting models get ``random_state=config["seed"]`` plus every entry of
    ``spec.get("params", {})``. Name the two Pipeline steps ``"preprocess"`` and ``"model"``,
    as the Week 1 pipeline does. Any other model name raises a ValueError that lists
    MODEL_NAMES.

    Args:
        spec: One run from config/models.yaml, for example ``{"model": "decision_tree"}``.
        numeric_cols: The numeric feature columns.
        categorical_cols: The categorical feature columns.
        config: The loaded config (for the seed, and Week 1's baseline settings).

    Returns:
        An unfitted Pipeline. Nothing is trained here.

    Hints:
        - ``params = spec.get("params") or {}``, then ``DecisionTreeClassifier(**params,
          random_state=...)``.
        - Optional: to use LightGBM or XGBoost, add the package to requirements.txt, then add
          a branch that imports it inside this function (``lightgbm.LGBMClassifier`` with
          ``verbose=-1``, or ``xgboost.XGBClassifier``).
    """
    name = spec["model"]
    if name == "logistic_regression":
        return build_model(config, build_preprocessor(numeric_cols, categorical_cols))
    params = {"random_state": config["seed"], **(spec.get("params") or {})}
    if name == "decision_tree":
        model = DecisionTreeClassifier(**params)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model {name!r}. Expected one of {sorted(MODEL_NAMES)}")
    return Pipeline([
        ("preprocess", tree_preprocessor(numeric_cols, categorical_cols)),
        ("model", model),
    ])


def undersample(
    X: pd.DataFrame, y: pd.Series, ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every defaulter and a random ``ratio`` non-defaulters per defaulter.

    With 320 defaulters and ratio 3, the result has 320 defaulters and 960 non-defaulters.
    If there are fewer non-defaulters than that, keep them all. The kept rows stay in their
    original order, and both outputs get a fresh 0..n-1 index. The inputs are not changed.

    Args:
        X: Feature rows.
        y: Labels (1 = default), aligned with ``X``.
        ratio: Non-defaulters to keep per defaulter, for example 3.
        seed: Seed for ``np.random.default_rng``, so the same rows are kept every time.

    Returns:
        ``(X_kept, y_kept)``.

    Hints:
        - Work with positions: ``pos = np.flatnonzero(y.to_numpy() == 1)``, the same for 0.
        - ``rng.choice(neg, size=k, replace=False)``, then ``np.sort`` the kept positions.
        - ``X.iloc[keep].reset_index(drop=True)``.
        - Only ever call this on TRAINING rows. ``fit_model`` and your
          ``cross_validate_auc`` make sure of that: never undersample a validation fold.
    """
    labels = y.to_numpy()
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    k = min(len(neg), int(ratio * len(pos)))
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate((pos, rng.choice(neg, size=k, replace=False))))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


# PROVIDED
def fit_model(
    model: Pipeline, X: pd.DataFrame, y: pd.Series,
    undersample_ratio: float | None, seed: int,
) -> Pipeline:
    """Fit ``model`` on ``(X, y)``, undersampling these rows first if a ratio is given."""
    if undersample_ratio:
        X, y = undersample(X, y, undersample_ratio, seed)
    return model.fit(X, y)


# PROVIDED
def describe_spec(spec: dict[str, Any]) -> str:
    """What a spec changes from the model's defaults, e.g. 'learning_rate=0.05, undersample=3'.

    An empty string means "every setting at its default". The check script uses this to tell
    a default boosting run from its variations.
    """
    changes = {**(spec.get("params") or {})}
    if spec.get("undersample"):
        changes["undersample"] = spec["undersample"]
    return ", ".join(f"{key}={changes[key]}" for key in sorted(changes))

