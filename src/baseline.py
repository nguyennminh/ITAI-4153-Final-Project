"""Logistic regression baseline on the application table.

Run it with ``python run.py baseline`` or ``python -m src.baseline``. It trains on the train
split only, scores the validation split, and writes reports/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_split
from src.metrics import evaluate
from src.utils import DEFAULT_CONFIG, load_config, set_seed

# PROVIDED
DAYS_EMPLOYED_SENTINEL = 365243
METRICS_KEYS = (
    "model", "roc_auc", "pr_auc", "ks", "n_train", "n_valid", "valid_default_rate", "seed",
)
MODEL_NAME = "logistic_regression"


# TODO(student)
def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known DAYS_EMPLOYED data problem and return a cleaned copy.

    In DAYS_EMPLOYED, the value ``DAYS_EMPLOYED_SENTINEL`` (365243, about 1,000 years) is
    a placeholder, not a real number of days.

    Args:
        df: Application rows. TARGET may or may not be present.

    Returns:
        A new DataFrame with the same rows where
        - DAYS_EMPLOYED holds NaN wherever it held the sentinel, and
        - a new integer column DAYS_EMPLOYED_ANOM is 1 where the sentinel was and 0 elsewhere.
        Every other column, including TARGET when present, is unchanged.

    Hints:
        - Do not change the DataFrame you were given. The tests check this.
        - Build the indicator before you overwrite the sentinel, or you lose the information.
        - Use the module constant, not a typed-in number.
        - It must work on frames without TARGET too.
    """
    cleaned = df.copy()
    is_sentinel = cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    cleaned["DAYS_EMPLOYED_ANOM"] = is_sentinel.astype(int)
    # mask() turns the column into floats so it can hold NaN
    cleaned["DAYS_EMPLOYED"] = cleaned["DAYS_EMPLOYED"].mask(is_sentinel)
    return cleaned


# TODO(student)
def get_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Decide which columns go into the model, split into numeric and categorical lists.

    Args:
        df: Application rows (after ``clean_application``).
        config: The loaded config. ``config["id_col"]`` and ``config["target_col"]`` name
            the id and target columns.

    Returns:
        ``(numeric_cols, categorical_cols)``: two lists of column names, in the order the
        columns appear in ``df``. Together they hold every column except the id and the target.
        Numeric means numbers or booleans; everything else (text, category) is categorical.

    Hints:
        - Which two columns must never be features, and why?
        - pandas has helpers in ``pd.api.types`` that check a column's dtype.
        - With pandas 3, text columns have dtype ``str``, not ``object``. Test for "numeric",
          and treat everything else as categorical.
    """
    excluded = {config["id_col"], config["target_col"]}
    features = [col for col in df.columns if col not in excluded]
    # Booleans count as numeric for is_numeric_dtype
    numeric_cols = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in features if col not in numeric_cols]
    return numeric_cols, categorical_cols


# TODO(student)
def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build the (unfitted) preprocessing step for the model.

    Args:
        numeric_cols: Columns to treat as numbers.
        categorical_cols: Columns to treat as categories.

    Returns:
        A ColumnTransformer that
        - fills missing numeric values with the column median and then standardizes them, and
        - fills missing categorical values with the most frequent value and then one-hot
          encodes them, ignoring categories it did not see during fitting.
        Its output must contain no missing values.

    Hints:
        - Chain the two steps for each column type with a Pipeline.
        - Validation data can contain a category that never appears in training.
        - Return the transformer unfitted; fitting happens later, on training data only.
    """
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, numeric_cols),
        ("categorical", categorical_steps, categorical_cols),
    ])


# TODO(student)
def build_model(config: dict[str, Any], preprocessor: ColumnTransformer) -> Pipeline:
    """Build the full (unfitted) model: preprocessing followed by logistic regression.

    Args:
        config: The loaded config. Read the model settings from ``config["baseline"]``
            and the random seed from ``config["seed"]``.
        preprocessor: The output of ``build_preprocessor``.

    Returns:
        A Pipeline whose last step is a LogisticRegression. It must support
        ``fit`` and ``predict_proba``.

    Hints:
        - Take every setting from the config, so a config change needs no code change.
        - Give each step a short, readable name.
    """
    settings = config["baseline"]
    classifier = LogisticRegression(
        max_iter=settings["max_iter"],
        class_weight=settings["class_weight"],
        random_state=config["seed"],
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


# TODO(student)
def write_metrics(metrics: dict[str, float], extra: dict[str, Any], path: str | Path) -> None:
    """Write the run's results to a JSON file.

    Args:
        metrics: The output of ``src.metrics.evaluate``.
        extra: The other values to record: model, n_train, n_valid, valid_default_rate, seed.
        path: Where to write the JSON file.

    The file holds one JSON object with every key in ``METRICS_KEYS``.

    Raises:
        ValueError: if any key in ``METRICS_KEYS`` is missing from metrics and extra combined.

    Hints:
        - The folder may not exist yet on a fresh clone.
        - Indent the JSON so a person can read it.
        - NumPy numbers are not JSON-serializable; make sure the values are plain Python types.
    """
    combined = {**metrics, **extra}
    missing = [key for key in METRICS_KEYS if key not in combined]
    if missing:
        raise ValueError(f"Cannot write metrics: missing keys {missing}")

    # NumPy scalars (np.float64, np.int64) are not JSON-serializable; .item() unwraps them
    record = {key: combined[key] for key in METRICS_KEYS}
    record = {key: v.item() if isinstance(v, np.generic) else v for key, v in record.items()}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


# TODO(student)
def run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the whole baseline: load, clean, train, evaluate, and save the results.

    Steps:
        1. Seed the random generators with ``config["seed"]``.
        2. Load the fixed train and validation frames (``load_split``).
        3. Clean both with ``clean_application``.
        4. Pick the feature columns from the training frame (``get_feature_columns``).
        5. Build the model (``build_preprocessor`` + ``build_model``) and fit it on the
           training rows only.
        6. Score the validation rows with ``predict_proba`` (the probability of TARGET = 1).
        7. Compute the metrics with ``src.metrics.evaluate``.
        8. Save them with ``write_metrics`` to ``config["paths"]["metrics_file"]``, adding
           model (``MODEL_NAME``), n_train, n_valid, valid_default_rate, and seed.

    Args:
        config: The loaded config.

    Returns:
        A dict with every key in ``METRICS_KEYS`` (the same values written to the file).

    Hints:
        - Nothing may be fitted on the validation rows: not the imputer, the scaler, the
          encoder, or the model.
        - ``predict_proba`` returns one column per class. Which one do you need?
        - The model must see the same columns, in the same order, at fit and at predict time.
        - Never pass TARGET in as a feature.
    """
    set_seed(config["seed"])
    target = config["target_col"]

    train, valid = load_split(config)
    train, valid = clean_application(train), clean_application(valid)

    # Columns come from the training frame and are reused, in the same order, for validation
    numeric_cols, categorical_cols = get_feature_columns(train, config)
    features = numeric_cols + categorical_cols

    model = build_model(config, build_preprocessor(numeric_cols, categorical_cols))
    model.fit(train[features], train[target])

    # predict_proba has one column per class; take the one for TARGET = 1
    positive = list(model.classes_).index(1)
    scores = model.predict_proba(valid[features])[:, positive]

    extra = {
        "model": MODEL_NAME,
        "n_train": len(train),
        "n_valid": len(valid),
        "valid_default_rate": float(valid[target].mean()),
        "seed": config["seed"],
    }
    path = config["paths"]["metrics_file"]
    write_metrics(evaluate(valid[target], scores), extra, path)
    return json.loads(Path(path).read_text())

# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.baseline [--config PATH]``."""
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_baseline(config)
    print(json.dumps(results, indent=2))
    print(f"Wrote {config['paths']['metrics_file']}")


if __name__ == "__main__":
    main()
