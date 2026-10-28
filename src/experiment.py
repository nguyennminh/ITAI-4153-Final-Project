"""Week 3 settings and the training data every experiment uses.

# PROVIDED: this whole module is finished. Read it and reuse it.

Every run in Week 3 trains on the same rows: the train split of your Week 2 feature table
(data/features/features.parquet), cleaned with YOUR Week 1 ``clean_application``. The
validation split is only ever scored, never fitted on.

Wednesday adds config/final.yaml (as ``config["week3"]["final"]``) and the report files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.baseline import clean_application, get_feature_columns
from src.compare import split_frame
from src.features.common import load_week2_config
from src.utils import DEFAULT_CONFIG

MODELS_CONFIG = "models.yaml"
FINAL_CONFIG = "final.yaml"
# Files the Wednesday commands write (relative to the repository root).
OUTPUTS = {
    "optuna_best": "reports/optuna_best.json",
    "threshold": "reports/threshold.json",
    "final_model": "reports/final_model.json",
    "cost_plot": "reports/figures/cost_curve.png",
    "reliability_plot": "reports/figures/reliability.png",
}


@dataclass
class TrainingData:
    """The fixed train and validation rows, and the feature columns, for one experiment."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    numeric: list[str]
    categorical: list[str]
    data_version: str

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


# PROVIDED
def load_week3_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load config.yaml and features.yaml (as in Week 2), models.yaml as ``config["week3"]``,
    and final.yaml as ``config["week3"]["final"]``.

    The MLflow and Optuna database paths are made absolute, so runs land in the same database
    whichever folder you start from. ``config["week3"]["outputs"]`` maps each OUTPUTS key to
    an absolute Path.
    """
    config = load_week2_config(path)
    models_path = Path(path).resolve().parent / MODELS_CONFIG
    if not models_path.exists():
        raise FileNotFoundError(f"{models_path} not found; copy it from the Week 3 overlay")
    with models_path.open() as f:
        week3 = yaml.safe_load(f)
    root = config["root"]
    week3["mlflow"]["tracking_uri"] = _absolute_sqlite(week3["mlflow"]["tracking_uri"], root)
    week3["runs_file"] = root / week3["runs_file"]
    week3["runs"] = week3.get("runs") or {}
    final_path = models_path.parent / FINAL_CONFIG
    if not final_path.exists():
        raise FileNotFoundError(f"{final_path} not found; copy it from the Wednesday overlay")
    with final_path.open() as f:
        week3["final"] = yaml.safe_load(f)
    optuna = week3["final"]["optuna"]
    optuna["storage"] = _absolute_sqlite(optuna["storage"], root)
    week3["outputs"] = {key: root / value for key, value in OUTPUTS.items()}
    config["week3"] = week3
    return config


# PROVIDED
def _absolute_sqlite(uri: str, root: Path) -> str:
    """sqlite:///name.db -> sqlite:////full/path/to/repo/name.db (other URIs unchanged)."""
    path = uri.removeprefix("sqlite:///")
    if uri.startswith("sqlite:///") and not Path(path).is_absolute():
        return f"sqlite:///{root / path}"
    return uri


# PROVIDED
def feature_table_path(config: dict[str, Any]) -> Path:
    """Where python run.py features (Week 2) writes the feature table."""
    return config["week2"]["paths"]["features_dir"] / "features.parquet"


# PROVIDED
def data_version(table: pd.DataFrame, split: pd.DataFrame) -> str:
    """A short fingerprint of the exact rows and split a model was trained on.

    It changes when any value in the feature table or the split file changes, and only then.
    Every run logs it, so two runs with the same data_version used the same data.
    """
    digest = hashlib.sha256()
    for frame in (table, split):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        digest.update("|".join(map(str, frame.columns)).encode())
    return digest.hexdigest()[:12]


# PROVIDED
def load_training_data(
    config: dict[str, Any], features: list[str] | None = None
) -> TrainingData:
    """Read the feature table, clean it, and split it with the fixed split file.

    Args:
        config: The loaded Week 3 config.
        features: Keep only these feature columns. None (the default) keeps all.

    Returns:
        A TrainingData with the train and validation rows and the feature columns, split
        into numeric and categorical by YOUR Week 1 ``get_feature_columns``.
    """
    path = feature_table_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run.py features` first.")
    raw = pd.read_parquet(path)
    split = pd.read_csv(config["paths"]["split_file"])
    version = data_version(raw, split)

    table = clean_application(raw)
    train, valid = split_frame(table, config)
    numeric, categorical = get_feature_columns(train, config)
    if features is not None:
        unknown = sorted(set(features) - set(numeric + categorical))
        if unknown:
            raise KeyError(f"features not in the feature table: {unknown[:5]}")
        numeric = [c for c in numeric if c in features]
        categorical = [c for c in categorical if c in features]
    columns = numeric + categorical
    target = config["target_col"]
    return TrainingData(
        X_train=train[columns], y_train=train[target],
        X_valid=valid[columns], y_valid=valid[target],
        numeric=numeric, categorical=categorical, data_version=version,
    )
