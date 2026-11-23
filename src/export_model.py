"""Export the registered model as a bundle the service can load anywhere.

# PROVIDED: this whole module is finished.

Run it with ``python run.py export``. Cloud Run cannot reach the MLflow database on your
laptop, so this copies everything the service needs into models/ (never committed):

    models/model.joblib            models:/credit-risk@staging: the champion (Week 3's model,
                                   or the one you promoted in Week 7)
    models/model_info.json         its version, threshold, features, and validation metrics
    models/feature_store.parquet   one row of model features per applicant: the 5,000 of
                                   the working sample, plus (Week 6) the 3,000 production
                                   applicants from data/production/store_extension.parquet,
                                   if python run.py pool has built it

and writes requirements-api.txt (committed): the service's packages, pinned to the exact
versions installed here. A model saved with one scikit-learn version must be loaded with
the same one, so the Docker image installs these versions, not "the newest".

Then you upload the three models/ files to your private Cloud Storage bucket (WEEK5.md).
The feature store holds data derived from the Kaggle data: the bucket must stay private.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn
import pandas as pd

from src.baseline import clean_application
from src.experiment import feature_table_path, load_week3_config
from src.production_pool import PRODUCTION_DIR
from src.retrain import champion_record
from src.tracking import setup_mlflow

MODELS_DIR = Path("models")
API_PACKAGES = ("fastapi", "uvicorn", "pydantic", "scikit-learn", "scipy", "numpy", "pandas",
                "pyarrow", "joblib", "google-cloud-storage")


# PROVIDED
def pinned_requirements() -> str:
    """requirements-api.txt: each package the service needs, at the version installed here."""
    lines = ["# Written by python run.py export: the exact versions the model was saved with.",
             f"# Python {platform.python_version()} (use the same minor version in the "
             "Dockerfile's FROM line)."]
    for package in API_PACKAGES:
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError as err:
            raise SystemExit(f"{package} is not installed: add it to requirements.txt and "
                             "run python run.py setup") from err
    return "\n".join(lines) + "\n"


# PROVIDED
def build_feature_store(config: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """One row per applicant: SK_ID_CURR plus the model's features, cleaned as in training.

    Week 6: the production applicants (python run.py pool) are added after the 5,000.
    """
    id_col = config["id_col"]
    table = clean_application(pd.read_parquet(feature_table_path(config)))
    missing = [f for f in features if f not in table.columns]
    if missing:
        raise SystemExit(f"the feature table lacks model features {missing[:5]}: rerun "
                         "python run.py features")
    store = table[[id_col, *features]]
    extension = config["root"] / PRODUCTION_DIR / "store_extension.parquet"
    if extension.exists():
        extra = pd.read_parquet(extension)
        absent = [f for f in features if f not in extra.columns]
        if absent:
            raise SystemExit(f"{extension} lacks model features {absent[:5]}: rerun "
                             "python run.py pool")
        overlap = set(extra[id_col]) & set(store[id_col])
        if overlap:
            raise SystemExit(f"{len(overlap)} production applicants are also in the sample: "
                             "rerun python run.py pool")
        store = pd.concat([store, extra[[id_col, *features]]], ignore_index=True)
    return store.reset_index(drop=True)


# PROVIDED
def export(config: dict[str, Any]) -> dict[str, Any]:
    """Write the bundle into models/ and requirements-api.txt; return model_info."""
    root = config["root"]
    record = champion_record(config)  # Week 7: reports/champion.json, else final_model.json
    setup_mlflow(config)
    model = mlflow.sklearn.load_model(record["registered"]["uri"])
    store = build_feature_store(config, record["features"])

    out = root / MODELS_DIR
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    store.to_parquet(out / "feature_store.parquet", index=False)
    reg = record["registered"]
    info = {
        "name": reg["name"], "version": str(reg["version"]), "uri": reg["uri"],
        "threshold": float(record["threshold"]), "features": record["features"],
        "n_features": len(record["features"]),
        "valid_auc": float(record["valid"]["valid_auc"]),
        "data_version": _data_version(config),
        "exported": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "n_applicants": len(store),
    }
    (out / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (root / "requirements-api.txt").write_text(pinned_requirements())
    return info


# PROVIDED
def _data_version(config: dict[str, Any]) -> str:
    """The data_version of the feature table the model was trained on (as in Week 3)."""
    from src.experiment import load_training_data

    return load_training_data(config).data_version


# PROVIDED
def main() -> None:
    config = load_week3_config()
    info = export(config)
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB"
             for p in sorted((config["root"] / MODELS_DIR).iterdir())}
    print(f"Exported {info['uri']} (version {info['version']}), threshold "
          f"{info['threshold']:.3f}, {info['n_features']} features, "
          f"{info['n_applicants']:,} applicants in the feature store")
    for name, size in sizes.items():
        print(f"  models/{name:<24}{size}")
    print(f"Wrote requirements-api.txt (Python {info['python']})")
    print("Next: upload the three models/ files to your bucket (docs/cloud_shell_week7.md).")


if __name__ == "__main__":
    main()
