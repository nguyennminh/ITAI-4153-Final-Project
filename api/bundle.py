"""Load everything the service needs at startup: the model, its info, and the feature store.

# PROVIDED: this whole module is finished. Read it and reuse it.

``python run.py export`` writes three files (the "bundle") into models/:

    model.joblib            the registered Week 3 model (scikit-learn pipeline)
    model_info.json         its version, threshold, features, and validation metrics
    feature_store.parquet   one row of model features per applicant, keyed by SK_ID_CURR

The service reads them from one of two places:

- ``MODEL_BUCKET`` set (on Cloud Run), for example ``gs://credit-risk-models-myproject``:
  the files are downloaded from that private Cloud Storage bucket when the service starts.
- otherwise from the folder ``MODEL_DIR`` (default ``models/``), on your laptop.

The service never touches MLflow or the raw data: Cloud Run cannot reach your laptop.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BUNDLE_FILES = ("model.joblib", "model_info.json", "feature_store.parquet")
ID_COL = "SK_ID_CURR"


@dataclass
class Bundle:
    """The loaded model, its info, and the feature store (indexed by applicant id)."""

    model: Any
    info: dict[str, Any]
    store: pd.DataFrame
    source: str

    @property
    def features(self) -> list[str]:
        """The model's input columns, in the order it was trained on."""
        return self.info["features"]


# PROVIDED
def download_bundle(bucket_uri: str, target: Path) -> Path:
    """Copy the bundle files from ``gs://bucket[/prefix]`` into the folder ``target``."""
    from google.cloud import storage  # imported here: only needed on Cloud Run

    bucket_name, _, prefix = bucket_uri.removeprefix("gs://").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    target.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        blob_name = f"{prefix.strip('/')}/{name}" if prefix.strip("/") else name
        bucket.blob(blob_name).download_to_filename(str(target / name))
    return target


# PROVIDED
def load_bundle(model_dir: str | Path | None = None) -> Bundle:
    """Load the bundle from ``MODEL_BUCKET`` if it is set, else from a local folder.

    Raises:
        FileNotFoundError: if a bundle file is missing (run ``python run.py export``).
    """
    bucket = os.environ.get("MODEL_BUCKET")
    if bucket:
        folder = download_bundle(bucket, Path(tempfile.mkdtemp(prefix="bundle-")))
        source = bucket
    else:
        folder = Path(model_dir or os.environ.get("MODEL_DIR", "models"))
        source = str(folder)
    missing = [name for name in BUNDLE_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"bundle files missing in {source}: {missing}. "
                                "Run python run.py export.")
    info = json.loads((folder / "model_info.json").read_text())
    store = pd.read_parquet(folder / "feature_store.parquet").set_index(ID_COL)
    model = joblib.load(folder / "model.joblib")
    return Bundle(model=model, info=info, store=store, source=source)
