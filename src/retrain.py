"""Retrain: the challenger. Same recipe as the champion, more (and newer) data.

# PROVIDED: this whole module is finished.

Run it with ``python run.py retrain``. The champion is the model your service runs (the
alias ``staging``; its record is reports/champion.json, or reports/final_model.json until
you first promote). The challenger is trained with exactly the champion's recipe (the same
model spec, calibration, and threshold rule from Week 3), on:

    the 4,000 training applicants (as always)
  + the labeled production applicants (config/retrain.yaml: batches 1 and 2),
    EXCEPT a fixed 30% of them: the "recent" slice, kept aside so that champion and
    challenger can be compared on recent applicants that NEITHER of them trained on.

The validation set stays untouched, as in every week: it is the other fixed slice.

The challenger is logged to MLflow (run "challenger") and registered as a new version of
``credit-risk`` with the alias ``challenger``. It does NOT serve: promoting it is a decision
(``python run.py champion``, then ``python run.py promote``). It writes
reports/retrain/challenger.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

from src.calibration import calibrate
from src.cost import choose_threshold, cost_curve, decision_summary, threshold_grid
from src.evaluation import cross_validate_auc, validation_metrics
from src.experiment import TrainingData, load_training_data, load_week3_config
from src.models import describe_spec, make_model

PRODUCTION = Path("data/production")
OUT = Path("reports/retrain")
CHAMPION = Path("reports/champion.json")
CHALLENGER = OUT / "challenger.json"
CHALLENGER_RUN = "challenger"


# PROVIDED
def load_retrain_config() -> dict[str, Any]:
    """The Week 3 config plus config/retrain.yaml as ``config["retrain"]``."""
    config = load_week3_config()
    path = config["root"] / "config" / "retrain.yaml"
    if not path.exists():
        raise SystemExit("config/retrain.yaml not found: copy it from the Week 7 zip")
    config["retrain"] = yaml.safe_load(path.read_text())
    return config


# PROVIDED
def champion_record(config: dict[str, Any]) -> dict[str, Any]:
    """The record of the model the service runs: champion.json once you have promoted or
    rolled back, else the Week 3 final_model.json."""
    root = config["root"]
    for path in (root / CHAMPION, config["week3"]["outputs"]["final_model"]):
        if path.exists():
            return json.loads(path.read_text())
    raise SystemExit("no champion: reports/final_model.json not found (Week 3)")


# PROVIDED
def recent_holdout(ids: pd.Series, share: float, seed: int) -> np.ndarray:
    """True for the fixed share of applicants kept aside as the recent slice.

    Decided by applicant id with a fixed seed: the same ids always give the same slice.
    """
    order = np.sort(ids.astype(int).unique())
    rng = np.random.default_rng(seed)
    held = set(order[rng.random(len(order)) < share])
    return ids.astype(int).isin(held).to_numpy()


# PROVIDED
def production_rows(config: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """The labeled production applicants: id, batch, TARGET, the model features, recent."""
    root, retrain = config["root"], config["retrain"]
    extension, labels = (root / PRODUCTION / "store_extension.parquet",
                         root / PRODUCTION / "labels.parquet")
    if not (extension.exists() and labels.exists()):
        raise SystemExit("data/production/ is incomplete: python run.py pool (Week 6)")
    rows = pd.read_parquet(extension).merge(pd.read_parquet(labels), on="SK_ID_CURR")
    rows = rows[rows["batch"].isin(retrain["labeled_batches"])].reset_index(drop=True)
    rows["recent"] = recent_holdout(rows["SK_ID_CURR"], retrain["recent_holdout_share"],
                                    retrain["holdout_seed"])
    return rows[["SK_ID_CURR", "batch", "TARGET", "recent", *features]]


@dataclass
class RetrainData:
    """Everything the challenger is trained and compared on."""

    data: TrainingData            # the fixed train / validation split (Week 1)
    production: pd.DataFrame      # labeled production applicants, with the 'recent' flag
    X: pd.DataFrame               # challenger training rows: train + production (not recent)
    y: pd.Series

    @property
    def recent(self) -> pd.DataFrame:
        return self.production[self.production["recent"]]


# PROVIDED
def retrain_data(config: dict[str, Any], features: list[str]) -> RetrainData:
    data = load_training_data(config, features)
    production = production_rows(config, features)
    new = production[~production["recent"]]
    columns = list(data.X_train.columns)
    X = pd.concat([data.X_train, new[columns]], ignore_index=True)
    y = pd.concat([data.y_train, new["TARGET"]], ignore_index=True).astype(int)
    return RetrainData(data=data, production=production, X=X, y=y)


# PROVIDED
def build_challenger(config: dict[str, Any], champion: dict[str, Any],
                     rd: RetrainData) -> tuple[Any, dict[str, Any]]:
    """Fit the champion's recipe on the challenger's rows; return the model and its record."""
    week3, seed = config["week3"], config["seed"]
    costs = week3["final"]["costs"]
    spec, method = champion["model"], champion["calibration"]
    calibrated = calibrate(make_model(spec, rd.data.numeric, rd.data.categorical, config),
                           method, week3["cv_folds"], seed)
    cv = cross_validate_auc(calibrated, rd.X, rd.y, week3["cv_folds"], seed)
    threshold = choose_threshold(cost_curve(rd.y, cv["oof"], threshold_grid(config), costs))
    model = clone(calibrated).fit(rd.X, rd.y)
    scores = model.predict_proba(rd.data.X_valid)[:, 1]
    record = {
        "model": spec, "calibration": method, "threshold": threshold, "costs": costs,
        "features": list(champion["features"]), "n_features": len(champion["features"]),
        "trained_on": {"train_split": len(rd.data.X_train),
                       "production": int((~rd.production["recent"]).sum()),
                       "labeled_batches": config["retrain"]["labeled_batches"],
                       "total": len(rd.X), "defaults": int(rd.y.sum())},
        "recent_slice": {"applicants": int(rd.production["recent"].sum()),
                         "defaults": int(rd.recent["TARGET"].sum())},
        "cv_auc_mean": cv["mean"], "cv_auc_std": cv["std"],
        "valid": {**validation_metrics(rd.data.y_valid, scores, week3["n_bootstrap"], seed),
                  **decision_summary(rd.data.y_valid, scores, threshold, costs)},
        "champion_version": str(champion["registered"]["version"]),
    }
    return model, record


# PROVIDED
def register_challenger(model: Any, record: dict[str, Any], rd: RetrainData,
                        config: dict[str, Any]) -> dict[str, Any]:
    """Log the challenger to MLflow and register it under the alias 'challenger'."""
    import mlflow
    from mlflow import MlflowClient

    from src.tracking import log_sklearn_model, run_tags, setup_mlflow

    setup_mlflow(config)
    name = config["week3"]["final"]["registry"]["name"]
    alias = config["retrain"]["registry"]["challenger_alias"]
    with mlflow.start_run(run_name=CHALLENGER_RUN, tags={
            **run_tags(config, rd.data.data_version, describe_spec(record["model"])),
            "spec": json.dumps(record["model"]),
            "labeled_batches": json.dumps(record["trained_on"]["labeled_batches"])}) as run:
        mlflow.log_params({"model": record["model"]["model"],
                           "calibration": record["calibration"],
                           "threshold": record["threshold"],
                           "n_train_rows": record["trained_on"]["total"],
                           **(record["model"].get("params") or {})})
        mlflow.log_metrics({"cv_auc_mean": record["cv_auc_mean"],
                            **{k.replace(".", ""): v for k, v in record["valid"].items()
                               if isinstance(v, (int, float))}})
        info = log_sklearn_model(model, registered_model_name=name,
                                 input_example=rd.data.X_valid.head(3))
        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(name, alias, version)
        client.set_model_version_tag(name, version, "threshold", str(record["threshold"]))
        client.set_model_version_tag(name, version, "role", "challenger")
        record["run_id"] = run.info.run_id
    return {"name": name, "version": version, "alias": alias,
            "uri": f"models:/{name}@{alias}"}


# PROVIDED
def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    config = load_retrain_config()
    champion = champion_record(config)
    rd = retrain_data(config, champion["features"])
    print(f"Champion: version {champion['registered']['version']}. Training the challenger "
          f"on {len(rd.X):,} applicants ({len(rd.data.X_train):,} train + "
          f"{len(rd.X) - len(rd.data.X_train):,} labeled production), keeping "
          f"{len(rd.recent):,} recent applicants aside ...")
    model, record = build_challenger(config, champion, rd)
    record["registered"] = register_challenger(model, record, rd, config)
    out = config["root"] / OUT
    out.mkdir(parents=True, exist_ok=True)
    (config["root"] / CHALLENGER).write_text(json.dumps(record, indent=2, default=float) + "\n")
    v = record["valid"]
    print(f"Challenger registered: {record['registered']['uri']} = version "
          f"{record['registered']['version']}")
    print(f"  threshold {record['threshold']:.3f}, CV AUC {record['cv_auc_mean']:.3f}, "
          f"validation AUC {v['valid_auc']:.3f}, cost {v['cost_at_threshold']:.3f}")
    print(f"Wrote {CHALLENGER}. Next: python run.py champion (compare, gate, decide).")


if __name__ == "__main__":
    main()
