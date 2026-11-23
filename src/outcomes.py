"""The truth arrives: how did the live model really do on the production batches?

Run it with ``python run.py outcomes``. In Week 6 you could only watch the inputs and the
scores; whether the decisions were right was unknown, because a default shows up months
after the loan. Today the outcomes of batches 1 and 2 are released (in the simulation:
data/production/labels.parquet, written by ``python run.py pool``).

It joins the predictions your LIVE service logged (data/logs/predictions.json, the Cloud
Logging download from Week 6) with the true outcomes, so every number below is about the
decisions your service actually made, with the scores it actually returned. It compares
each batch with the validation set (the model's performance before it went live) and with
what the monitoring said in Week 6 (reports/monitoring/summary.json), and writes:

    reports/retrain/outcomes.json   the numbers
    reports/retrain/outcomes.md     the same as a table, for your decision record
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score  # noqa: F401  (for you to use)

from src.cost import expected_cost  # noqa: F401  (your Week 3 function, for you to use)
from src.experiment import load_training_data
from src.monitoring import LOGS, batch_of, load_logs
from src.retrain import load_retrain_config

LABELS = Path("data/production/labels.parquet")
MODELS = Path("models")
MONITORING = Path("reports/monitoring/summary.json")
OUT = Path("reports/retrain")
PERFORMANCE_KEYS = ("n", "n_default", "default_rate", "mean_score", "auc", "cost",
                    "approval_rate", "bad_rate_approved")


# TODO(student)
def performance(y_true: ArrayLike, scores: ArrayLike, threshold: float,
                costs: dict[str, float]) -> dict[str, float]:
    """How a model really did on a set of applicants whose outcomes are known.

    Args:
        y_true: 1 = the applicant defaulted, 0 = repaid.
        scores: the model's probability of default for each applicant.
        threshold: reject when the score is at or above it.
        costs: the cost matrix from final.yaml (approve_defaulter, reject_good).

    Returns:
        A dict with exactly the keys in ``PERFORMANCE_KEYS``, plain Python numbers:

        - ``n``: applicants; ``n_default``: how many defaulted (int)
        - ``default_rate``: share who defaulted; ``mean_score``: mean of the scores.
          If the model is calibrated, these two are close.
        - ``auc``: ROC-AUC (how well the scores rank defaulters above good payers)
        - ``cost``: expected cost per applicant at the threshold (your Week 3
          ``expected_cost``)
        - ``approval_rate``: share approved (score below the threshold)
        - ``bad_rate_approved``: among the APPROVED applicants, the share who defaulted:
          the loans that go bad. NaN if nobody was approved.

    Hints:
        - Convert with ``np.asarray`` first; ``approved = scores < threshold``.
        - ``y[approved].mean()`` is the bad rate among the approved.
    """
    raise NotImplementedError("TODO(student): the real performance on known outcomes")


# PROVIDED
def live_predictions(root: Path, logs_path: Path) -> pd.DataFrame:
    """The live log (one row per applicant), with its batch and the true outcome."""
    labels_path = root / LABELS
    if not labels_path.exists():
        raise SystemExit(f"{LABELS} not found: it is written by python run.py pool (Week 6)")
    logs, source = load_logs(root / logs_path)
    if source != "cloud_logging":
        print(f"NOTE: {logs_path} is a local file, not the Cloud Logging download")
    logs["batch"] = batch_of(logs["SK_ID_CURR"])
    labels = pd.read_parquet(labels_path)[["SK_ID_CURR", "TARGET"]]
    joined = logs.merge(labels, on="SK_ID_CURR", how="inner")
    return joined.assign(probability=joined["probability"].astype(float), source=source)


# PROVIDED
def run_outcomes(config: dict[str, Any], logs_path: Path) -> dict[str, Any]:
    root = config["root"]
    costs = config["week3"]["final"]["costs"]
    retrain = config["retrain"]
    info = json.loads((root / MODELS / "model_info.json").read_text())
    threshold = float(info["threshold"])

    data = load_training_data(config, info["features"])
    model = joblib.load(root / MODELS / "model.joblib")
    valid_scores = model.predict_proba(data.X_valid[info["features"]])[:, 1]
    record: dict[str, Any] = {
        "model_version": info["version"], "threshold": threshold,
        "validation": performance(data.y_valid, valid_scores, threshold, costs),
        "batches": {},
    }
    joined = live_predictions(root, logs_path)
    record["source"] = str(joined["source"].iloc[0]) if len(joined) else None
    monitoring = (json.loads((root / MONITORING).read_text())["batches"]
                  if (root / MONITORING).exists() else {})
    for batch in retrain["labeled_batches"]:
        rows = joined[joined["batch"] == batch]
        if rows.empty:
            raise SystemExit(f"no logged predictions with outcomes for batch {batch}: is "
                             f"{logs_path} the Week 6 download?")
        real = performance(rows["TARGET"], rows["probability"], threshold, costs)
        seen = monitoring.get(str(batch), {})
        record["batches"][str(batch)] = {
            "real": real,
            "monitoring_said": {k: seen.get(k) for k in ("approval_rate", "mean_score",
                                                         "score_psi", "features_major")},
            "cost_change_vs_validation": round(real["cost"] - record["validation"]["cost"], 4),
        }
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "outcomes.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    (out / "outcomes.md").write_text(render(record))
    return record


# PROVIDED
def _fmt(value: Any, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.1%}" if pct else (f"{value:.3f}" if isinstance(value, float) else str(value))


# PROVIDED
def render(record: dict[str, Any]) -> str:
    rows = [("validation (before going live)", record["validation"])] + [
        (f"batch {b} (live)", rec["real"]) for b, rec in record["batches"].items()]
    lines = ["# Real performance of the live model", "",
             f"Model version {record['model_version']}, threshold {record['threshold']:.3f}. "
             "Batches: the predictions the live service logged, joined with the true "
             "outcomes. Written by python run.py outcomes.", "",
             "| Applicants | n | Default rate | Mean score | AUC | Cost per applicant | "
             "Approval rate | Bad rate among approved |",
             "|---|---|---|---|---|---|---|---|"]
    for name, p in rows:
        lines.append(f"| {name} | {p['n']} | {_fmt(p['default_rate'], True)} | "
                     f"{_fmt(p['mean_score'])} | {_fmt(p['auc'])} | {_fmt(p['cost'])} | "
                     f"{_fmt(p['approval_rate'], True)} | "
                     f"{_fmt(p['bad_rate_approved'], True)} |")
    lines += ["", "## What the monitoring said in Week 6 (before the outcomes)", "",
              "| Batch | Approval rate | Mean score | Score PSI | Features with major drift |",
              "|---|---|---|---|---|"]
    for b, rec in record["batches"].items():
        m = rec["monitoring_said"]
        lines.append(f"| {b} | {_fmt(m['approval_rate'], True)} | {_fmt(m['mean_score'])} | "
                     f"{_fmt(m['score_psi'])} | {_fmt(m['features_major'])} |")
    return "\n".join(lines) + "\n"


# PROVIDED
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Real performance of the live model.")
    parser.add_argument("--logs", default=str(LOGS), help="the Week 6 prediction log")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    record = run_outcomes(load_retrain_config(), Path(args.logs))
    print((Path(OUT) / "outcomes.md").read_text())
    print(f"Wrote {OUT}/outcomes.json and outcomes.md")
    for b, rec in record["batches"].items():
        print(f"Batch {b}: cost per applicant {rec['real']['cost']:.3f} "
              f"({rec['cost_change_vs_validation']:+.3f} vs validation)")


if __name__ == "__main__":
    main()
