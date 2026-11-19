"""Monitoring: compare what the live model received with what it was trained on.

Run it with ``python run.py monitor`` after downloading your service's prediction log to
data/logs/predictions.json (WEEK6.md says how). For each production batch in the log it
compares the features the model actually received (logged by your service) with the
model's training data (``X_train``), feature by feature, and the scores with the scores on
the validation set. It writes, in reports/monitoring/:

    summary.json          per batch: requests, approval rate, score shift, how many
                          features drifted, the top drifted features, Evidently's verdict
    drift_batchN.csv      every feature: PSI, KS test, missing shares, means (only these
                          summaries, no applicant rows, so it can be committed)
    drift_top.png         the most drifted features and the scores, training vs. batches
    evidently/            Evidently's interactive drift report per batch (open it in a
                          browser; large, so never committed)

Two measures of "did this feature change?":

- **PSI** (population stability index): split the training values into 10 bins with equal
  shares (plus one bin for missing values), and compare the share of production values in
  each bin with the training share. 0 means identical; the usual rule of thumb is < 0.1 no
  real change, 0.1 to 0.25 a moderate change, > 0.25 a major change. Missing values count:
  a feature that suddenly goes missing more often drifts, even if its other values do not.
- **KS test** (Kolmogorov-Smirnov): the largest gap between the two cumulative
  distributions (the statistic, 0 to 1), and the p-value for "both come from the same
  distribution". With thousands of rows even a tiny, harmless difference gets a small
  p-value, so read the statistic, not only the p-value.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.experiment import load_training_data, load_week3_config

LOGS = Path("data/logs/predictions.json")
PRODUCTION_DIR = Path("data/production")
MODELS = Path("models")
OUT = Path("reports/monitoring")
EVENT = "prediction"
EPS = 1e-4                       # a share of 0 would make the logarithm infinite
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
TOP = 10
MIN_BATCH = 100                  # fewer logged requests than this: too few to compare


# PROVIDED
def quantile_edges(reference: pd.Series | np.ndarray, bins: int = 10) -> np.ndarray:
    """Bin edges that split the reference's non-missing values into ``bins`` equal shares.

    Returns the inner edges only (``bins - 1`` of them, fewer if values repeat): values below
    the first edge go to bin 0, values at or above the last edge to the last bin.
    """
    values = pd.Series(reference, dtype=float).dropna().to_numpy()
    if len(values) == 0:
        return np.array([])
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))


# PROVIDED
def bin_shares(values: pd.Series | np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The share of ``values`` in each bin defined by ``edges``, plus a last bin for missing.

    Returns ``len(edges) + 2`` shares that add up to 1.
    """
    x = pd.Series(values, dtype=float).to_numpy()
    missing = np.isnan(x)
    counts = np.bincount(np.searchsorted(edges, x[~missing], side="right"),
                         minlength=len(edges) + 1)
    return np.append(counts, missing.sum()) / max(len(x), 1)


# TODO(student)
def psi(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, bins: int = 10,
        edges: np.ndarray | None = None) -> float:
    """Population stability index of ``current`` against ``reference``.

    Steps:
        1. If ``edges`` is None, compute them from the REFERENCE with
           ``quantile_edges(reference, bins)`` (always the reference: it defines "normal").
        2. ``e = bin_shares(reference, edges)`` and ``a = bin_shares(current, edges)``: the
           expected (training) and actual (production) share in each bin.
        3. Replace shares below ``EPS`` by ``EPS`` (``np.clip(e, EPS, None)``), so an empty
           bin does not make the logarithm infinite.
        4. PSI = sum over the bins of (a - e) * ln(a / e). Return it as a float.

    Every term is >= 0, so PSI is never negative, and it is 0 when the shares are equal.
    """
    raise NotImplementedError("TODO(student): the population stability index")


# PROVIDED
def category_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature: one bin per training category (missing as its own).

    Categories are turned into numbers (their position in the training categories) and
    passed to your ``psi`` with one edge between each pair of neighbours. A category never
    seen in training lands in the missing bin.
    """
    categories = sorted(reference.dropna().astype(str).unique())
    codes = {c: float(i) for i, c in enumerate(categories)}

    def encode(s: pd.Series) -> pd.Series:
        return s.astype("string").map(codes).astype(float)

    edges = np.arange(len(categories) - 1) + 0.5
    return psi(encode(reference), encode(current), edges=edges)


# PROVIDED
def load_logs(path: Path) -> tuple[pd.DataFrame, str]:
    """Read prediction records: a Cloud Logging download (a JSON list of entries, from
    ``gcloud logging read ... --format=json``) or a local file of JSON lines
    (PREDICTION_LOG_FILE). Returns one row per record, the features as columns, and the
    source ("cloud_logging" or "local_file")."""
    if not path.exists():
        raise SystemExit(f"{path} not found: download your service's log first (WEEK6.md)")
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        entries, source = json.loads(text), "cloud_logging"
        # Cloud Logging may lift some fields (timestamp, severity) out of the payload into
        # the entry itself: take them from there when they are missing.
        records = [{"timestamp": e.get("timestamp"), **e["jsonPayload"]} for e in entries
                   if isinstance(e.get("jsonPayload"), dict)]
    else:
        source = "local_file"
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = [r for r in records if r.get("event") == EVENT and isinstance(r.get("features"),
                                                                             dict)]
    if not records:
        raise SystemExit(f"{path} holds no prediction records: is log_prediction called in "
                         "/predict, and did you send the batches to this service?")
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("features", "overrides")}
                         for r in records])
    features = pd.DataFrame([r["features"] for r in records])
    logs = pd.concat([meta, features.add_prefix("f:")], axis=1)
    # One answer per applicant and model version: a request sent twice counts once.
    logs = (logs.sort_values("timestamp")
            .drop_duplicates(["SK_ID_CURR", "model_version"], keep="last")
            .reset_index(drop=True))
    return logs, source


# PROVIDED
def batch_of(ids: pd.Series) -> pd.Series:
    """Which production batch each logged applicant came from (0 = none of them)."""
    lookup: dict[int, int] = {}
    for b in (1, 2, 3):
        path = PRODUCTION_DIR / f"batch{b}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    lookup[int(json.loads(line)["SK_ID_CURR"])] = b
    if not lookup:
        raise SystemExit("data/production/batch*.jsonl not found: run python run.py pool")
    return ids.astype(int).map(lookup).fillna(0).astype(int)


# PROVIDED
def drift_table(reference: pd.DataFrame, current: pd.DataFrame,
                categorical: list[str]) -> pd.DataFrame:
    """One row per feature: PSI, KS statistic and p-value, missing shares, and means."""
    rows = []
    for col in reference.columns:
        ref, cur = reference[col], current[col]
        row: dict[str, Any] = {"feature": col,
                               "kind": "categorical" if col in categorical else "numeric",
                               "missing_ref": float(ref.isna().mean()),
                               "missing_cur": float(cur.isna().mean())}
        if col in categorical:
            row["psi"] = category_psi(ref, cur)
            row.update(ks_stat=np.nan, ks_pvalue=np.nan, mean_ref=np.nan, mean_cur=np.nan)
        else:
            ref, cur = ref.astype(float), cur.astype(float)
            row["psi"] = psi(ref, cur)
            if ref.notna().sum() and cur.notna().sum():
                ks = stats.ks_2samp(ref.dropna(), cur.dropna())
                row.update(ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue))
            else:
                row.update(ks_stat=np.nan, ks_pvalue=np.nan)
            row.update(mean_ref=float(ref.mean()), mean_cur=float(cur.mean()))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    table["level"] = np.select([table["psi"] >= PSI_MAJOR, table["psi"] >= PSI_MODERATE],
                               ["major", "moderate"], "none")
    return table


# PROVIDED
def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> dict:
    """Evidently's data drift report (saved as HTML), and its overall verdict."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset()]).run(reference_data=reference,
                                                   current_data=current)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    verdict: dict[str, Any] = {"html": str(path)}
    for metric in snapshot.dict()["metrics"]:
        if "DriftedColumnsCount" in str(metric.get("metric_name", "")):
            verdict.update(drifted_columns=int(metric["value"]["count"]),
                           drifted_share=round(float(metric["value"]["share"]), 3),
                           columns=len(reference.columns))
    return verdict


# PROVIDED
def plot_top(reference: pd.DataFrame, batches: dict[int, pd.DataFrame], features: list[str],
             ref_scores: np.ndarray, scores: dict[int, np.ndarray], path: Path) -> None:
    """Histograms of the most drifted numeric features and of the scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [*features, "score"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2))
    for ax, name in zip(np.atleast_1d(axes), panels, strict=True):
        ref = ref_scores if name == "score" else reference[name].astype(float).dropna()
        lo, hi = np.nanquantile(ref, [0.01, 0.99])
        grid = np.linspace(lo, hi, 30)
        ax.hist(ref, bins=grid, density=True, alpha=0.4, label="training", color="grey")
        for b, frame in batches.items():
            cur = scores[b] if name == "score" else frame[name].astype(float).dropna()
            ax.hist(cur, bins=grid, density=True, histtype="step", linewidth=1.6,
                    label=f"batch {b}")
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# PROVIDED
def run_monitoring(config: dict[str, Any], logs_path: Path, evidently: bool = True) -> dict:
    """Compare every logged batch with the training data; write reports/monitoring/."""
    root = config["root"]
    info = json.loads((root / MODELS / "model_info.json").read_text())
    features = info["features"]
    data = load_training_data(config, features)
    reference = data.X_train[features]
    model = joblib.load(root / MODELS / "model.joblib")
    ref_scores = model.predict_proba(data.X_valid[features])[:, 1]
    threshold = float(info["threshold"])

    logs, source = load_logs(root / logs_path)
    logs["batch"] = batch_of(logs["SK_ID_CURR"])
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "source": source, "logs_file": str(logs_path), "records": len(logs),
        "model_versions": sorted(map(str, logs["model_version"].unique())),
        "reference": {"rows": len(reference), "what": "X_train, the model's training rows",
                      "valid_approval_rate": round(float((ref_scores < threshold).mean()), 4),
                      "valid_mean_score": round(float(ref_scores.mean()), 4)},
        "thresholds": {"psi_moderate": PSI_MODERATE, "psi_major": PSI_MAJOR},
        "batches": {},
    }
    frames, scores, tables = {}, {}, {}
    counts = logs["batch"].value_counts()
    summary["not_compared"] = {"not_in_a_batch": int(counts.get(0, 0)),
                               "batches_with_too_few": {str(b): int(n) for b, n in counts.items()
                                                        if b > 0 and n < MIN_BATCH}}
    for b in sorted(int(x) for x, n in counts.items() if x > 0 and n >= MIN_BATCH):
        rows = logs[logs["batch"] == b]
        current = rows[[f"f:{f}" for f in features]].set_axis(features, axis=1)
        for col in data.numeric:
            current[col] = pd.to_numeric(current[col], errors="coerce")
        table = drift_table(reference, current, data.categorical)
        table.to_csv(out / f"drift_batch{b}.csv", index=False, float_format="%.5g")
        p = rows["probability"].astype(float).to_numpy()
        record = {
            "requests": len(rows), "first": rows["timestamp"].min(),
            "last": rows["timestamp"].max(),
            "approval_rate": round(float((rows["decision"] == "approve").mean()), 4),
            "mean_score": round(float(p.mean()), 4),
            "score_psi": round(psi(ref_scores, p), 4),
            "features_major": int((table["level"] == "major").sum()),
            "features_moderate": int((table["level"] == "moderate").sum()),
            "top": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in table.head(TOP)[["feature", "psi", "ks_stat", "missing_ref",
                                              "missing_cur", "mean_ref", "mean_cur",
                                              "level"]].to_dict("records")],
        }
        if evidently:
            record["evidently"] = evidently_report(reference, current,
                                                   out / "evidently" / f"batch{b}.html")
        summary["batches"][str(b)] = record
        frames[b], scores[b], tables[b] = current, p, table
    if not frames:
        raise SystemExit("none of the logged requests belong to a production batch: send "
                         "them with python run.py send --batch 1 (and 2)")
    last = tables[max(tables)]
    top_numeric = last[last["kind"] == "numeric"]["feature"].head(4).tolist()
    plot_top(reference, frames, top_numeric, ref_scores, scores, out / "drift_top.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


# PROVIDED
def main() -> None:
    parser = argparse.ArgumentParser(description="Drift report from the prediction log.")
    parser.add_argument("--logs", default=str(LOGS), help="the downloaded prediction log")
    parser.add_argument("--no-evidently", action="store_true",
                        help="skip the Evidently HTML reports (faster)")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    summary = run_monitoring(load_week3_config(), Path(args.logs),
                             evidently=not args.no_evidently)
    ref = summary["reference"]
    print(f"{summary['records']:,} logged predictions ({summary['source']}), model "
          f"version(s) {', '.join(summary['model_versions'])}")
    print(f"Validation set: approval rate {ref['valid_approval_rate']:.1%}, mean score "
          f"{ref['valid_mean_score']:.3f}\n")
    for b, rec in summary["batches"].items():
        ev = rec.get("evidently", {})
        ev_text = (f", Evidently: {ev['drifted_columns']} of {ev['columns']} columns drifted"
                   if "drifted_columns" in ev else "")
        print(f"Batch {b}: {rec['requests']:,} requests, approval {rec['approval_rate']:.1%}, "
              f"mean score {rec['mean_score']:.3f}, score PSI {rec['score_psi']:.3f}; "
              f"{rec['features_major']} features major, {rec['features_moderate']} moderate"
              f"{ev_text}")
        for t in rec["top"][:5]:
            print(f"    {t['feature']:<34} PSI {t['psi']:.3f}  missing "
                  f"{t['missing_ref']:.0%} -> {t['missing_cur']:.0%}  ({t['level']})")
    print(f"\nWrote {OUT}/summary.json, drift_batch*.csv, drift_top.png, evidently/*.html")


if __name__ == "__main__":
    main()
