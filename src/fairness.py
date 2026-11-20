"""Fairness: does the model treat groups of applicants differently?

Run it with ``python run.py fairness``. On the validation set (applicants the model never
trained on, with known outcomes) it scores every applicant, applies your threshold, and
compares groups by **gender** (CODE_GENDER) and by **age band** (from DAYS_BIRTH):

    approval_rate   share approved                       (who gets a loan)
    default_rate    share who really defaulted           (the base rate of the group)
    mean_score      mean estimated probability of default
                    (close to default_rate = well calibrated for that group)
    tpr             share of the group's DEFAULTERS that the model rejects  (risk caught)
    fpr             share of the group's GOOD payers that the model rejects (wrongly declined)

Every rate comes with a **95% Wilson interval**: the range the true rate plausibly lies in,
given how few applicants the group has. The validation set has about 1,000 applicants and
only about 80 defaulters, so a group's TPR may rest on a handful of people: a wide interval
means "we cannot tell yet", not "no difference".

It also checks for **proxies**: even without CODE_GENDER, can the other features tell
gender apart? If they can, removing the gender column would not remove gender from the model.

It writes reports/fairness/fairness.json, fairness.md (the tables), and fairness.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.experiment import load_week3_config

MODELS = Path("models")
OUT = Path("reports/fairness")
AGE_BANDS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
MIN_GROUP = 20            # smaller groups are reported as too small, not compared
FEW_DEFAULTERS = 30       # a TPR based on fewer defaulters than this is marked unreliable
GROUPS = {"gender": "CODE_GENDER", "age_band": "AGE_BAND"}


# TODO(student)
def group_table(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per group with its size, outcomes, and error rates.

    Args:
        frame: one row per applicant, with columns ``group_col`` (the group), ``y`` (1 = the
            applicant defaulted), ``p`` (the estimated probability of default) and
            ``reject`` (True = the model declines them).
        group_col: the column that holds the group.

    Returns:
        A DataFrame with one row per group, sorted by group, and the columns:

        - ``group``: the group's value
        - ``n``: applicants; ``n_default``: how many defaulted (y == 1); ``n_good``: the rest
        - ``approval_rate``: share NOT rejected
        - ``default_rate``: share with y == 1
        - ``mean_score``: mean of p
        - ``tpr``: among the defaulters, the share rejected (NaN if there are none)
        - ``fpr``: among the good payers, the share rejected (NaN if there are none)

    Hints:
        - ``for group, rows in frame.groupby(group_col):`` then build one dict per group,
          and ``pd.DataFrame(list_of_dicts)``.
        - ``rows.loc[rows["y"] == 1, "reject"].mean()`` is the TPR (NaN for no rows).
    """
    raise NotImplementedError("TODO(student): the per-group table")


# PROVIDED
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a rate of k successes out of n."""
    if n == 0 or np.isnan(k):
        return (np.nan, np.nan)
    rate = k / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# PROVIDED
def add_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """Wilson intervals for the approval rate, TPR and FPR, and a reliability note."""
    out = table.copy()
    for rate, size in (("approval_rate", "n"), ("tpr", "n_default"), ("fpr", "n_good")):
        bounds = [wilson(r * n, n) for r, n in zip(out[rate], out[size], strict=True)]
        out[f"{rate}_lo"] = [b[0] for b in bounds]
        out[f"{rate}_hi"] = [b[1] for b in bounds]
    out["note"] = np.select(
        [out["n"] < MIN_GROUP, out["n_default"] < FEW_DEFAULTERS],
        [f"fewer than {MIN_GROUP} applicants: too small to compare",
         f"fewer than {FEW_DEFAULTERS} defaulters: TPR unreliable"], "")
    return out


# PROVIDED
def gaps(table: pd.DataFrame) -> dict[str, float]:
    """The largest difference between groups that are large enough to compare."""
    usable = table[table["n"] >= MIN_GROUP]
    out = {}
    for rate in ("approval_rate", "tpr", "fpr", "default_rate"):
        values = usable[rate].dropna()
        out[f"{rate}_gap"] = round(float(values.max() - values.min()), 4) if len(values) else None
    approvals = usable["approval_rate"]
    out["approval_ratio"] = round(float(approvals.min() / approvals.max()), 4) \
        if len(approvals) and approvals.max() > 0 else None
    return out


# PROVIDED
def proxy_check(train: pd.DataFrame, valid: pd.DataFrame, features: list[str],
                categorical: list[str], seed: int) -> dict[str, Any]:
    """How well do the OTHER features predict gender? (AUC on the validation set.)"""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    keep = {"F", "M"}
    tr, va = train[train["CODE_GENDER"].isin(keep)], valid[valid["CODE_GENDER"].isin(keep)]
    others = [f for f in features if f != "CODE_GENDER"]
    cats = [c for c in categorical if c in others]
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                     unknown_value=-1,
                                                     encoded_missing_value=-1), cats)],
                             remainder="passthrough")
    model = Pipeline([("preprocess", prep),
                      ("model", HistGradientBoostingClassifier(max_iter=200,
                                                               random_state=seed))])
    model.fit(tr[others], tr["CODE_GENDER"] == "M")
    auc = roc_auc_score(va["CODE_GENDER"] == "M", model.predict_proba(va[others])[:, 1])
    single = {}
    for f in others:
        if f in cats or va[f].notna().sum() < 50:
            continue
        x = va[f].fillna(va[f].median())
        if x.nunique() > 1:
            single[f] = abs(roc_auc_score(va["CODE_GENDER"] == "M", x) - 0.5) + 0.5
    strongest = sorted(single, key=single.get, reverse=True)[:5]
    return {"target": "CODE_GENDER (M vs F)", "auc": round(float(auc), 4),
            "strongest_single_features": {f: round(single[f], 3) for f in strongest},
            "gender_is_a_model_feature": "CODE_GENDER" in features}


# PROVIDED
def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25
    return pd.cut(years, AGE_BANDS, labels=AGE_LABELS, right=False).astype(str)


# PROVIDED
def plot(tables: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 * len(tables), figsize=(14, 3.4))
    for i, (attribute, table) in enumerate(tables.items()):
        table = table[table["n"] >= MIN_GROUP]
        for ax, rate, title in ((axes[2 * i], "approval_rate", "approval rate"),
                                (axes[2 * i + 1], "fpr", "good payers declined (FPR)")):
            lo = table[rate] - table[f"{rate}_lo"]
            hi = table[f"{rate}_hi"] - table[rate]
            ax.bar(table["group"].astype(str), table[rate], yerr=[lo, hi], capsize=4,
                   color="#4C72B0")
            ax.set_title(f"{title}\nby {attribute}", fontsize=9)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", labelsize=8, rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# PROVIDED
def render(record: dict[str, Any]) -> str:
    lines = ["# Fairness tables", "",
             f"Model version {record['model_version']}, threshold {record['threshold']:.3f}, "
             f"{record['n']:,} validation applicants. Written by python run.py fairness. "
             "Brackets: 95% Wilson interval.", ""]
    for attribute, rows in record["groups"].items():
        lines += [f"## By {attribute}", "",
                  "| Group | n | Defaulters | Approval rate | Default rate | Mean score | "
                  "TPR (risk caught) | FPR (good payers declined) | Note |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            def rate(name: str, row: dict = r) -> str:
                if row[name] is None or np.isnan(row[name]):
                    return "n/a"
                return f"{row[name]:.1%} [{row[name + '_lo']:.0%}, {row[name + '_hi']:.0%}]"
            lines.append(f"| {r['group']} | {r['n']} | {r['n_default']} | "
                         f"{rate('approval_rate')} | {r['default_rate']:.1%} | "
                         f"{r['mean_score']:.3f} | {rate('tpr')} | {rate('fpr')} | {r['note']} |")
        g = record["gaps"][attribute]
        lines += ["", f"Largest gaps: approval {g['approval_rate_gap']}, TPR {g['tpr_gap']}, "
                  f"FPR {g['fpr_gap']}; lowest / highest approval rate: {g['approval_ratio']}.",
                  ""]
    p = record["proxy"]
    lines += ["## Proxy check", "",
              f"Predicting {p['target']} from the other {record['n_features'] - 1} features: "
              f"AUC {p['auc']:.3f} (0.5 = they carry no information about gender). Strongest "
              "single features: " + ", ".join(f"{f} ({a})" for f, a in
                                              p["strongest_single_features"].items()) + ".", ""]
    return "\n".join(lines)


# PROVIDED
def run_fairness(config: dict[str, Any]) -> dict[str, Any]:
    from src.baseline import clean_application
    from src.compare import split_frame
    from src.experiment import feature_table_path, load_training_data

    root, target = config["root"], config["target_col"]
    info = json.loads((root / MODELS / "model_info.json").read_text())
    features, threshold = info["features"], float(info["threshold"])
    model = joblib.load(root / MODELS / "model.joblib")
    table = clean_application(pd.read_parquet(feature_table_path(config)))
    train, valid = split_frame(table, config)

    frame = pd.DataFrame({"CODE_GENDER": valid["CODE_GENDER"].astype(str),
                          "AGE_BAND": age_band(valid["DAYS_BIRTH"]),
                          "y": valid[target].astype(int),
                          "p": model.predict_proba(valid[features])[:, 1]})
    frame["reject"] = frame["p"] >= threshold
    tables = {name: add_intervals(group_table(frame, col)) for name, col in GROUPS.items()}
    categorical = load_training_data(config, features).categorical
    record = {
        "model_version": info["version"], "threshold": threshold, "data": "validation set",
        "n": len(frame), "n_features": len(features),
        "groups": {name: t.replace({np.nan: None}).to_dict("records")
                   for name, t in tables.items()},
        "gaps": {name: gaps(t) for name, t in tables.items()},
        "proxy": proxy_check(train, valid, features, categorical, config["seed"]),
    }
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    (out / "fairness.md").write_text(render(record))
    plot(tables, out / "fairness.png")
    return record


# PROVIDED
def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    run_fairness(load_week3_config())
    print((Path(OUT) / "fairness.md").read_text())
    print(f"Wrote {OUT}/fairness.json, fairness.md, fairness.png")


if __name__ == "__main__":
    main()
