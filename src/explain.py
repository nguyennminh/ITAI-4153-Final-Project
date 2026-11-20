"""Explanations: which features drive the model, and why one applicant was declined.

Run it with ``python run.py explain``. It computes **SHAP values** for the validation set:
for every applicant and every feature, how much that feature pushed this applicant's score
up (towards "reject") or down, compared with the average applicant. The contributions of
all features add up to the applicant's score (in log-odds, the model's internal scale).

It writes, in reports/explain/:

    shap_global.csv / .png   global importance: the mean absolute SHAP value per feature
    explanations.json / .md  three declined validation applicants (the most risky, a typical
                             one, and a borderline one), each with the reasons, in plain
                             language, that pushed their risk up the most

How it works: your registered model is the tuned gradient boosting pipeline wrapped in a
calibrator (Week 3). SHAP's TreeExplainer reads the trees of the boosting model directly, so
it is exact and fast. The calibrator only rescales the score (higher stays higher), so a
feature that raises the boosted score also raises the calibrated probability.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.baseline import clean_application
from src.compare import split_frame
from src.experiment import feature_table_path, load_week3_config

MODELS = Path("models")
OUT = Path("reports/explain")
N_REASONS = 3
GLOBAL_TOP = 20

# Plain-language names for features that often matter. Other features keep their column name.
PLAIN = {
    "EXT_SOURCE_1": "external credit score 1", "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3", "DAYS_BIRTH": "age",
    "DAYS_EMPLOYED": "time in the current job", "DAYS_ID_PUBLISH": "time since the ID was issued",
    "DAYS_REGISTRATION": "time since the registration changed",
    "DAYS_LAST_PHONE_CHANGE": "time since the phone number changed",
    "AMT_CREDIT": "the loan amount", "AMT_ANNUITY": "the yearly repayment",
    "AMT_GOODS_PRICE": "the price of the goods", "AMT_INCOME_TOTAL": "the yearly income",
    "CREDIT_INCOME_RATIO": "the loan as a multiple of income",
    "ANNUITY_INCOME_RATIO": "the yearly repayment as a share of income",
    "CREDIT_GOODS_RATIO": "the loan compared with the goods price",
    "ANNUITY_CREDIT_RATIO": "the yearly repayment as a share of the loan",
    "CODE_GENDER": "gender", "NAME_EDUCATION_TYPE": "education",
    "NAME_INCOME_TYPE": "income type", "ORGANIZATION_TYPE": "employer type",
    "OCCUPATION_TYPE": "occupation", "OWN_CAR_AGE": "the age of the applicant's car",
    "REGION_RATING_CLIENT_W_CITY": "the region's rating",
    # Week 2 features
    "BUREAU_COUNT": "credits at other lenders", "BUREAU_ACTIVE_COUNT": "active credits elsewhere",
    "BUREAU_AMT_DEBT_MEAN": "average debt at other lenders",
    "BUREAU_AMT_OVERDUE_SUM": "overdue amount at other lenders",
    "BUREAU_DAYS_CREDIT_MAX": "time since the latest credit elsewhere",
    "BUREAU_DAYS_OVERDUE_MAX": "longest overdue period elsewhere",
    "PREV_COUNT": "earlier applications here", "PREV_REFUSED_RATE": "share of earlier "
    "applications refused", "PREV_APPROVED_RATE": "share of earlier applications approved",
    "PREV_DAYS_DECISION_MAX": "time since the latest earlier application",
    "PREV_LAST_STATUS": "the latest earlier application's status",
    "BB_LATE_SHARE": "share of months late at other lenders",
    "BB_WORST_STATUS": "worst monthly status at other lenders",
    "POS_LATE_SHARE": "share of months late on earlier loans",
    "POS_COMPLETED_SHARE": "share of earlier loans completed", "POS_DPD_MAX": "most days "
    "past due on earlier loans", "INST_LATE_SHARE": "share of instalments paid late",
    "INST_DAYS_LATE_MAX": "most days late on an instalment",
    "INST_UNDERPAID_SHARE": "share of instalments underpaid",
    "INST_PAYMENT_DIFF_MEAN": "average shortfall per instalment",
    "CC_UTILIZATION_MEAN": "average credit card use (share of the limit)",
    "CC_UTILIZATION_MAX": "highest credit card use (share of the limit)",
    "CC_DRAWINGS_MEAN": "average monthly card withdrawals", "CC_DPD_MAX": "most days past due "
    "on a credit card",
}
YEARS = ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH")


# PROVIDED
def plain_name(feature: str) -> str:
    """A feature's name in plain language (its column name if there is none)."""
    return PLAIN.get(feature, feature)


# PROVIDED
def plain_value(feature: str, value: Any) -> str:
    """A feature's value in plain units: ages and durations in years or days, amounts with
    thousands separators, shares and scores with 3 significant digits."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, str):
        return value
    if feature in YEARS:
        return f"{abs(value) / 365.25:.1f} years"
    if "DAYS" in feature:
        return f"{abs(value):,.0f} days"
    if feature.startswith("AMT_"):
        return f"{value:,.0f}"
    return f"{value:.3g}"


# PROVIDED
def describe(feature: str, value: Any, typical: Any) -> str:
    """One feature, for a person: 'external credit score 3 is 0.12 (typical: 0.54)'."""
    return f"{plain_name(feature)} is {plain_value(feature, value)} " \
           f"(typical: {plain_value(feature, typical)})"


# TODO(student)
def explain_applicant(contributions: pd.Series, values: pd.Series, typical: pd.Series,
                      n: int = N_REASONS) -> list[str]:
    """The reasons, in plain language, that pushed one applicant's risk UP the most.

    Args:
        contributions: this applicant's SHAP value per feature (index: feature names).
            Positive = pushed the score towards "reject", negative = towards "approve".
        values: this applicant's feature values (same index).
        typical: the typical (median, or most common) training value of each feature.
        n: how many reasons.

    Returns:
        Up to ``n`` sentences, one per feature, for the features with the LARGEST POSITIVE
        contributions, the largest first. Each sentence is
        ``describe(feature, values[feature], typical[feature])`` followed by
        ``f" (raises the risk score by {contribution:.2f})"``. Features that lowered the risk
        (contribution <= 0) are never reasons for a decline, so there can be fewer than n.

    Hints:
        - ``contributions[contributions > 0].sort_values(ascending=False).head(n)``
    """
    raise NotImplementedError("TODO(student): the applicant's top reasons")


# PROVIDED
def validation_set(config: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """The validation applicants: id, target, and the model's features (cleaned)."""
    table = clean_application(pd.read_parquet(feature_table_path(config)))
    _, valid = split_frame(table, config)
    return valid[[config["id_col"], config["target_col"], *features]]


# PROVIDED
def inner_models(model: Any) -> list[Any]:
    """The fitted (preprocess, model) pipelines inside the calibrated model."""
    pipelines = [c.estimator for c in getattr(model, "calibrated_classifiers_", [])] or [model]
    for p in pipelines:
        if not hasattr(p, "named_steps") or "preprocess" not in p.named_steps:
            raise SystemExit("explain expects the Week 3 model: a pipeline with a "
                             "'preprocess' step, inside the calibrator")
    return pipelines


# PROVIDED
def _original(name: str, features: list[str]) -> str:
    """The model feature a transformed column comes from (num__X -> X, cat__X_value -> X)."""
    short = name.split("__", 1)[-1]
    matches = [f for f in features if short == f or short.startswith(f + "_")]
    return max(matches, key=len) if matches else short


# PROVIDED
def shap_values(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP values per applicant and model feature (log-odds), and the base value.

    With several calibration folds, their boosting models' SHAP values are averaged.
    """
    import shap

    features = list(X.columns)
    frames, bases = [], []
    for pipeline in inner_models(model):
        pre = pipeline.named_steps["preprocess"]
        booster = pipeline.steps[-1][1]
        transformed = pre.transform(X)
        names = [_original(n, features) for n in pre.get_feature_names_out()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(booster)
            values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:  # (rows, columns, classes): keep the "default" class
            values = values[:, :, -1]
        frame = pd.DataFrame(values, columns=names, index=X.index)
        frames.append(frame.T.groupby(level=0).sum().T.reindex(columns=features, fill_value=0))
        base = np.ravel(explainer.expected_value)[-1]
        bases.append(float(base))
    return sum(frames) / len(frames), float(np.mean(bases))


# PROVIDED
def typical_values(X: pd.DataFrame) -> pd.Series:
    """The median of each numeric feature, the most common value of each categorical one."""
    out = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            out[col] = float(X[col].median())
        else:
            mode = X[col].mode()
            out[col] = mode.iloc[0] if len(mode) else None
    return pd.Series(out)


# PROVIDED
def pick_declined(scored: pd.DataFrame) -> pd.DataFrame:
    """Three declined applicants: the riskiest, a typical decline, and a borderline one."""
    declined = scored[scored["decision"] == "reject"].sort_values("probability",
                                                                  ascending=False)
    if len(declined) < 3:
        raise SystemExit("fewer than 3 declined validation applicants")
    picks = [0, len(declined) // 2, len(declined) - 1]
    return declined.iloc[picks].assign(kind=["riskiest", "typical decline", "borderline"])


# PROVIDED
def plot_global(importance: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.head(GLOBAL_TOP)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("mean |SHAP value| (log-odds): average push on the score")
    ax.set_title(f"Global feature importance, top {len(top)}")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# PROVIDED
def run_explain(config: dict[str, Any]) -> dict[str, Any]:
    root, id_col = config["root"], config["id_col"]
    info = json.loads((root / MODELS / "model_info.json").read_text())
    features, threshold = info["features"], float(info["threshold"])
    model = joblib.load(root / MODELS / "model.joblib")
    valid = validation_set(config, features)
    X = valid[features]

    contributions, base = shap_values(model, X)
    importance = contributions.abs().mean().sort_values(ascending=False)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    importance.rename("mean_abs_shap").to_csv(out / "shap_global.csv", index_label="feature",
                                              float_format="%.5f")
    plot_global(importance, out / "shap_global.png")

    train, _ = split_frame(clean_application(pd.read_parquet(feature_table_path(config))), config)
    typical = typical_values(train[features])
    scored = valid[[id_col]].assign(probability=model.predict_proba(X)[:, 1])
    scored["decision"] = np.where(scored["probability"] >= threshold, "reject", "approve")
    applicants = []
    for idx, row in pick_declined(scored).iterrows():
        c, v = contributions.loc[idx], X.loc[idx]
        helped = c[c < 0].sort_values().head(2)
        applicants.append({
            id_col: int(row[id_col]), "kind": row["kind"],
            "probability": round(float(row["probability"]), 4), "decision": "reject",
            "reasons": explain_applicant(c, v, typical),
            "lowered_risk": [describe(f, v[f], typical[f]) for f in helped.index],
        })
    record = {"model_version": info["version"], "threshold": threshold,
              "data": "validation set", "n": len(X), "base_value_log_odds": round(base, 4),
              "global_top": importance.head(GLOBAL_TOP).index.tolist(),
              "applicants": applicants}
    (out / "explanations.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    (out / "explanations.md").write_text(render(record, id_col))
    return record


# PROVIDED
def render(record: dict[str, Any], id_col: str) -> str:
    lines = ["# Why these applicants were declined", "",
             f"Model version {record['model_version']}, threshold {record['threshold']:.3f}: "
             "an applicant whose estimated probability of default is at or above the "
             "threshold is declined. Written by python run.py explain.", ""]
    for a in record["applicants"]:
        lines += [f"## Applicant {a[id_col]} ({a['kind']})", "",
                  f"Estimated probability of default: {a['probability']:.1%} "
                  f"(threshold {record['threshold']:.1%}). Main reasons:", ""]
        lines += [f"{i}. {reason}" for i, reason in enumerate(a["reasons"], start=1)]
        if a["lowered_risk"]:
            lines += ["", "What lowered the risk: " + "; ".join(a["lowered_risk"]) + "."]
        lines.append("")
    return "\n".join(lines)


# PROVIDED
def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    record = run_explain(load_week3_config())
    print(f"SHAP values for {record['n']:,} validation applicants. Top features: "
          f"{', '.join(record['global_top'][:5])}\n")
    for a in record["applicants"]:
        print(f"Applicant {a['SK_ID_CURR']} ({a['kind']}), probability {a['probability']:.1%}:")
        for reason in a["reasons"]:
            print(f"  - {reason}")
    print(f"\nWrote {OUT}/shap_global.csv, shap_global.png, explanations.json, explanations.md")


if __name__ == "__main__":
    main()
