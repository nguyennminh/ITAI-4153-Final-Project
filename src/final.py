"""Build, check, and register the model the later weeks will serve.

Run it with ``python run.py final`` (or ``python -m src.final``). From config/final.yaml it:

1. takes the model spec (the best Optuna trial by default),
2. wraps the model in calibration (your ``calibrate``),
3. chooses the threshold from calibrated out-of-fold scores on the TRAIN split,
4. fits on the whole train split and scores the validation split ONCE: AUC with its 95%
   interval, expected cost at the threshold, calibration before and after, speed, size,
5. scores the Monday run named in ``compare_to`` (tree_tuned) the same way, with the paired
   interval of the AUC difference,
6. logs everything as the MLflow run "final_model" and registers the model (your
   ``register_model``) under ``registry.name`` with the alias ``registry.alias``,
7. writes reports/final_model.json and reports/figures/reliability.png.

Week 5 loads the model by its alias (``models:/credit-risk@staging``), so registering a
better model later only means moving the alias.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient  # noqa: F401  (for you to use)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.calibration import calibrate, calibration_summary, reliability_table
from src.cost import choose_threshold, cost_curve, decision_summary, threshold_grid
from src.evaluation import (
    cross_validate_auc,
    measure_latency_ms,
    model_size_kb,
    validation_metrics,
)
from src.experiment import load_training_data, load_week3_config
from src.models import describe_spec, fit_model, make_model
from src.plots import plot_reliability
from src.tracking import (
    latest_top_level,
    load_runs,
    log_sklearn_model,  # noqa: F401  (for you to use)
    run_tags,
    setup_mlflow,
)
from src.tune import model_spec
from src.uncertainty import bootstrap_auc_diff_ci
from src.utils import DEFAULT_CONFIG

FINAL_RUN = "final_model"


# TODO(student)
def register_model(
    model: Any, X_example: pd.DataFrame, config: dict[str, Any], threshold: float,
) -> dict[str, Any]:
    """Save the fitted model with the OPEN run, register it, and point the alias at it.

    1. ``log_sklearn_model(model, registered_model_name=<name>, input_example=X_example)``
       (from src.tracking) saves the model and creates a new version of the registered
       model. The input example lets MLflow record the columns the model expects.
    2. The new version number is ``info.registered_model_version``.
    3. With ``client = MlflowClient()``: ``client.set_registered_model_alias(name, alias,
       version)`` so the alias points at this version, and
       ``client.set_model_version_tag(name, version, "threshold", str(threshold))`` so the
       decision rule travels with the model.

    Name and alias come from final.yaml ``registry``.

    Args:
        model: The fitted (calibrated) model.
        X_example: A few rows of input, for example ``data.X_valid.head(3)``.
        config: The loaded Week 3 config.
        threshold: The chosen decision threshold.

    Returns:
        ``{"name": ..., "version": <str>, "alias": ..., "uri": "models:/<name>@<alias>"}``.
    """

    registry = config["week3"]["final"]["registry"]
    name = registry["name"]
    alias = registry["alias"]

    info = log_sklearn_model(
        model,
        registered_model_name=name,
        input_example=X_example,
    )

    version = str(info.registered_model_version)

    client = MlflowClient()
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, "threshold", str(threshold))

    return {
        "name": name,
        "version": version,
        "alias": alias,
        "uri": f"models:/{name}@{alias}",
    }

    raise NotImplementedError("TODO(student): register the model and set its alias")


# PROVIDED
def score_comparison_run(config: dict[str, Any], name: str) -> tuple[np.ndarray, dict]:
    """Refit the Monday run ``name`` from its logged spec and score the validation split."""
    runs = latest_top_level(load_runs(config))
    match = runs[runs["run_name"] == name]
    if match.empty or "tags.spec" not in match.columns:
        raise KeyError(f"no run named '{name}' with a spec tag: python run.py train {name}")
    spec = json.loads(match.iloc[0]["tags.spec"])
    data = load_training_data(config)
    model = fit_model(make_model(spec, data.numeric, data.categorical, config),
                      data.X_train, data.y_train, spec.get("undersample"), config["seed"])
    scores = model.predict_proba(data.X_valid)[:, 1]
    p50, p95 = measure_latency_ms(model, data.X_valid)
    return scores, {"run": name, "spec": spec, "n_features": len(data.features),
                    **validation_metrics(data.y_valid, scores, config["week3"]["n_bootstrap"],
                                         config["seed"]),
                    "latency_p50_ms": p50, "latency_p95_ms": p95,
                    "model_size_kb": model_size_kb(model)}


# PROVIDED
def run_final(config: dict[str, Any]) -> dict[str, Any]:
    """Steps 1-7 in the module docstring. Returns what it writes to final_model.json."""
    final, week3 = config["week3"]["final"], config["week3"]
    seed, folds, costs = config["seed"], week3["cv_folds"], final["costs"]
    spec = model_spec(config)
    if spec.get("undersample"):
        raise ValueError("the final model must train on the real default rate: calibrated "
                         "probabilities from undersampled rows are too high. Remove "
                         "undersample from the final spec.")
    data = load_training_data(config)
    base = make_model(spec, data.numeric, data.categorical, config)
    calibrated = calibrate(base, final["calibration"], folds, seed)
    out = week3["outputs"]

    with mlflow.start_run(run_name=FINAL_RUN, tags={
            **run_tags(config, data.data_version, describe_spec(spec)),
            "spec": json.dumps(spec)}) as run:
        # 3. Threshold from calibrated out-of-fold scores on the train split.
        cv = cross_validate_auc(calibrated, data.X_train, data.y_train, folds, seed)
        curve = cost_curve(data.y_train, cv["oof"], threshold_grid(config), costs)
        threshold = choose_threshold(curve)

        # 4. Fit on the whole train split; score the validation split once.
        raw_model = clone(base).fit(data.X_train, data.y_train)
        model = clone(calibrated).fit(data.X_train, data.y_train)
        raw = raw_model.predict_proba(data.X_valid)[:, 1]
        scores = model.predict_proba(data.X_valid)[:, 1]
        y_valid = data.y_valid.to_numpy()
        p50, p95 = measure_latency_ms(model, data.X_valid)
        plot = plot_reliability({"before calibration": reliability_table(y_valid, raw),
                                 f"after ({final['calibration']})":
                                     reliability_table(y_valid, scores)},
                                out["reliability_plot"], "Reliability on the validation split")

        # 5. The Monday comparison run, on the same validation rows.
        tree_scores, tree = score_comparison_run(config, final["compare_to"])
        low, high = bootstrap_auc_diff_ci(y_valid, scores, tree_scores, week3["n_bootstrap"],
                                          seed)
        tree["auc_diff"] = float(roc_auc_score(y_valid, scores) - tree["valid_auc"])
        tree["auc_diff_ci"] = [low, high]

        record = {
            "run_id": run.info.run_id, "model": spec, "calibration": final["calibration"],
            "threshold": threshold, "costs": costs,
            "features": data.features, "n_features": len(data.features),
            "cv_auc_mean": cv["mean"], "cv_auc_std": cv["std"],
            "valid": {**validation_metrics(y_valid, scores, week3["n_bootstrap"], seed),
                      **decision_summary(y_valid, scores, threshold, costs)},
            "calibration_valid": {"before": calibration_summary(y_valid, raw),
                                  "after": calibration_summary(y_valid, scores)},
            "latency_p50_ms": p50, "latency_p95_ms": p95, "model_size_kb": model_size_kb(model),
            "compare_to": tree,
        }
        mlflow.log_params({"model": spec["model"], "calibration": final["calibration"],
                           "threshold": threshold, "n_features": len(data.features),
                           **(spec.get("params") or {})})
        mlflow.log_metrics({
            "cv_auc_mean": cv["mean"], "cv_auc_std": cv["std"],
            **{k.replace(".", ""): v for k, v in record["valid"].items()},
            "brier_before": record["calibration_valid"]["before"]["brier"],
            "brier_after": record["calibration_valid"]["after"]["brier"],
            "latency_p50_ms": p50, "latency_p95_ms": p95,
            "model_size_kb": record["model_size_kb"],
        })
        mlflow.log_artifact(str(plot))

        # 6. Register the model under its alias.
        record["registered"] = register_model(model, data.X_valid.head(3), config, threshold)

    out["final_model"].write_text(json.dumps(record, indent=2) + "\n")
    return record


# PROVIDED
def print_summary(record: dict[str, Any]) -> None:
    """The numbers the modeling report needs, side by side."""
    tree, valid = record["compare_to"], record["valid"]
    reg = record["registered"]
    print(f"Registered {reg['uri']} (version {reg['version']}), threshold "
          f"{record['threshold']:.3f}, {record['n_features']} features\n")
    def interval(m: dict[str, Any]) -> str:
        return f"[{m['valid_auc_ci_low']:.3f}, {m['valid_auc_ci_high']:.3f}]"

    rows = [
        ("valid AUC", f"{valid['valid_auc']:.3f}", f"{tree['valid_auc']:.3f}"),
        ("  95% CI", interval(valid), interval(tree)),
        ("latency p95 (ms)", f"{record['latency_p95_ms']:.1f}", f"{tree['latency_p95_ms']:.1f}"),
        ("model size (KB)", f"{record['model_size_kb']:.0f}", f"{tree['model_size_kb']:.0f}"),
        ("features", str(record["n_features"]), str(tree["n_features"])),
    ]
    print(f"{'':<24}{'final model':>16}{tree['run']:>16}")
    for label, mine, theirs in rows:
        print(f"{label:<24}{mine:>16}{theirs:>16}")
    low, high = tree["auc_diff_ci"]
    print(f"\nAUC difference (final - {tree['run']}): {tree['auc_diff']:+.3f}, paired 95% CI "
          f"[{low:+.3f}, {high:+.3f}]")
    before, after = record["calibration_valid"]["before"], record["calibration_valid"]["after"]
    print(f"Brier score {before['brier']:.4f} -> {after['brier']:.4f}; mean predicted "
          f"{before['mean_predicted']:.3f} -> {after['mean_predicted']:.3f} "
          f"(default rate {after['default_rate']:.3f})")
    print(f"Validation cost per applicant: {valid['cost_at_threshold']:.3f} at the threshold, "
          f"{valid['cost_at_0.5']:.3f} at 0.5, {valid['cost_approve_all']:.3f} approving all")


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.final``."""
    parser = argparse.ArgumentParser(description="Build, check, and register the final model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    setup_mlflow(config)
    record = run_final(config)
    print_summary(record)
    out = config["week3"]["outputs"]
    print(f"\nWrote {out['final_model']}\nWrote {out['reliability_plot']}")


if __name__ == "__main__":
    main()
