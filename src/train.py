"""Train the runs named in config/models.yaml and log each one to MLflow.

Run it with ``python run.py train NAME [NAME ...]`` (or ``python -m src.train ...``):

    python run.py train                  list the runs you can train
    python run.py train hgb_default      train one run
    python run.py train tree_tuned       grid-search the decision tree (your tune_tree)
    python run.py train --all            every run in models.yaml, then tree_tuned

Every run uses the same rows (the train split of your feature table), the same folds, and
the same seed, so the only thing that differs between two runs is the model and its settings.
"""

from __future__ import annotations

import argparse
import itertools  # noqa: F401  (for you to use)
import json
import time
from typing import Any

import mlflow
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    cross_validate_auc,
    measure_latency_ms,
    model_size_kb,
    validation_metrics,
)
from src.experiment import TrainingData, load_training_data, load_week3_config
from src.models import describe_spec, fit_model, make_model
from src.tracking import log_sklearn_model, run_tags, setup_mlflow
from src.utils import DEFAULT_CONFIG, set_seed

TUNED_TREE = "tree_tuned"


# PROVIDED: the worked example. Read it before you write tune_tree.
def train_and_log(
    name: str, spec: dict[str, Any], data: TrainingData, config: dict[str, Any],
    nested: bool = False, log_model: bool = True,
) -> dict[str, Any]:
    """Cross-validate, fit, and score one model spec, and log all of it as one MLflow run.

    Args:
        name: The run name shown in the MLflow UI.
        spec: One run from models.yaml: ``model``, optional ``params`` and ``undersample``.
        data: The fixed train and validation rows.
        config: The loaded Week 3 config.
        nested: True to log this run inside the run that is currently open (a child run).
        log_model: True to save the fitted model with the run (skipped for grid children).

    Returns:
        ``{"run_id": ..., "name": ..., "spec": ..., **every logged metric}``.
    """
    set_seed(config["seed"])
    week3, seed = config["week3"], config["seed"]
    ratio = spec.get("undersample")
    model = make_model(spec, data.numeric, data.categorical, config)
    settings = describe_spec(spec)

    # 1. Open a run. The tags say which code and which data produced it, and the exact spec.
    tags = {**run_tags(config, data.data_version, settings), "spec": json.dumps(spec),
            "step": "train"}
    with mlflow.start_run(run_name=name, nested=nested, tags=tags) as run:
        # 2. Params: what was trained.
        mlflow.log_params({"model": spec["model"], "undersample": ratio or "none",
                           "n_features": len(data.features), **(spec.get("params") or {})})

        # 3. Cross-validation on the train split: the number to compare runs by.
        cv = cross_validate_auc(model, data.X_train, data.y_train, week3["cv_folds"], seed,
                                ratio)

        # 4. Fit once on the whole train split, then score train and validation.
        start = time.perf_counter()
        fitted = fit_model(clone(model), data.X_train, data.y_train, ratio, seed)
        fit_seconds = time.perf_counter() - start
        train_scores = fitted.predict_proba(data.X_train)[:, 1]
        valid_scores = fitted.predict_proba(data.X_valid)[:, 1]
        p50, p95 = measure_latency_ms(fitted, data.X_valid)

        # 5. Metrics: how well, how fast, how big.
        metrics = {
            "cv_auc_mean": cv["mean"],
            "cv_auc_std": cv["std"],
            "train_auc": float(roc_auc_score(data.y_train, train_scores)),
            **validation_metrics(data.y_valid, valid_scores, week3["n_bootstrap"], seed),
            "fit_seconds": fit_seconds,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "model_size_kb": model_size_kb(fitted),
        }
        mlflow.log_metrics(metrics)

        # 6. The fitted model itself, so it can be loaded or registered later.
        if log_model:
            log_sklearn_model(fitted)
    return {"run_id": run.info.run_id, "name": name, "spec": spec, **metrics}


# TODO(student)
def tune_tree(data: TrainingData, config: dict[str, Any]) -> dict[str, Any]:
    """Grid-search the decision tree's max_depth and min_samples_leaf with stratified CV.

    1. Open ONE parent run named "tree_tuning" (tags from ``run_tags``).
    2. Inside it, for every combination in ``config["week3"]["tree_grid"]`` (every
       max_depth with every min_samples_leaf), call ``train_and_log`` as a CHILD run
       (``nested=True, log_model=False``), named for example ``tree_d5_l50``.
    3. Pick the combination with the highest ``cv_auc_mean``. Choose by CV, never by
       validation AUC: picking the best of 30 on the validation set fits the validation set.
       On a tie, keep the combination that came first.
    4. Log the winner on the parent run: params ``best_max_depth`` and
       ``best_min_samples_leaf``, metric ``best_cv_auc_mean``.
    5. After the parent run is closed, train the winner once more as a normal top-level run
       named ``TUNED_TREE`` ("tree_tuned"), with ``log_model=True``.

    Args:
        data: The fixed train and validation rows.
        config: The loaded Week 3 config.

    Returns:
        The ``train_and_log`` result of the "tree_tuned" run.

    Hints:
        - ``grid = config["week3"]["tree_grid"]`` and
          ``itertools.product(grid["max_depth"], grid["min_samples_leaf"])``.
        - A spec is a dict: ``{"model": "decision_tree", "params": {"max_depth": d,
          "min_samples_leaf": leaf}}``.
        - Look at ``train_and_log`` for how to open a run with tags.
    """
    raise NotImplementedError("TODO(student): grid-search the tree and train the winner")


# PROVIDED
def print_result(result: dict[str, Any]) -> None:
    """One line per run: what to compare (CV), and the overfitting signal (train vs CV)."""
    print(f"{result['name']:<22} CV AUC {result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f}"
          f"   train {result['train_auc']:.3f}   valid {result['valid_auc']:.3f} "
          f"[{result['valid_auc_ci_low']:.3f}, {result['valid_auc_ci_high']:.3f}]"
          f"   {result['fit_seconds']:.1f}s fit")


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.train NAME [NAME ...] | --all``."""
    parser = argparse.ArgumentParser(description="Train runs from config/models.yaml.")
    parser.add_argument("names", nargs="*", help="run names from models.yaml, or tree_tuned")
    parser.add_argument("--all", action="store_true", help="every run, then tree_tuned")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    config = load_week3_config(args.config)
    runs = config["week3"]["runs"]
    names = [*runs, TUNED_TREE] if args.all else args.names
    if not names:
        print("Runs you can train (python run.py train NAME):")
        for name, spec in runs.items():
            print(f"  {name:<22}{spec['model']}  {describe_spec(spec) or '(defaults)'}")
        print(f"  {TUNED_TREE:<22}decision_tree grid search (tune_tree)")
        return
    unknown = [n for n in names if n not in runs and n != TUNED_TREE]
    if unknown:
        raise SystemExit(f"unknown run(s) {unknown}; see python run.py train")

    setup_mlflow(config)
    data = load_training_data(config)
    print(f"{len(data.y_train):,} train rows ({int(data.y_train.sum())} defaults), "
          f"{len(data.features)} features, data_version {data.data_version}\n")
    for name in names:
        result = tune_tree(data, config) if name == TUNED_TREE else \
            train_and_log(name, runs[name], data, config)
        print_result(result)
    print("\nSee every run: python run.py ui   (then open http://127.0.0.1:5050)")


if __name__ == "__main__":
    main()
