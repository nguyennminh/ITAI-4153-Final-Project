"""Turn scores into decisions: the threshold that costs the lender least.

Run it with ``python run.py threshold`` (or ``python -m src.cost``).

A score is not a decision. The service must answer "approve" or "reject", so it needs a
threshold: reject when score >= threshold. 0.5 is the default only because it is the middle.
With the Week 1 costs (approving a defaulter costs 10, rejecting a good customer costs 1),
missing a defaulter is 10 times worse, so the best threshold is far below 0.5.

The threshold is chosen on OUT-OF-FOLD scores of the TRAIN split (each applicant scored by a
model that never saw them), never on the validation split: choosing it there would fit the
validation set, and the validation cost would then look better than it really is.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation import cross_validate_auc
from src.experiment import load_training_data, load_week3_config
from src.models import describe_spec, fit_model, make_model
from src.plots import plot_cost_curve
from src.tracking import run_tags, setup_mlflow
from src.tune import model_spec
from src.utils import DEFAULT_CONFIG

ArrayLike = np.ndarray | pd.Series | list


# TODO(student)
def expected_cost(
    y_true: ArrayLike, scores: ArrayLike, threshold: float, costs: dict[str, float]
) -> float:
    """Average cost per applicant when we reject every score >= ``threshold``.

    - A defaulter (y = 1) we approve (score < threshold) costs ``costs["approve_defaulter"]``.
    - A good customer (y = 0) we reject (score >= threshold) costs ``costs["reject_good"]``.
    - Correct decisions cost 0.

    Example: y = [1, 0, 0, 1, 0], scores = [0.9, 0.2, 0.6, 0.3, 0.1], threshold 0.5, costs
    10 and 1. We reject applicants 0 and 2. Applicant 3 is an approved defaulter (10),
    applicant 2 a rejected good customer (1): total 11, so 11 / 5 = 2.2 per applicant.

    Args:
        y_true: True labels (1 = default).
        scores: Predicted default probabilities, aligned with ``y_true``.
        threshold: Reject when score >= threshold.
        costs: ``{"approve_defaulter": ..., "reject_good": ...}`` (final.yaml).

    Returns:
        The total cost divided by the number of applicants.

    Hints:
        - ``y = np.asarray(y_true)``, ``reject = np.asarray(scores) >= threshold``.
        - Approved defaulters: ``(y == 1) & ~reject``. Rejected good: ``(y == 0) & reject``.
    """
    raise NotImplementedError("TODO(student): the expected cost per applicant")


# TODO(student)
def cost_curve(
    y_true: ArrayLike, scores: ArrayLike, thresholds: ArrayLike, costs: dict[str, float]
) -> pd.DataFrame:
    """The expected cost at every threshold in ``thresholds``, one row each, in that order.

    Returns:
        A DataFrame with columns ``threshold``, ``expected_cost`` (from ``expected_cost``),
        ``reject_rate`` (share of applicants rejected), ``approved_defaulters`` and
        ``rejected_good`` (counts).
    """
    raise NotImplementedError("TODO(student): one row per threshold")


# TODO(student)
def choose_threshold(curve: pd.DataFrame) -> float:
    """The threshold with the lowest expected cost in ``curve``.

    When several thresholds tie on the lowest cost, return the LARGEST of them: it rejects
    the fewest applicants for the same cost.

    Args:
        curve: The output of ``cost_curve``.

    Returns:
        The chosen threshold, as a plain float.
    """
    raise NotImplementedError("TODO(student): the cheapest threshold")


# PROVIDED
def threshold_grid(config: dict[str, Any]) -> np.ndarray:
    """The thresholds to try, from final.yaml."""
    grid = config["week3"]["final"]["threshold_grid"]
    return np.round(np.arange(grid["start"], grid["stop"] + grid["step"] / 2, grid["step"]), 4)


# PROVIDED
def decision_summary(
    y_true: ArrayLike, scores: ArrayLike, threshold: float, costs: dict[str, float]
) -> dict[str, float]:
    """Cost at the chosen threshold, at 0.5, and for the two do-nothing policies."""
    return {
        "cost_at_threshold": expected_cost(y_true, scores, threshold, costs),
        "cost_at_0.5": expected_cost(y_true, scores, 0.5, costs),
        "cost_approve_all": expected_cost(y_true, scores, np.inf, costs),
        "cost_reject_all": expected_cost(y_true, scores, -np.inf, costs),
        "reject_rate": float(np.mean(np.asarray(scores) >= threshold)),
    }


# PROVIDED
def run_threshold(config: dict[str, Any]) -> dict[str, Any]:
    """Choose the threshold from out-of-fold train scores; report its cost on validation."""
    final = config["week3"]["final"]
    seed, costs = config["seed"], final["costs"]
    spec = model_spec(config)
    data = load_training_data(config)
    model = make_model(spec, data.numeric, data.categorical, config)

    with mlflow.start_run(run_name="threshold",
                          tags=run_tags(config, data.data_version, describe_spec(spec))):
        oof = cross_validate_auc(model, data.X_train, data.y_train,
                                 config["week3"]["cv_folds"], seed, spec.get("undersample"))["oof"]
        curve = cost_curve(data.y_train, oof, threshold_grid(config), costs)
        threshold = choose_threshold(curve)

        fitted = fit_model(clone(model), data.X_train, data.y_train, spec.get("undersample"), seed)
        valid_scores = fitted.predict_proba(data.X_valid)[:, 1]
        record = {
            "threshold": threshold, "chosen_on": "train out-of-fold scores", "costs": costs,
            "model": spec, "calibrated": False,
            "train_oof": decision_summary(data.y_train, oof, threshold, costs),
            "valid": decision_summary(data.y_valid, valid_scores, threshold, costs),
        }
        out = config["week3"]["outputs"]
        plot = plot_cost_curve(curve, threshold, out["cost_plot"],
                               "Expected cost by threshold (train, out-of-fold)")
        out["threshold"].parent.mkdir(parents=True, exist_ok=True)
        out["threshold"].write_text(json.dumps(record, indent=2) + "\n")
        mlflow.log_params({"threshold": threshold, **{f"cost_{k}": v for k, v in costs.items()}})
        mlflow.log_metrics({f"valid_{k.replace('.', '')}": v for k, v in record["valid"].items()})
        mlflow.log_artifact(str(plot))
        mlflow.log_artifact(str(out["threshold"]))
    return record


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.cost``."""
    parser = argparse.ArgumentParser(description="Choose the decision threshold from costs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    setup_mlflow(config)
    record = run_threshold(config)
    print(f"Chosen threshold: {record['threshold']:.3f} (reject when score >= it), chosen on "
          f"{record['chosen_on']}\n")
    print(f"{'cost per applicant':<26}{'train (oof)':>12}{'validation':>12}")
    for key in ("cost_at_threshold", "cost_at_0.5", "cost_approve_all", "cost_reject_all",
                "reject_rate"):
        print(f"{key:<26}{record['train_oof'][key]:>12.3f}{record['valid'][key]:>12.3f}")
    out = config["week3"]["outputs"]
    print(f"\nWrote {out['threshold']}\nWrote {out['cost_plot']}")


if __name__ == "__main__":
    main()
