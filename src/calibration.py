"""Calibration: make a score of 0.08 mean "8 in 100 such applicants default".

A model can rank applicants well (high AUC) and still give probabilities that are too high
or too low. Boosting trained to fit the training rows closely tends to push scores toward
0 and 1. The cost-based threshold assumes the scores ARE probabilities (with costs 10 and 1,
the textbook threshold is 1 / (1 + 10) = 0.091), so calibration matters here.

Calibration fits a small monotone map from score to probability: isotonic (a step function,
flexible, needs more data) or sigmoid (Platt scaling, a smooth S-curve, needs less). It never
changes the ORDER of the scores much, so the AUC barely moves; it changes what they mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: F401
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold  # noqa: F401  (for you to use)
from sklearn.pipeline import Pipeline

METHODS = ("isotonic", "sigmoid")


# TODO(student)
def calibrate(model: Pipeline, method: str, n_folds: int, seed: int) -> CalibratedClassifierCV:
    """Wrap an UNFITTED model so that fitting it also calibrates its probabilities.

    Return ``CalibratedClassifierCV`` around ``model`` with:

    - ``method``: "isotonic" or "sigmoid"; anything else raises a ValueError naming METHODS.
    - ``cv``: a ``StratifiedKFold`` with ``n_folds`` folds, ``shuffle=True`` and
      ``random_state=seed``, so the calibration map is learned from out-of-fold scores
      (scores from rows the model did not train on).
    - ``ensemble=False``: after calibrating, keep ONE model refitted on all the rows, not one
      per fold. One model is 5 times smaller and faster to score than five.

    Args:
        model: An unfitted Pipeline from ``make_model``.
        method: "isotonic" or "sigmoid".
        n_folds: Folds used to learn the calibration map.
        seed: Seed for the folds.

    Returns:
        An unfitted CalibratedClassifierCV. Call ``.fit(X, y)`` on it like any model.

    Hints:
        - ``CalibratedClassifierCV(model, method=..., cv=..., ensemble=False)``.
    """
    raise NotImplementedError("TODO(student): wrap the model in CalibratedClassifierCV")


# PROVIDED
def reliability_table(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed default rate against mean predicted probability, in equal-size score bins.

    Equal-size (quantile) bins, because almost every score is small: equal-width bins would
    put nearly everyone in the first bin.
    """
    observed, predicted = calibration_curve(y_true, scores, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


# PROVIDED
def calibration_summary(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Brier score (lower is better) and mean predicted probability vs the real default rate."""
    return {
        "brier": float(brier_score_loss(y_true, scores)),
        "mean_predicted": float(np.mean(scores)),
        "default_rate": float(np.mean(y_true)),
    }
