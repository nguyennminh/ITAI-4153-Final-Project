"""Evaluation metrics for a default-probability model.

Accuracy is misleading when only about 8 percent of applicants default, so we use
ranking metrics that look at the scores themselves.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ArrayLike = np.ndarray | list


# PROVIDED
def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve.

    It is the chance that a random defaulter scores above a random non-defaulter.
    """
    return float(roc_auc_score(y_true, y_score))


# PROVIDED
def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision). Its floor is the default rate."""
    return float(average_precision_score(y_true, y_score))


# TODO(student)
def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Kolmogorov-Smirnov (KS) statistic between the scores of the two classes.

    Args:
        y_true: 0/1 labels, where 1 means the applicant defaulted.
        y_score: Model scores (higher means riskier), same length as ``y_true``.

    Returns:
        A float between 0 and 1. It is 1.0 when every defaulter scores above every
        non-defaulter, and near 0 when the scores do not separate the classes at all.

    Hints:
        - KS is the largest vertical gap between the cumulative distribution of scores
          for defaulters and the one for non-defaulters, over all possible thresholds.
        - Sketch both cumulative curves on paper for a 4-row example before coding.
        - A function you already know from ROC curves can give you both cumulative rates
          at every threshold.
        - Return a plain Python float.
    """
    # At each threshold, tpr is the share of defaulters scoring above it and fpr the share of
    # non-defaulters; their gap equals the gap between the two cumulative score curves.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


# TODO(student)
def evaluate(y_true: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
    """Compute every validation metric we report.

    Args:
        y_true: 0/1 labels.
        y_score: Model scores, same length as ``y_true``.

    Returns:
        ``{"roc_auc": ..., "pr_auc": ..., "ks": ...}`` with plain Python floats.

    Hints:
        - The three metric functions in this module already exist (one of them is yours).
        - These exact key names are read by other code and by the check script.
    """
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
    }
