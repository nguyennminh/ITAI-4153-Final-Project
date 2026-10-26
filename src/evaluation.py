"""Measuring a model: cross-validated AUC, validation metrics with intervals, speed, size.

Why cross-validation? The validation set has only 81 defaults, so its AUC moves by about
0.03 from the luck of which applicants landed in it. That is as big as the differences
between the models you compare this week. Stratified 5-fold CV on the 4,000 training
applicants scores every one of them once, on a model that never saw them, and the spread
of the 5 fold AUCs shows how noisy the number is.
"""

from __future__ import annotations

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone  # noqa: F401  (for you to use)
from sklearn.metrics import roc_auc_score  # noqa: F401
from sklearn.model_selection import StratifiedKFold  # noqa: F401
from sklearn.pipeline import Pipeline

from src.metrics import evaluate
from src.models import fit_model  # noqa: F401
from src.uncertainty import bootstrap_auc_ci


def cross_validate_auc(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int,
    undersample_ratio: float | None = None,
) -> dict[str, Any]:
    """Stratified k-fold ROC-AUC, plus an out-of-fold score for every row.

    For each of ``n_folds`` stratified folds (shuffled with ``random_state=seed``): fit a
    fresh copy of ``model`` on the other folds with ``fit_model`` (which undersamples those
    training rows if ``undersample_ratio`` is set), score the held-out fold with
    ``predict_proba``, and record that fold's ROC-AUC.

    Args:
        model: An unfitted Pipeline (from ``make_model``). It is not changed.
        X: Feature rows (the TRAIN split only).
        y: Labels aligned with ``X``.
        n_folds: Number of folds, for example 5.
        seed: Seed for the fold assignment and for undersampling.
        undersample_ratio: Passed to ``fit_model``. The held-out fold is never undersampled.

    Returns:
        ``{"fold_aucs": [...], "mean": float, "std": float, "oof": np.ndarray}``, where
        ``oof[i]`` is row i's score from the one model that did not train on it. ``std`` is
        ``np.std`` of the fold AUCs.

    Hints:
        - ``StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y)``
          yields positions; use ``X.iloc[...]`` and ``y.iloc[...]``.
        - ``clone(model)`` gives an unfitted copy with the same settings. Fitting ``model``
          itself in every fold would reuse one object for all folds.
        - Start with ``oof = np.full(len(y), np.nan)`` and fill one fold at a time.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    for train_idx, valid_idx in folds.split(X, y):
        fitted = fit_model(
            clone(model), X.iloc[train_idx], y.iloc[train_idx],
            undersample_ratio=undersample_ratio, seed=seed,
        )
        scores = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = scores
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], scores)))
    return {
        "fold_aucs": fold_aucs,
        "mean": float(np.mean(fold_aucs)),
        "std": float(np.std(fold_aucs)),
        "oof": oof,
    }


# PROVIDED
def validation_metrics(
    y_valid: pd.Series, scores: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """ROC-AUC with its 95% bootstrap interval, PR-AUC, and KS on the validation split.

    Uses YOUR Week 1 ``evaluate`` and the Week 2 ``bootstrap_auc_ci``.
    """
    y = np.asarray(y_valid)
    metrics = evaluate(y, scores)
    low, high = bootstrap_auc_ci(y, scores, n_boot, seed)
    return {
        "valid_auc": float(metrics["roc_auc"]),
        "valid_auc_ci_low": low,
        "valid_auc_ci_high": high,
        "valid_pr_auc": float(metrics["pr_auc"]),
        "valid_ks": float(metrics["ks"]),
    }


# PROVIDED
def measure_latency_ms(
    model: Pipeline, X: pd.DataFrame, n_calls: int = 100, seed: int = 0
) -> tuple[float, float]:
    """Time ``predict_proba`` on ONE applicant at a time, as a live service would call it.

    Returns:
        ``(p50, p95)`` in milliseconds: the typical call, and the slowest 1 in 20.
    """
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(X), size=n_calls)
    for i in rows[:5]:  # warm-up calls are not timed
        model.predict_proba(X.iloc[[i]])
    times = []
    for i in rows:
        start = time.perf_counter()
        model.predict_proba(X.iloc[[i]])
        times.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    return float(p50), float(p95)


# PROVIDED
def model_size_kb(model: Pipeline) -> float:
    """Size of the fitted model when saved (pickled), in kilobytes."""
    return len(pickle.dumps(model)) / 1024
