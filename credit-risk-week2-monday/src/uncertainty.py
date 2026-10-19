"""How sure can we be about an AUC measured on about 80 defaults?

# PROVIDED: this whole module is finished. Read it and reuse it.

The validation set has 1,000 applicants but only about 80 defaults, so an AUC moves by a few
hundredths just from which applicants happened to land in it. The bootstrap estimates that
wobble: draw 1,000 applicants with replacement, recompute the AUC, repeat many times, and
take the middle 95 percent of the results.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

ArrayLike = np.ndarray | list


# PROVIDED
def _resamples(y_true: np.ndarray, n_boot: int, seed: int):
    """Yield index arrays drawn with replacement, skipping draws that miss a class."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if 0 < y_true[idx].sum() < n:
            yield idx


# PROVIDED
def bootstrap_auc_ci(
    y_true: ArrayLike, y_score: ArrayLike, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """95% bootstrap interval for ROC-AUC, as (low, high)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    aucs = [roc_auc_score(y_true[i], y_score[i]) for i in _resamples(y_true, n_boot, seed)]
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


# PROVIDED
def bootstrap_auc_diff_ci(
    y_true: ArrayLike,
    score_new: ArrayLike,
    score_old: ArrayLike,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap interval for AUC(new) - AUC(old) on the same applicants, as (low, high).

    Both models are scored on the same resampled applicants each time ("paired"), which
    removes the luck of the draw that both share. If the interval is entirely above 0, the
    new features help by more than chance; if it contains 0, we cannot tell.
    """
    y_true = np.asarray(y_true)
    score_new, score_old = np.asarray(score_new), np.asarray(score_old)
    diffs = [
        roc_auc_score(y_true[i], score_new[i]) - roc_auc_score(y_true[i], score_old[i])
        for i in _resamples(y_true, n_boot, seed)
    ]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)
