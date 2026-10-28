"""Evaluation plots for the modeling report.

# PROVIDED: this whole module is finished. Each function saves one PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


# PROVIDED
def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# PROVIDED
def plot_cost_curve(curve: pd.DataFrame, threshold: float, path: Path, title: str) -> Path:
    """Expected cost per applicant against the threshold, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve["threshold"], curve["expected_cost"], color="#1f5f99")
    best = curve.loc[(curve["threshold"] - threshold).abs().idxmin()]
    ax.axvline(threshold, color="#c0392b", linestyle="--",
               label=f"chosen {threshold:.3f}: cost {best['expected_cost']:.3f}")
    at_half = curve.loc[(curve["threshold"] - 0.5).abs().idxmin()]
    ax.scatter([at_half["threshold"]], [at_half["expected_cost"]], color="#7f8c8d", zorder=3,
               label=f"0.5: cost {at_half['expected_cost']:.3f}")
    ax.set_xlabel("threshold (reject when score >= threshold)")
    ax.set_ylabel("expected cost per applicant")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


# PROVIDED
def plot_reliability(tables: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    """Reliability curves: mean predicted probability against the observed default rate.

    ``tables`` maps a label (for example "before") to a frame with ``mean_predicted`` and
    ``observed_rate`` columns, one row per bin. A calibrated model lies on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    top = max(max(t["mean_predicted"].max(), t["observed_rate"].max()) for t in tables.values())
    top = min(1.0, top * 1.1)
    ax.plot([0, top], [0, top], color="#7f8c8d", linestyle=":", label="perfectly calibrated")
    for (label, table), color in zip(tables.items(), ["#c0392b", "#1f5f99", "#27ae60"],
                                     strict=False):
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color=color,
                label=label)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("mean predicted probability (per bin)")
    ax.set_ylabel("observed default rate (per bin)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)

