"""MLflow helpers: where runs are stored, what every run is tagged with, and the run table.

# PROVIDED: this whole module is finished. Read it and reuse it.

Every run is stored in ONE local database, mlflow.db in the repository root. Look at the
runs with ``python run.py ui`` (then open http://127.0.0.1:5050), and export them for your
submission with ``python run.py runs``, which writes reports/mlflow_runs.csv.

Each run records, so that anyone can rebuild it:
    params    the model and every setting that differs from its defaults
    metrics   CV AUC, train AUC, validation AUC with its interval, speed, size
    tags      git_commit (the code), data_version (the data and split), settings
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from src.experiment import load_week3_config
from src.utils import DEFAULT_CONFIG

RUN_COLUMNS = [
    "run_name", "parent", "start_time", "params.model", "tags.settings", "metrics.cv_auc_mean",
    "metrics.cv_auc_std", "metrics.train_auc", "metrics.valid_auc", "metrics.valid_auc_ci_low",
    "metrics.valid_auc_ci_high", "metrics.valid_pr_auc", "metrics.valid_ks",
    "metrics.fit_seconds", "metrics.latency_p95_ms", "metrics.model_size_kb",
    "tags.git_commit", "tags.data_version", "run_id",
]


# PROVIDED
def setup_mlflow(config: dict[str, Any]) -> str:
    """Point MLflow at the project's database and experiment; return the experiment id.

    Saved models go to mlruns/ in the repository root, whichever folder you run from.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # hide a log line about AI agents
    settings = config["week3"]["mlflow"]
    mlflow.set_tracking_uri(settings["tracking_uri"])
    name = settings["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = (config["root"] / "mlruns").resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    return mlflow.set_experiment(name).experiment_id


# PROVIDED
def log_sklearn_model(model: Any, **kwargs: Any) -> Any:
    """Save a fitted scikit-learn model with the open run; return MLflow's ModelInfo.

    ``kwargs`` go to ``mlflow.sklearn.log_model`` (for example ``registered_model_name``).
    The model is stored with cloudpickle: newer MLflow versions default to a format (skops)
    that refuses some ordinary scikit-learn pipelines.
    """
    # MLflow prints warnings on every save that are not about your project (pickle safety for
    # your own models, integer columns in the input example, the pip version): hide them.
    loggers = [logging.getLogger(name) for name in ("mlflow.sklearn", "mlflow.utils.environment")]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
            return mlflow.sklearn.log_model(model, name="model",
                                            serialization_format="cloudpickle", **kwargs)
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


# PROVIDED
def git_commit(root: Path) -> str:
    """The current commit (short hash), with '-dirty' if tracked files have unsaved changes.

    A dirty run used code that is not in any commit, so nobody can rebuild it exactly:
    commit before the runs you want to keep.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "no-git"
    return f"{commit}-dirty" if dirty else commit


# PROVIDED
def run_tags(config: dict[str, Any], data_version: str, settings: str = "") -> dict[str, str]:
    """The tags every run carries."""
    return {
        "git_commit": git_commit(config["root"]),
        "data_version": data_version,
        "settings": settings,
        "week": "3",
    }


# PROVIDED
def load_runs(config: dict[str, Any]) -> pd.DataFrame:
    """Every run in the experiment, newest first, with ``run_name`` and ``parent`` columns.

    ``parent`` is the run_name of the parent run (for example "tree_tuning") or empty for a
    top-level run.
    """
    setup_mlflow(config)
    runs = mlflow.search_runs(experiment_names=[config["week3"]["mlflow"]["experiment"]],
                              order_by=["attributes.start_time DESC"])
    if runs.empty:
        return runs.assign(run_name=pd.Series(dtype=str), parent=pd.Series(dtype=str))
    runs = runs[runs["status"] == "FINISHED"].copy()
    runs["run_name"] = runs["tags.mlflow.runName"]
    names = runs.set_index("run_id")["run_name"]
    parent_ids = runs.get("tags.mlflow.parentRunId", pd.Series(index=runs.index, dtype=str))
    runs["parent"] = parent_ids.map(names).fillna("")
    return runs.reset_index(drop=True)


# PROVIDED
def latest_top_level(runs: pd.DataFrame) -> pd.DataFrame:
    """The newest top-level run of each name (a rerun replaces the older run of that name)."""
    if runs.empty:
        return runs
    top = runs[runs["parent"] == ""]
    return top.drop_duplicates("run_name", keep="first").reset_index(drop=True)


# PROVIDED
def export_runs(config: dict[str, Any]) -> Path:
    """Write every finished run to reports/mlflow_runs.csv (the file you submit)."""
    runs = load_runs(config)
    columns = [c for c in RUN_COLUMNS if c in runs.columns]
    extra = sorted(c for c in runs.columns
                   if c.startswith(("params.", "metrics.")) and c not in columns)
    path = config["week3"]["runs_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    runs[columns + extra].to_csv(path, index=False, float_format="%.4f")
    return path


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.tracking`` exports the run table."""
    parser = argparse.ArgumentParser(description="Export every MLflow run to a CSV file.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    path = export_runs(config)
    runs = pd.read_csv(path)
    print(f"Wrote {path}: {len(runs)} runs "
          f"({int((runs['parent'].isna() | (runs['parent'] == '')).sum())} top-level)")


if __name__ == "__main__":
    main()
