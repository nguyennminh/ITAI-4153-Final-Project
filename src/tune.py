"""Tune the boosting model with Optuna, logging every trial to MLflow.

Run it with ``python run.py tune`` (or ``python -m src.tune``). Each run adds
``trials_per_run`` trials (config/final.yaml) to ONE saved study in optuna.db, so you can
run 15 trials in class and 15 more at home: the second run continues where the first stopped.

Grid search (Monday's tree) tries every combination. Optuna's default sampler (TPE) instead
looks at the trials so far and proposes settings near the ones that scored well, which finds
good settings with far fewer trials when there are 5 or 6 settings to tune at once.

In MLflow, each ``tune`` is one parent run (``optuna_<study>``) with one child run per trial.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import optuna

from src.evaluation import cross_validate_auc  # noqa: F401  (for you to use)
from src.experiment import TrainingData, load_training_data, load_week3_config
from src.models import describe_spec, make_model  # noqa: F401
from src.tracking import run_tags, setup_mlflow
from src.utils import DEFAULT_CONFIG

BEST_TRIAL = "best_trial"


# TODO(student)
def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Ask the trial for one set of HistGradientBoostingClassifier settings.

    Suggest at least these, with these ranges (they cover what helped on Monday):

    ==================  ===============================  ====================
    setting             range                            Optuna call
    ==================  ===============================  ====================
    learning_rate       0.01 to 0.3, on a log scale      ``suggest_float(..., log=True)``
    max_iter            100 to 500, in steps of 50       ``suggest_int(..., step=50)``
    max_leaf_nodes      4 to 32                          ``suggest_int``
    min_samples_leaf    20 to 200                        ``suggest_int``
    l2_regularization   0.001 to 10, on a log scale      ``suggest_float(..., log=True)``
    max_features        0.3 to 1.0                       ``suggest_float``
    ==================  ===============================  ====================

    Args:
        trial: The Optuna trial.

    Returns:
        The settings as a dict, keyed by the parameter names above (these go straight into
        the model as ``spec["params"]``).

    Hints:
        - ``trial.suggest_float("learning_rate", 0.01, 0.3, log=True)``. The first argument
          is the name Optuna records, so use the parameter's own name.
        - Why a log scale? 0.01 to 0.03 matters as much as 0.1 to 0.3. A plain range would
          spend most trials above 0.1.
        - Using another library (LightGBM, XGBoost)? Suggest its own parameter names instead.
    """
    raise NotImplementedError("TODO(student): suggest the boosting settings")


# TODO(student)
def objective(trial: optuna.Trial, data: TrainingData, config: dict[str, Any]) -> float:
    """Score one trial: its settings' mean CV AUC on the train split, logged as a child run.

    1. ``params = suggest_params(trial)``, and a spec with the model from final.yaml:
       ``{"model": config["week3"]["final"]["model"]["model"], "params": params}``.
    2. Build it with ``make_model`` and score it with ``cross_validate_auc`` (folds and seed
       from the config, as on Monday).
    3. Log a CHILD run (``nested=True``) named ``trial_000``, ``trial_001``, ... (from
       ``trial.number``) with tags ``run_tags(config, data.data_version,
       describe_spec(spec))`` plus ``{"optuna_study": <study name>}``, the params, and the
       metrics ``cv_auc_mean`` and ``cv_auc_std``.
    4. Return the mean CV AUC (Optuna maximizes it).

    Never use the validation split here: 30 trials picked by validation AUC fit the
    validation set, just like Monday's tree grid would have.

    Args:
        trial: The Optuna trial.
        data: The fixed train and validation rows.
        config: The loaded Week 3 config.

    Returns:
        The trial's mean CV AUC.

    Hints:
        - Study name: ``config["week3"]["final"]["optuna"]["study"]``.
        - Tags: ``{**run_tags(...), "optuna_study": study_name}``.
        - ``mlflow.log_params(params)`` and ``mlflow.log_metrics({...})``.
    """
    raise NotImplementedError("TODO(student): CV-score the trial's settings and log them")


# PROVIDED
def model_spec(config: dict[str, Any]) -> dict[str, Any]:
    """The model spec from final.yaml, with ``params: best_trial`` replaced by the best trial."""
    spec = dict(config["week3"]["final"]["model"])
    if spec.get("params") == BEST_TRIAL:
        path = config["week3"]["outputs"]["optuna_best"]
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found: run `python run.py tune` first, "
                                    "or write the settings in config/final.yaml")
        spec["params"] = json.loads(path.read_text())["params"]
    return spec


# PROVIDED
def run_study(config: dict[str, Any], n_trials: int) -> optuna.Study:
    """Add ``n_trials`` trials to the saved study inside one MLflow parent run."""
    settings = config["week3"]["final"]["optuna"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=settings["study"], storage=settings["storage"], load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"]),
    )
    data = load_training_data(config)
    done = len(study.trials)

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  trial {trial.number:3d}  CV AUC {trial.value:.4f}   best so far "
              f"{study.best_value:.4f} (trial {study.best_trial.number})", flush=True)

    with mlflow.start_run(run_name=f"optuna_{settings['study']}",
                          tags={**run_tags(config, data.data_version),
                                "optuna_study": settings["study"]}):
        study.optimize(lambda trial: objective(trial, data, config), n_trials=n_trials,
                       callbacks=[report])
        best = study.best_trial
        mlflow.log_params({f"best_{key}": value for key, value in best.params.items()})
        mlflow.log_metrics({"best_cv_auc_mean": best.value, "n_trials": len(study.trials)})

    path = config["week3"]["outputs"]["optuna_best"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study": settings["study"], "trial": best.number,
                                "cv_auc_mean": best.value, "n_trials": len(study.trials),
                                "params": best.params}, indent=2) + "\n")
    print(f"\n{len(study.trials) - done} new trials, {len(study.trials)} in the study "
          f"(target {settings['target_trials']}). Best: trial {best.number}, "
          f"CV AUC {best.value:.4f}\n{json.dumps(best.params, indent=2)}\nWrote {path}")
    return study


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.tune [--trials N]``."""
    parser = argparse.ArgumentParser(description="Add Optuna trials to the tuning study.")
    parser.add_argument("--trials", type=int, default=None,
                        help="trials to add (default: trials_per_run in final.yaml)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    config = load_week3_config(args.config)
    setup_mlflow(config)
    run_study(config, args.trials or config["week3"]["final"]["optuna"]["trials_per_run"])


if __name__ == "__main__":
    main()
