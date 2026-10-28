"""Week 3 Wednesday tests. Tests of TODO(student) functions fail until you write them.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.

Every expected number below can be worked out by hand. The data is made up
(tests/week3_fixtures.py), and the project has its own MLflow and Optuna databases.
"""

from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from src.calibration import calibrate
from src.cost import choose_threshold, cost_curve, expected_cost
from src.experiment import load_training_data
from src.final import register_model
from src.models import make_model
from src.tracking import load_runs, setup_mlflow
from src.tune import run_study, suggest_params
from tests.week3_fixtures import make_week3_project

COSTS = {"approve_defaulter": 10, "reject_good": 1}
Y = [1, 0, 0, 1, 0]
SCORES = [0.9, 0.2, 0.6, 0.3, 0.1]
SMALL_HGB = {"model": "hist_gradient_boosting", "params": {"max_iter": 30}}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> dict:
    config = make_week3_project(tmp_path_factory.mktemp("week3wed"))
    setup_mlflow(config)
    return config


@pytest.fixture(scope="module")
def data(project: dict):
    return load_training_data(project)


def test_expected_cost_worked_example() -> None:
    assert expected_cost(Y, SCORES, 0.5, COSTS) == pytest.approx(11 / 5)


def test_expected_cost_extremes() -> None:
    assert expected_cost(Y, SCORES, 0.0, COSTS) == pytest.approx(3 / 5)   # reject everyone
    assert expected_cost(Y, SCORES, 1.01, COSTS) == pytest.approx(20 / 5)  # approve everyone


def test_score_equal_to_threshold_is_rejected() -> None:
    assert expected_cost([0], [0.5], 0.5, COSTS) == pytest.approx(1.0)
    assert expected_cost([1], [0.5], 0.5, COSTS) == pytest.approx(0.0)


def test_expected_cost_accepts_pandas_and_numpy() -> None:
    value = expected_cost(pd.Series(Y), np.array(SCORES), 0.5, COSTS)
    assert isinstance(value, float) and value == pytest.approx(2.2)


# ---------------------------------------------------------------- cost_curve

def test_cost_curve_one_row_per_threshold() -> None:
    curve = cost_curve(Y, SCORES, [0.0, 0.5, 1.01], COSTS)
    assert list(curve.columns[:5]) == ["threshold", "expected_cost", "reject_rate",
                                       "approved_defaulters", "rejected_good"]
    assert curve["threshold"].tolist() == [0.0, 0.5, 1.01]
    assert curve["expected_cost"].tolist() == pytest.approx([0.6, 2.2, 4.0])
    assert curve["reject_rate"].tolist() == pytest.approx([1.0, 0.4, 0.0])
    assert curve["approved_defaulters"].tolist() == [0, 1, 2]
    assert curve["rejected_good"].tolist() == [3, 1, 0]


# ---------------------------------------------------------------- choose_threshold

def test_choose_threshold_takes_the_cheapest_and_the_largest_on_a_tie() -> None:
    curve = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4],
                          "expected_cost": [0.5, 0.3, 0.3, 0.6]})
    chosen = choose_threshold(curve)
    assert isinstance(chosen, float) and chosen == pytest.approx(0.3)


def test_calibrated_scores_give_the_textbook_threshold() -> None:
    """With true probabilities and costs 10 and 1, reject when p >= 1/11 (about 0.091)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, size=200_000)
    y = (rng.random(p.size) < p).astype(int)
    thresholds = np.round(np.arange(0.02, 0.30, 0.005), 3)
    chosen = choose_threshold(cost_curve(y, p, thresholds, COSTS))
    assert abs(chosen - 1 / 11) < 0.02


# ---------------------------------------------------------------- suggest_params and objective

def test_suggest_params_defines_a_search_space() -> None:
    trial = optuna.create_study(direction="maximize").ask()
    params = suggest_params(trial)
    assert set(params) == set(trial.distributions), "suggest every value through the trial"
    assert len(params) >= 4
    assert trial.distributions["learning_rate"].log, "search learning_rate on a log scale"
    HistGradientBoostingClassifier(**params)  # every name is a real setting


def test_tuning_logs_every_trial_and_continues(project: dict) -> None:
    study = run_study(project, n_trials=2)
    runs = load_runs(project)
    trials = runs[(runs["parent"] != "") & (runs.get("tags.optuna_study") == "test-study")]
    assert sorted(trials["run_name"]) == ["trial_000", "trial_001"]
    logged = dict(zip(trials["run_name"], trials["metrics.cv_auc_mean"], strict=True))
    assert logged["trial_000"] == pytest.approx(study.trials[0].value), \
        "return the same mean CV AUC that you log"
    assert trials["params.learning_rate"].notna().all(), "log the trial's params"
    assert (trials["tags.data_version"] != "").all(), "use run_tags for the child runs"

    study = run_study(project, n_trials=1)  # a second tune continues the saved study
    assert len(study.trials) == 3
    best = json.loads(project["week3"]["outputs"]["optuna_best"].read_text())
    assert best["n_trials"] == 3 and best["cv_auc_mean"] == pytest.approx(study.best_value)


# ---------------------------------------------------------------- calibrate

def test_calibrate_wraps_the_model(project: dict, data) -> None:
    model = make_model(SMALL_HGB, data.numeric, data.categorical, project)
    wrapped = calibrate(model, "sigmoid", n_folds=3, seed=7)
    assert isinstance(wrapped, CalibratedClassifierCV)
    assert wrapped.method == "sigmoid" and wrapped.ensemble is False
    assert isinstance(wrapped.cv, StratifiedKFold)
    assert wrapped.cv.n_splits == 3 and wrapped.cv.shuffle and wrapped.cv.random_state == 7
    with pytest.raises(NotFittedError):
        check_is_fitted(wrapped)


def test_calibrate_rejects_unknown_methods(project: dict, data) -> None:
    model = make_model(SMALL_HGB, data.numeric, data.categorical, project)
    with pytest.raises(ValueError, match="isotonic"):
        calibrate(model, "platt", n_folds=3, seed=0)


def test_calibration_repairs_shifted_probabilities(project: dict, data) -> None:
    """Class weights push every score up; calibration brings the average back to the rate."""
    spec = {"model": "hist_gradient_boosting",
            "params": {"max_iter": 30, "class_weight": "balanced"}}
    model = make_model(spec, data.numeric, data.categorical, project)
    rate = data.y_train.mean()
    raw = model.fit(data.X_train, data.y_train).predict_proba(data.X_train)[:, 1].mean()
    fixed = calibrate(make_model(spec, data.numeric, data.categorical, project), "isotonic",
                      3, 0).fit(data.X_train, data.y_train)
    assert raw > rate + 0.1
    assert abs(fixed.predict_proba(data.X_train)[:, 1].mean() - rate) < 0.03



def test_register_model_sets_the_alias(project: dict, data) -> None:
    model = make_model(SMALL_HGB, data.numeric, data.categorical, project)
    model.fit(data.X_train, data.y_train)
    with mlflow.start_run(run_name="register_test"):
        info = register_model(model, data.X_valid.head(3), project, threshold=0.12)
    assert info["name"] == "credit-risk" and info["alias"] == "staging"
    assert info["uri"] == "models:/credit-risk@staging"
    version = MlflowClient().get_model_version_by_alias("credit-risk", "staging")
    assert str(version.version) == info["version"]
    assert version.tags["threshold"] == "0.12"
    loaded = mlflow.sklearn.load_model(info["uri"])
    np.testing.assert_allclose(loaded.predict_proba(data.X_valid),
                               model.predict_proba(data.X_valid))
