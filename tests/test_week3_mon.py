"""Week 3 Monday tests. Tests of TODO(student) functions fail until you write them.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.

The data is made up (tests/week3_fixtures.py), so the tests run in seconds without Kaggle.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from src import train
from src.evaluation import cross_validate_auc
from src.experiment import data_version, load_training_data
from src.models import describe_spec, make_model, undersample
from src.tracking import git_commit, load_runs, setup_mlflow
from tests.week3_fixtures import SpyClassifier, make_feature_table, make_week3_project, spy_data


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One temporary project (with its own MLflow database) shared by this file's tests."""
    config = make_week3_project(tmp_path_factory.mktemp("week3"))
    setup_mlflow(config)
    return config


@pytest.fixture(scope="module")
def data(project: dict):
    return load_training_data(project)


# ---------------------------------------------------------------- provided code (passes now)

def test_training_data_uses_the_fixed_split(project: dict, data) -> None:
    split = pd.read_csv(project["paths"]["split_file"])
    assert len(data.X_train) == (split["split"] == "train").sum()
    assert len(data.X_valid) == (split["split"] == "valid").sum()
    assert "TARGET" not in data.features and "SK_ID_CURR" not in data.features
    assert "DAYS_EMPLOYED_ANOM" in data.features, "your Week 1 clean_application runs first"
    assert data.data_version == load_training_data(project).data_version


def test_data_version_changes_with_the_data() -> None:
    table = make_feature_table()
    split = pd.DataFrame({"SK_ID_CURR": table["SK_ID_CURR"], "split": "train"})
    changed = table.copy()
    changed.loc[0, "SIGNAL"] += 1
    assert data_version(table, split) == data_version(table.copy(), split.copy())
    assert data_version(table, split) != data_version(changed, split)


def test_logistic_regression_is_your_week1_pipeline(project: dict, data) -> None:
    model = make_model({"model": "logistic_regression"}, data.numeric, data.categorical, project)
    assert isinstance(model.steps[-1][1], LogisticRegression)


def test_describe_spec() -> None:
    assert describe_spec({"model": "hist_gradient_boosting"}) == ""
    spec = {"model": "x", "params": {"max_iter": 9, "learning_rate": 0.1}, "undersample": 3}
    assert describe_spec(spec) == "learning_rate=0.1, max_iter=9, undersample=3"


def test_git_commit_outside_a_repository(tmp_path) -> None:
    assert git_commit(tmp_path) == "no-git"


# ---------------------------------------------------------------- make_model

@pytest.mark.parametrize("name, estimator, params", [
    ("decision_tree", DecisionTreeClassifier, {"max_depth": 3, "min_samples_leaf": 7}),
    ("hist_gradient_boosting", HistGradientBoostingClassifier, {"learning_rate": 0.05}),
])
def test_make_model_builds_a_seeded_pipeline(project, data, name, estimator, params) -> None:
    model = make_model({"model": name, "params": params}, data.numeric, data.categorical,
                       project)
    assert isinstance(model, Pipeline)
    assert [step for step, _ in model.steps] == ["preprocess", "model"]
    final = model.named_steps["model"]
    assert isinstance(final, estimator)
    assert final.random_state == project["seed"]
    for key, value in params.items():
        assert getattr(final, key) == value, f"{key} from spec['params'] was not passed on"
    model.fit(data.X_train, data.y_train)  # NaN and text columns must not break it
    assert model.predict_proba(data.X_valid).shape == (len(data.X_valid), 2)


def test_make_model_defaults_when_no_params(project, data) -> None:
    model = make_model({"model": "decision_tree"}, data.numeric, data.categorical, project)
    assert model.named_steps["model"].max_depth is None


def test_make_model_rejects_unknown_names(project, data) -> None:
    with pytest.raises(ValueError, match="random_forest"):
        make_model({"model": "random_forest"}, data.numeric, data.categorical, project)


# ---------------------------------------------------------------- undersample

def test_undersample_keeps_every_defaulter() -> None:
    X, y = spy_data(300, 45)
    X_kept, y_kept = undersample(X, y, 2, seed=0)
    assert int(y_kept.sum()) == 45
    assert int((y_kept == 0).sum()) == 90
    assert set(X.loc[y == 1, "SK_ID"]) <= set(X_kept["SK_ID"])


def test_undersample_keeps_order_and_alignment() -> None:
    X, y = spy_data(300, 45)
    before = X.copy()
    X_kept, y_kept = undersample(X, y, 2, seed=0)
    pd.testing.assert_frame_equal(X, before)  # the input is not changed
    assert list(X_kept.index) == list(range(len(X_kept)))
    assert list(y_kept.index) == list(range(len(y_kept)))
    assert X_kept["SK_ID"].is_monotonic_increasing, "keep the original row order"
    labels = dict(zip(X["SK_ID"], y, strict=True))
    assert all(labels[i] == t for i, t in zip(X_kept["SK_ID"], y_kept, strict=True))


def test_undersample_is_repeatable() -> None:
    X, y = spy_data(300, 45)
    first = undersample(X, y, 2, seed=1)[0]["SK_ID"].tolist()
    assert first == undersample(X, y, 2, seed=1)[0]["SK_ID"].tolist()
    assert first != undersample(X, y, 2, seed=2)[0]["SK_ID"].tolist()


def test_undersample_keeps_all_when_too_few() -> None:
    X, y = spy_data(100, 40)
    X_kept, y_kept = undersample(X, y, 5, seed=0)
    assert len(X_kept) == 100 and int(y_kept.sum()) == 40


# ---------------------------------------------------------------- cross_validate_auc

def test_cv_returns_folds_and_oof_scores(project, data) -> None:
    model = make_model({"model": "hist_gradient_boosting", "params": {"max_iter": 20}},
                       data.numeric, data.categorical, project)
    result = cross_validate_auc(model, data.X_train, data.y_train, n_folds=3, seed=0)
    assert len(result["fold_aucs"]) == 3
    assert result["mean"] == pytest.approx(np.mean(result["fold_aucs"]))
    assert result["std"] == pytest.approx(np.std(result["fold_aucs"]))
    assert len(result["oof"]) == len(data.y_train)
    assert not np.isnan(result["oof"]).any(), "every row needs an out-of-fold score"
    assert result["mean"] > 0.7, "SIGNAL predicts TARGET, so the AUC should be well above 0.5"
    with pytest.raises(NotFittedError):
        check_is_fitted(model)  # fit copies (clone), never the model you were given


def test_cv_scores_are_out_of_fold(project) -> None:
    """A tree with no limits memorizes. Scored on rows it never saw, it must look useless."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"A": rng.normal(size=300), "B": rng.normal(size=300)})
    y = pd.Series((rng.random(300) < 0.2).astype(int))
    model = make_model({"model": "decision_tree"}, ["A", "B"], [], project)
    result = cross_validate_auc(model, X, y, n_folds=5, seed=0)
    assert roc_auc_score(y, model.fit(X, y).predict_proba(X)[:, 1]) == 1.0  # memorized
    assert roc_auc_score(y, result["oof"]) < 0.65, "oof[i] must come from a model without row i"


def test_cv_folds_are_stratified_and_shuffled() -> None:
    X, y = spy_data(300, 45)
    SpyClassifier.fits.clear()
    cross_validate_auc(SpyClassifier(), X, y, n_folds=5, seed=0)
    assert len(SpyClassifier.fits) == 5
    assert all(fit["n_pos"] == 36 for fit in SpyClassifier.fits), "use StratifiedKFold"
    held_out = [set(X["SK_ID"]) - fit["ids"] for fit in SpyClassifier.fits]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y)
    expected = [set(X["SK_ID"].iloc[test]) for _, test in folds]
    assert held_out == expected, "StratifiedKFold(n_splits, shuffle=True, random_state=seed)"


def test_cv_undersamples_training_folds_only() -> None:
    X, y = spy_data(300, 45)
    SpyClassifier.fits.clear()
    result = cross_validate_auc(SpyClassifier(), X, y, n_folds=5, seed=0, undersample_ratio=1)
    assert all(fit["n"] == 72 for fit in SpyClassifier.fits), "36 defaulters + 36 others"
    assert not np.isnan(result["oof"]).any(), "the held-out fold is scored in full"


# ---------------------------------------------------------------- train_and_log (provided)

def test_train_and_log_records_a_complete_run(project, data) -> None:
    result = train.train_and_log("test_run", {"model": "decision_tree",
                                              "params": {"max_depth": 3}}, data, project)
    run = mlflow.get_run(result["run_id"])
    for key in ("cv_auc_mean", "train_auc", "valid_auc", "valid_auc_ci_low", "latency_p95_ms",
                "model_size_kb"):
        assert key in run.data.metrics
    assert run.data.params["max_depth"] == "3"
    assert run.data.tags["data_version"] == data.data_version
    assert run.data.tags["settings"] == "max_depth=3"


# ---------------------------------------------------------------- tune_tree

def test_tune_tree_chooses_by_cross_validation_and_retrains(project, data, monkeypatch) -> None:
    calls = []

    def fake_train_and_log(name, spec, data, config, nested=False, log_model=True):
        active = mlflow.active_run()
        parent = active.info.run_name if active is not None else None
        calls.append({"name": name, "spec": spec, "nested": nested, "log_model": log_model,
                      "parent": parent})
        p = spec["params"]
        cv = {(2, 5): 0.70, (2, 40): 0.74, (4, 5): 0.72, (4, 40): 0.74}[
            (p["max_depth"], p["min_samples_leaf"])]
        # Validation AUC disagrees with CV on purpose: the choice must follow CV.
        return {"run_id": "x", "name": name, "spec": spec, "cv_auc_mean": cv,
                "valid_auc": 1 - cv}

    monkeypatch.setattr(train, "train_and_log", fake_train_and_log)
    result = train.tune_tree(data, project)

    grid_calls, final = calls[:-1], calls[-1]
    tried = {(c["spec"]["params"]["max_depth"], c["spec"]["params"]["min_samples_leaf"])
             for c in grid_calls}
    assert tried == {(2, 5), (2, 40), (4, 5), (4, 40)}, "try every combination once"
    assert all(c["nested"] and not c["log_model"] for c in grid_calls)
    assert all(c["parent"] == "tree_tuning" for c in grid_calls), "inside a 'tree_tuning' run"
    assert final["name"] == "tree_tuned" and not final["nested"] and final["parent"] is None
    assert final["spec"]["params"] == {"max_depth": 2, "min_samples_leaf": 40}, \
        "highest CV AUC; on a tie, the first combination"
    assert result["name"] == "tree_tuned"

    runs = load_runs(project)
    parent = runs[runs["run_name"] == "tree_tuning"].iloc[0]
    assert parent["params.best_max_depth"] == "2"
    assert parent["params.best_min_samples_leaf"] == "40"
    assert parent["metrics.best_cv_auc_mean"] == pytest.approx(0.74)
