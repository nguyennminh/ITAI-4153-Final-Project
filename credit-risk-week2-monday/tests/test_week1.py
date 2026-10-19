"""Week 1 tests. Tests of TODO(student) functions fail until you write them.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import baseline, data, metrics, summarize, utils
from tests.conftest import REPO_ROOT, SENTINEL

# ----------------------------------------------------------------------------- provided code


def test_load_config_resolves_paths(project):
    root = project["root"]
    assert project["paths"]["sample_dir"] == root / "data" / "sample"
    assert project["paths"]["split_file"].is_absolute()
    assert len(project["tables"]) == 7


def test_reduce_memory_shrinks_and_keeps_values(app_df):
    before = app_df.copy()
    small = utils.reduce_memory(app_df)
    pd.testing.assert_frame_equal(app_df, before)  # input unchanged
    assert utils.memory_mb(small) <= utils.memory_mb(app_df)
    assert small["DAYS_BIRTH"].dtype.itemsize < app_df["DAYS_BIRTH"].dtype.itemsize
    assert small["AMT_CREDIT"].dtype == np.float32
    assert (small["DAYS_BIRTH"] == app_df["DAYS_BIRTH"]).all()


def test_load_table_and_application(project):
    app = data.load_application(project)
    assert len(app) == 500
    assert "TARGET" in app.columns
    assert len(data.load_table("bureau_balance", project)) > 0


def test_roc_and_pr_auc():
    y = np.array([0, 0, 1, 1])
    assert metrics.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.pr_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- ks_statistic


def test_ks_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0)


def test_ks_random_scores_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=20000)
    assert metrics.ks_statistic(y, rng.random(20000)) < 0.05


def test_ks_known_value():
    # Defaulters score 0.35 and 0.8; non-defaulters 0.1 and 0.4. Largest gap is 0.5.
    y = np.array([0, 0, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.5)


def test_ks_returns_python_float():
    assert isinstance(metrics.ks_statistic([0, 1, 0, 1], [0.2, 0.6, 0.3, 0.9]), float)


def test_evaluate_keys():
    result = metrics.evaluate(np.array([0, 1, 0, 1]), np.array([0.2, 0.6, 0.3, 0.9]))
    assert set(result) == {"roc_auc", "pr_auc", "ks"}
    assert all(isinstance(v, float) for v in result.values())


# ----------------------------------------------------------------------------- load_split


def test_load_split_no_overlap_and_complete(project):
    train, valid = data.load_split(project)
    app = data.load_application(project)
    train_ids, valid_ids = set(train["SK_ID_CURR"]), set(valid["SK_ID_CURR"])
    assert not train_ids & valid_ids
    assert train_ids | valid_ids == set(app["SK_ID_CURR"])
    assert (len(train), len(valid)) == (400, 100)
    assert list(train.columns) == list(app.columns)
    assert list(valid.index) == list(range(len(valid)))


def test_load_split_rejects_duplicate_ids(project):
    path = project["paths"]["split_file"]
    split = pd.read_csv(path)
    split.loc[0, "SK_ID_CURR"] = split.loc[1, "SK_ID_CURR"]
    split.to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.load_split(project)


# ----------------------------------------------------------------------------- clean_application


def test_clean_replaces_sentinel_and_adds_flag(app_df):
    n_sentinel = int((app_df["DAYS_EMPLOYED"] == SENTINEL).sum())
    cleaned = baseline.clean_application(app_df)
    assert not (cleaned["DAYS_EMPLOYED"] == SENTINEL).any()
    assert cleaned["DAYS_EMPLOYED"].isna().sum() == n_sentinel
    assert set(cleaned["DAYS_EMPLOYED_ANOM"].unique()) <= {0, 1}
    assert int(cleaned["DAYS_EMPLOYED_ANOM"].sum()) == n_sentinel
    assert len(cleaned) == len(app_df)


def test_clean_does_not_mutate_input_or_touch_target(app_df):
    before = app_df.copy()
    cleaned = baseline.clean_application(app_df)
    pd.testing.assert_frame_equal(app_df, before)
    pd.testing.assert_series_equal(cleaned["TARGET"], before["TARGET"])
    other = [c for c in before.columns if c != "DAYS_EMPLOYED"]
    pd.testing.assert_frame_equal(cleaned[other], before[other])


def test_clean_works_without_target(app_df):
    cleaned = baseline.clean_application(app_df.drop(columns="TARGET"))
    assert "TARGET" not in cleaned.columns
    assert "DAYS_EMPLOYED_ANOM" in cleaned.columns


# ------------------------------------------------------------------------- preprocessing and model


def test_get_feature_columns_excludes_id_and_target(app_df, project):
    numeric, categorical = baseline.get_feature_columns(app_df, project)
    assert "SK_ID_CURR" not in numeric + categorical
    assert "TARGET" not in numeric + categorical
    assert "AMT_CREDIT" in numeric
    assert "CODE_GENDER" in categorical


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)


def test_build_preprocessor_outputs_no_nan(app_df, project):
    cleaned = baseline.clean_application(app_df)
    numeric, categorical = baseline.get_feature_columns(cleaned, project)
    pre = baseline.build_preprocessor(numeric, categorical)
    out = _dense(pre.fit_transform(cleaned[numeric + categorical]))
    assert out.shape[0] == len(cleaned)
    assert not np.isnan(out).any()


def test_build_preprocessor_handles_unseen_category(app_df, project):
    cleaned = baseline.clean_application(app_df)
    numeric, categorical = baseline.get_feature_columns(cleaned, project)
    pre = baseline.build_preprocessor(numeric, categorical)
    pre.fit(cleaned[numeric + categorical])
    new = cleaned.head(5).copy()
    new["OCCUPATION_TYPE"] = "Astronaut"
    out = _dense(pre.transform(new[numeric + categorical]))
    assert out.shape[0] == 5
    assert not np.isnan(out).any()


def test_build_model_fits_and_predicts_proba(app_df, project):
    cleaned = baseline.clean_application(app_df)
    numeric, categorical = baseline.get_feature_columns(cleaned, project)
    model = baseline.build_model(project, baseline.build_preprocessor(numeric, categorical))
    assert isinstance(model, Pipeline)
    final = model.steps[-1][1]
    assert isinstance(final, LogisticRegression)
    assert final.max_iter == project["baseline"]["max_iter"]
    model.fit(cleaned[numeric + categorical], cleaned["TARGET"])
    proba = model.predict_proba(cleaned[numeric + categorical])
    assert proba.shape == (len(cleaned), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


# ----------------------------------------------------------------------------- write_metrics


def test_write_metrics_schema(tmp_path):
    path = tmp_path / "nested" / "metrics.json"
    extra = {"model": "logistic_regression", "n_train": 400, "n_valid": 100,
             "valid_default_rate": np.float64(0.08), "seed": 42}
    baseline.write_metrics({"roc_auc": 0.7, "pr_auc": 0.2, "ks": 0.3}, extra, path)
    written = json.loads(path.read_text())
    assert set(baseline.METRICS_KEYS) <= set(written)
    assert written["roc_auc"] == pytest.approx(0.7)
    assert written["valid_default_rate"] == pytest.approx(0.08)


def test_write_metrics_rejects_missing_keys(tmp_path):
    with pytest.raises(ValueError):
        baseline.write_metrics({"roc_auc": 0.7}, {"model": "x"}, tmp_path / "metrics.json")


def test_run_baseline_end_to_end(project):
    results = baseline.run_baseline(project)
    written = json.loads(project["paths"]["metrics_file"].read_text())
    assert set(baseline.METRICS_KEYS) <= set(written)
    assert (written["n_train"], written["n_valid"]) == (400, 100)
    assert 0.0 <= written["ks"] <= 1.0
    assert results["roc_auc"] == pytest.approx(written["roc_auc"])


# ----------------------------------------------------------------------------- summarize


def test_summarize_table_values(app_df):
    before = app_df.copy()
    row = summarize.summarize_table(app_df, "application_train")
    pd.testing.assert_frame_equal(app_df, before)  # input unchanged
    assert set(row) == set(summarize.SUMMARY_KEYS)
    assert row["table"] == "application_train"
    assert (row["rows"], row["cols"]) == app_df.shape
    assert row["mb_after"] <= row["mb_before"]
    expected_missing = 100 * app_df.isna().sum().sum() / app_df.size
    assert row["missing_pct"] == pytest.approx(expected_missing, abs=0.01)


def test_summarize_table_empty_frame():
    row = summarize.summarize_table(pd.DataFrame({"a": [], "b": []}), "empty")
    assert (row["rows"], row["cols"]) == (0, 2)
    assert row["missing_pct"] == 0


def test_summarize_all_writes_report(project):
    rows = summarize.summarize_all(project)
    path = project["paths"]["reports_dir"] / summarize.SUMMARY_FILENAME
    written = json.loads(path.read_text())
    assert len(rows) == len(written) == 7
    assert [r["table"] for r in written] == list(project["tables"])
    assert all(r["mb_after"] <= r["mb_before"] for r in written)


# ----------------------------------------------------------------------------- the real split file
# These run only once the instructor's config/split_ids.csv is in the repository.

SPLIT_FILE = REPO_ROOT / "config" / "split_ids.csv"
needs_split = pytest.mark.skipif(not SPLIT_FILE.exists(), reason="split file not added yet")


@pytest.fixture
def real_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_FILE)


@needs_split
def test_split_ids_unique_and_both_splits(real_split):
    assert real_split["SK_ID_CURR"].is_unique
    assert set(real_split["split"]) == {"train", "valid"}


@needs_split
def test_split_valid_fraction(real_split):
    assert (real_split["split"] == "valid").mean() == pytest.approx(0.2, abs=0.01)


@needs_split
def test_split_ids_in_sample_and_default_rates_similar(real_split):
    config = utils.load_config(REPO_ROOT / "config" / "config.yaml")
    app_path: Path = config["paths"]["sample_dir"] / config["tables"]["application_train"]
    if not app_path.exists():
        pytest.skip("sample not built yet (run python run.py sample)")
    app = pd.read_csv(app_path, usecols=["SK_ID_CURR", "TARGET"])
    assert set(real_split["SK_ID_CURR"]) == set(app["SK_ID_CURR"])
    merged = real_split.merge(app, on="SK_ID_CURR")
    rates = merged.groupby("split")["TARGET"].mean()
    assert abs(rates["train"] - rates["valid"]) < 0.02
