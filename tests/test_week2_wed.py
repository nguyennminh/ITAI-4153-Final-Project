"""Week 2 Wednesday tests. Tests of TODO(student) functions fail until you write them.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.

Every expected number below can be worked out by hand from tests/week2_fixtures.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import compare, make_features, validate
from src.features import common, ratios
from src.features.bureau import aggregate_bureau
from src.features.join import join_features
from src.features.payments import (
    CREDIT_CARD_FEATURES,
    INSTALLMENTS_FEATURES,
    aggregate_credit_card,
    aggregate_installments,
)
from src.features.ratios import RATIO_FEATURES, add_ratio_features
from tests.conftest import REPO_ROOT, SENTINEL
from tests.week2_fixtures import (
    make_full_application,
    make_week2_child_tables,
    make_week2_project,
    small_application_full,
    small_bureau,
    small_bureau_balance,
    small_credit_card,
    small_installments,
    small_pos_cash,
    small_previous,
)


def by_id(df: pd.DataFrame) -> pd.DataFrame:
    """Index a per-applicant result by applicant id, so tests can look rows up."""
    assert "SK_ID_CURR" in df.columns, "return SK_ID_CURR as a column (use reset_index)"
    assert not df["SK_ID_CURR"].duplicated().any(), "one row per applicant"
    return df.set_index("SK_ID_CURR")


def raw_tables() -> dict[str, pd.DataFrame]:
    """The seven small raw tables, clean."""
    return {
        "application_train": small_application_full(),
        "bureau": small_bureau(),
        "bureau_balance": small_bureau_balance(),
        "previous_application": small_previous(),
        "POS_CASH_balance": small_pos_cash(),
        "installments_payments": small_installments(),
        "credit_card_balance": small_credit_card(),
    }


def input_problems(tables: dict[str, pd.DataFrame]) -> list[str]:
    """The problems validate_inputs reports ([] when it passes)."""
    try:
        validate.validate_inputs(tables)
    except validate.DataValidationError as err:
        return err.problems
    return []


# ----------------------------------------------------------------------------- provided code


def test_safe_divide_gives_nan_not_inf():
    out = common.safe_divide(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 0.0, np.nan]))
    assert out.iloc[0] == 0.5
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])


def test_bureau_schema_worked_example():
    assert validate.run_schema(validate.bureau_schema(), small_bureau(), "bureau") == []
    bad = small_bureau().assign(CREDIT_ACTIVE=["Active", "active", "Active", "Closed", "Closed"])
    problems = validate.run_schema(validate.bureau_schema(), bad, "bureau")
    assert len(problems) == 1 and "CREDIT_ACTIVE" in problems[0]


def test_describe_failures_names_a_missing_column():
    problems = validate.run_schema(validate.bureau_schema(),
                                   small_bureau().drop(columns="DAYS_CREDIT"), "bureau")
    assert problems == ["bureau: missing column(s) DAYS_CREDIT"]


# ----------------------------------------------------------------------------- installments


def test_installments_columns_and_counts():
    out = by_id(aggregate_installments(small_installments()))
    assert sorted(out.columns) == sorted(INSTALLMENTS_FEATURES)
    assert out.loc[1, "INST_COUNT"] == 4  # the unpaid row still counts
    assert out.loc[2, "INST_COUNT"] == 1


def test_installments_late_and_underpaid_shares_skip_missing_payments():
    out = by_id(aggregate_installments(small_installments()))
    assert out.loc[1, "INST_LATE_SHARE"] == pytest.approx(1 / 3)       # not 1/4
    assert out.loc[1, "INST_UNDERPAID_SHARE"] == pytest.approx(1 / 3)
    assert out.loc[2, "INST_LATE_SHARE"] == pytest.approx(1.0)
    assert out.loc[2, "INST_UNDERPAID_SHARE"] == pytest.approx(0.0)


def test_installments_days_late_never_negative():
    out = by_id(aggregate_installments(small_installments()))
    assert out.loc[1, "INST_DAYS_LATE_MAX"] == 10
    assert out.loc[2, "INST_DAYS_LATE_MAX"] == 20
    assert out.loc[3, "INST_DAYS_LATE_MAX"] == 0  # paid 5 days early: 0, not -5


def test_installments_payment_difference():
    out = by_id(aggregate_installments(small_installments()))
    assert out.loc[1, "INST_PAYMENT_DIFF_MEAN"] == pytest.approx(-20 / 3)
    assert out.loc[2, "INST_PAYMENT_DIFF_MEAN"] == pytest.approx(50.0)


# ----------------------------------------------------------------------------- credit card


def test_credit_card_values():
    out = by_id(aggregate_credit_card(small_credit_card()))
    assert sorted(out.columns) == sorted(CREDIT_CARD_FEATURES)
    one = out.loc[1]
    assert one["CC_MONTHS_COUNT"] == 3
    assert one["CC_BALANCE_MEAN"] == pytest.approx(1390 / 3)
    assert one["CC_UTILIZATION_MEAN"] == pytest.approx((0.5 + 0.9 - 0.01) / 3)
    assert one["CC_UTILIZATION_MAX"] == pytest.approx(0.9)
    assert one["CC_DRAWINGS_MEAN"] == pytest.approx(100 / 3)
    assert one["CC_DPD_MAX"] == 3


def test_credit_card_zero_limit_gives_missing_utilization_not_inf():
    out = by_id(aggregate_credit_card(small_credit_card()))
    assert np.isnan(out.loc[2, "CC_UTILIZATION_MEAN"])
    assert np.isnan(out.loc[2, "CC_UTILIZATION_MAX"])
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    assert out.loc[2, "CC_BALANCE_MEAN"] == pytest.approx(250.0)


# ----------------------------------------------------------------------------- ratios


def test_ratio_values():
    out = by_id(add_ratio_features(small_application_full()))
    assert out.loc[3, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)
    assert out.loc[3, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.15)
    assert out.loc[3, "CREDIT_GOODS_RATIO"] == pytest.approx(300000 / 270000)
    assert out.loc[3, "ANNUITY_CREDIT_RATIO"] == pytest.approx(0.05)


def test_ratio_zero_or_missing_denominator_gives_nan():
    app = small_application_full()
    app.loc[app["SK_ID_CURR"] == 2, "AMT_GOODS_PRICE"] = 0.0
    out = by_id(add_ratio_features(app))
    assert np.isnan(out.loc[2, "CREDIT_GOODS_RATIO"])    # goods price 0
    assert np.isnan(out.loc[5, "CREDIT_GOODS_RATIO"])    # goods price missing
    assert np.isnan(out.loc[2, "ANNUITY_INCOME_RATIO"])  # annuity missing
    assert not np.isinf(out[RATIO_FEATURES].to_numpy()).any()


def test_ratios_keep_input_and_other_columns():
    app = small_application_full()
    before = app.copy()
    out = add_ratio_features(app)
    pd.testing.assert_frame_equal(app, before)
    pd.testing.assert_frame_equal(out[app.columns], app)
    assert set(RATIO_FEATURES) <= set(out.columns)


# ----------------------------------------------------------------------------- input validation


def test_input_schema_clean_tables_pass():
    assert input_problems(raw_tables()) == []


def test_input_schema_allows_sentinel_and_missing_annuity():
    tables = raw_tables()
    assert (tables["application_train"]["DAYS_EMPLOYED"] == SENTINEL).any()
    assert tables["application_train"]["AMT_ANNUITY"].isna().any()
    assert input_problems(tables) == []


@pytest.mark.parametrize("column, corrupt", [
    ("AMT_INCOME_TOTAL", lambda df: df.rename(columns={"AMT_INCOME_TOTAL": "AMT_INCOME"})),
    ("AMT_CREDIT", lambda df: df.assign(AMT_CREDIT=[np.nan, 1.0, 1.0, 1.0, 1.0])),
    ("AMT_CREDIT", lambda df: df.assign(AMT_CREDIT=[-5.0, 1.0, 1.0, 1.0, 1.0])),
    ("DAYS_BIRTH", lambda df: df.assign(DAYS_BIRTH=[12000, -1, -1, -1, -1])),
    ("AMT_ANNUITY", lambda df: df.assign(AMT_ANNUITY=[-1.0, 1.0, 1.0, 1.0, 1.0])),
    ("DAYS_EMPLOYED", lambda df: df.assign(DAYS_EMPLOYED=[500, -1, -1, -1, -1])),
    ("TARGET", lambda df: df.assign(TARGET=[2, 0, 0, 0, 1])),
    ("SK_ID_CURR", lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True)),
])
def test_input_schema_catches_application_problems(column, corrupt):
    tables = raw_tables()
    tables["application_train"] = corrupt(tables["application_train"])
    problems = input_problems(tables)
    assert problems, "validate_inputs should have failed"
    assert any(column in p and p.startswith("application_train") for p in problems), problems


@pytest.mark.parametrize("table, column, corrupt", [
    ("previous_application", "NAME_CONTRACT_STATUS",
     lambda df: df.assign(NAME_CONTRACT_STATUS=df["NAME_CONTRACT_STATUS"].str.upper())),
    ("bureau_balance", "STATUS", lambda df: df.assign(STATUS=df["STATUS"].replace("C", "c"))),
    ("POS_CASH_balance", "SK_DPD", lambda df: df.assign(SK_DPD=-df["SK_DPD"] - 1)),
    ("installments_payments", "AMT_PAYMENT", lambda df: df.assign(AMT_PAYMENT=-df["AMT_PAYMENT"])),
    ("credit_card_balance", "AMT_CREDIT_LIMIT_ACTUAL",
     lambda df: df.assign(AMT_CREDIT_LIMIT_ACTUAL=-1.0)),
    ("installments_payments", "SK_ID_CURR",
     lambda df: df.assign(SK_ID_CURR=[1, 1, np.nan, 1, 2, 3])),
    ("credit_card_balance", "SK_DPD", lambda df: df.drop(columns="SK_DPD")),
])
def test_input_schema_catches_child_table_problems(table, column, corrupt):
    tables = raw_tables()
    tables[table] = corrupt(tables[table])
    problems = input_problems(tables)
    assert any(column in p and p.startswith(table) for p in problems), problems


def test_input_schema_reports_all_problems_together():
    tables = raw_tables()
    app = tables["application_train"].rename(columns={"AMT_INCOME_TOTAL": "AMT_INCOME"})
    tables["application_train"] = app.assign(DAYS_BIRTH=[1, -1, -1, -1, -1])
    tables["installments_payments"] = small_installments().assign(AMT_PAYMENT=-1.0)
    problems = input_problems(tables)
    assert len(problems) >= 3, problems


# ----------------------------------------------------------------------------- feature validation


def small_feature_table() -> pd.DataFrame:
    """Two feature columns of each kind, one row per applicant."""
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "X_COUNT": [0, 2, 1, 5],
        "X_LATE_SHARE": [0.0, 0.5, np.nan, 1.0],
        "X_APPROVED_RATE": [0.2, 1.0, 0.0, np.nan],
        "X_CREDIT_RATIO": [1.5, np.nan, 0.0, 3.0],
        "X_AMT_MEAN": [10.0, np.nan, np.nan, -4.0],
    })


FEATURES = ["X_COUNT", "X_LATE_SHARE", "X_APPROVED_RATE", "X_CREDIT_RATIO", "X_AMT_MEAN"]
LIMITS = {"default": 0.5}


def feature_problems(df: pd.DataFrame, n: int = 4, limits: dict | None = None) -> list[str]:
    schema = validate.feature_schema(FEATURES, n, limits or LIMITS)
    return validate.run_schema(schema, df, "features")


def test_feature_schema_clean_table_passes():
    assert feature_problems(small_feature_table()) == []


@pytest.mark.parametrize("column, change", [
    ("X_COUNT", {"X_COUNT": [0, -1, 1, 5]}),
    ("X_COUNT", {"X_COUNT": [0, np.nan, 1, 5]}),
    ("X_LATE_SHARE", {"X_LATE_SHARE": [0.0, 1.5, np.nan, 1.0]}),
    ("X_APPROVED_RATE", {"X_APPROVED_RATE": [-0.1, 1.0, 0.0, np.nan]}),
    ("X_CREDIT_RATIO", {"X_CREDIT_RATIO": [1.5, np.inf, 0.0, 3.0]}),
    ("X_CREDIT_RATIO", {"X_CREDIT_RATIO": [-2.0, np.nan, 0.0, 3.0]}),
    ("X_AMT_MEAN", {"X_AMT_MEAN": [np.nan, np.nan, np.nan, -4.0]}),  # 75% missing > 50%
    ("SK_ID_CURR", {"SK_ID_CURR": [1, 2, 2, 4]}),
])
def test_feature_schema_catches_problems(column, change):
    problems = feature_problems(small_feature_table().assign(**change))
    assert any(column in p for p in problems), problems


def test_feature_schema_catches_missing_column():
    problems = feature_problems(small_feature_table().drop(columns="X_AMT_MEAN"))
    assert any("X_AMT_MEAN" in p for p in problems), problems


def test_feature_schema_catches_wrong_row_count():
    assert feature_problems(small_feature_table(), n=5)


def test_feature_schema_named_null_limit_checks_other_column():
    df = small_feature_table().assign(AMT_CREDIT=[1.0, np.nan, 2.0, 3.0])
    assert feature_problems(df, limits={"default": 0.5, "AMT_CREDIT": 0.0})
    assert not feature_problems(df.assign(AMT_CREDIT=1.0),
                                limits={"default": 0.5, "AMT_CREDIT": 0.0})


# ----------------------------------------------------------------------------- make_features


def test_build_feature_table_end_to_end(tmp_path):
    config = make_week2_project(tmp_path)
    tables = make_features.load_tables(config, config["paths"]["sample_dir"])
    features = make_features.build_feature_table(tables, config)
    app = tables["application_train"]
    assert features["SK_ID_CURR"].tolist() == app["SK_ID_CURR"].tolist()
    missing = [c for c in make_features.EXPECTED_FEATURES if c not in features.columns]
    assert not missing, missing
    pd.testing.assert_frame_equal(features[app.columns], app)  # application columns untouched


def test_build_feature_table_fails_before_aggregating(tmp_path):
    config = make_week2_project(tmp_path)
    tables = make_features.load_tables(config, config["paths"]["sample_dir"])
    tables["bureau"] = tables["bureau"].drop(columns="CREDIT_ACTIVE")
    with pytest.raises(validate.DataValidationError, match="CREDIT_ACTIVE"):
        make_features.build_feature_table(tables, config)  # not a KeyError


def test_build_feature_table_validates_its_output(tmp_path, monkeypatch):
    config = make_week2_project(tmp_path)
    tables = make_features.load_tables(config, config["paths"]["sample_dir"])
    real = ratios.add_ratio_features

    def broken(df: pd.DataFrame) -> pd.DataFrame:
        return real(df).assign(CREDIT_INCOME_RATIO=np.inf)

    monkeypatch.setattr(make_features, "add_ratio_features", broken)
    monkeypatch.setattr(ratios, "add_ratio_features", broken)
    with pytest.raises(validate.DataValidationError, match="CREDIT_INCOME_RATIO"):
        make_features.build_feature_table(tables, config)


def test_command_line_writes_table_and_report(tmp_path):
    config = make_week2_project(tmp_path)
    shutil_copy_code(tmp_path)
    result = subprocess.run([sys.executable, "-m", "src.make_features"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    table = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
    assert len(table) == config["sample_size"]
    report = json.loads((tmp_path / "reports" / "validation_sample.json").read_text())
    assert report["status"] == "pass"


def test_command_line_fails_loudly_on_bad_data(tmp_path):
    make_week2_project(tmp_path)
    shutil_copy_code(tmp_path)
    bad = tmp_path / "data" / "bad"
    bad.mkdir()
    for csv in (tmp_path / "data" / "sample").glob("*.csv"):
        (bad / csv.name).write_bytes(csv.read_bytes())
    app = pd.read_csv(bad / "application_train.csv")
    app.rename(columns={"AMT_CREDIT": "AMT_CREDITS"}).to_csv(bad / "application_train.csv",
                                                             index=False)
    result = subprocess.run([sys.executable, "-m", "src.make_features", "--sample-dir",
                             "data/bad"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "AMT_CREDIT" in result.stdout and "Traceback" not in result.stderr
    assert not (tmp_path / "data" / "features" / "features_bad.parquet").exists()
    report = json.loads((tmp_path / "reports" / "validation_bad.json").read_text())
    assert report["status"] == "fail" and report["problems"]


def shutil_copy_code(root):
    """Copy src/ (your code) into a temporary project so the command can run there."""
    shutil.copytree(REPO_ROOT / "src", root / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))


# ----------------------------------------------------------------------------- comparison


def test_compare_runs_every_table_and_the_ratios(tmp_path):
    config = make_week2_project(tmp_path, n_bootstrap=20)
    report = compare.run_comparison(config)
    names = [r["feature_set"] for r in report["results"]]
    assert names == list(config["week2"]["feature_sets"])
    assert report["results"][-1]["n_features"] > report["results"][-2]["n_features"]


def test_full_join_on_generated_data_keeps_one_row_per_applicant():
    app = make_full_application()
    tables = make_week2_child_tables(app["SK_ID_CURR"].to_numpy())
    joined = join_features(app, {
        "bureau": aggregate_bureau(tables["bureau"]),
        "installments_payments": aggregate_installments(tables["installments_payments"]),
        "credit_card_balance": aggregate_credit_card(tables["credit_card_balance"]),
    })
    assert len(joined) == len(app)
    assert (joined["CC_MONTHS_COUNT"] >= 0).all()
