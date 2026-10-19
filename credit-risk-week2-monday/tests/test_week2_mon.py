"""Week 2 Monday tests. Tests of TODO(student) functions fail until you write them.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.

Every expected number below can be worked out by hand from tests/week2_fixtures.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import baseline, compare, uncertainty
from src.features import common
from src.features.bureau import (
    BUREAU_BALANCE_FEATURES,
    aggregate_bureau,
    aggregate_bureau_balance,
)
from src.features.join import assert_one_row_per_applicant, join_features
from src.features.previous import (
    POS_CASH_FEATURES,
    PREVIOUS_FEATURES,
    aggregate_pos_cash,
    aggregate_previous,
)
from tests.conftest import SENTINEL
from tests.week2_fixtures import (
    make_week2_project,
    small_application,
    small_bureau,
    small_bureau_balance,
    small_pos_cash,
    small_previous,
)


def by_id(df: pd.DataFrame) -> pd.DataFrame:
    """Index a per-applicant result by applicant id, so tests can look rows up."""
    assert "SK_ID_CURR" in df.columns, "return SK_ID_CURR as a column (use reset_index)"
    assert not df["SK_ID_CURR"].duplicated().any(), "one row per applicant"
    return df.set_index("SK_ID_CURR")


# ----------------------------------------------------------------------------- provided code


def test_replace_sentinel_leaves_input_alone():
    df = small_previous()
    out = common.replace_sentinel(df, ["DAYS_TERMINATION", "NOT_A_COLUMN"])
    assert out["DAYS_TERMINATION"].isna().sum() == 2
    assert (df["DAYS_TERMINATION"] == SENTINEL).sum() == 2  # input unchanged


def test_require_columns_names_table_and_columns():
    with pytest.raises(KeyError, match=r"bureau.*AMT_X"):
        common.require_columns(small_bureau(), ["SK_ID_CURR", "AMT_X"], "bureau")


def test_aggregate_bureau_worked_example():
    out = by_id(aggregate_bureau(small_bureau()))
    assert sorted(out.index) == [1, 2, 4]  # applicants 3 and 5 have no bureau rows
    one = out.loc[1]
    assert one["BUREAU_COUNT"] == 3
    assert one["BUREAU_ACTIVE_COUNT"] == 2
    assert one["BUREAU_CLOSED_COUNT"] == 1
    assert one["BUREAU_AMT_CREDIT_SUM_MEAN"] == pytest.approx(200.0)
    assert one["BUREAU_AMT_DEBT_MEAN"] == pytest.approx(75.0)  # the NaN is ignored
    assert one["BUREAU_DAYS_CREDIT_MAX"] == -50


def test_bootstrap_interval_contains_the_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    score = y + rng.normal(0, 1.2, size=400)
    low, high = uncertainty.bootstrap_auc_ci(y, score, n_boot=200)
    point = uncertainty.roc_auc_score(y, score)
    assert low < point < high
    assert (low, high) == uncertainty.bootstrap_auc_ci(y, score, n_boot=200)  # seeded


def test_bootstrap_difference_of_identical_scores_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    score = rng.random(200)
    assert uncertainty.bootstrap_auc_diff_ci(y, score, score, n_boot=50) == (0.0, 0.0)


# ----------------------------------------------------------------------------- previous_application


def test_previous_columns_and_one_row_per_applicant():
    out = aggregate_previous(small_previous())
    missing = [c for c in PREVIOUS_FEATURES if c not in out.columns]
    assert not missing, f"missing columns {missing}"
    assert sorted(by_id(out).index) == [1, 2]


def test_previous_counts_and_rates():
    out = by_id(aggregate_previous(small_previous()))
    assert out.loc[1, "PREV_COUNT"] == 3
    assert out.loc[1, "PREV_APPROVED_RATE"] == pytest.approx(1 / 3)
    assert out.loc[1, "PREV_REFUSED_RATE"] == pytest.approx(1 / 3)
    assert out.loc[2, "PREV_APPROVED_RATE"] == pytest.approx(0.5)
    assert out.loc[2, "PREV_REFUSED_RATE"] == pytest.approx(0.0)  # "Unused offer" is neither


def test_previous_means_and_recency():
    out = by_id(aggregate_previous(small_previous()))
    assert out.loc[1, "PREV_AMT_APPLICATION_MEAN"] == pytest.approx(2000.0)
    assert out.loc[1, "PREV_AMT_CREDIT_MEAN"] == pytest.approx(1300.0)
    assert out.loc[1, "PREV_DAYS_DECISION_MAX"] == -100
    assert out.loc[2, "PREV_DAYS_DECISION_MAX"] == -50


def test_previous_ignores_the_365243_sentinel():
    out = by_id(aggregate_previous(small_previous()))
    # Applicant 1 has one real DAYS_TERMINATION (-400) and two "no date" values.
    assert out.loc[1, "PREV_DAYS_TERMINATION_MEAN"] == pytest.approx(-400.0)
    assert out.loc[2, "PREV_DAYS_TERMINATION_MEAN"] == pytest.approx(-355.0)


def test_previous_last_status_breaks_ties_by_larger_sk_id_prev():
    out = by_id(aggregate_previous(small_previous()))
    assert out.loc[1, "PREV_LAST_STATUS"] == "Canceled"  # 102 and 103 tie; 103 wins
    assert out.loc[2, "PREV_LAST_STATUS"] == "Approved"


def test_previous_last_status_does_not_depend_on_row_order():
    shuffled = small_previous().sample(frac=1, random_state=3)
    out = by_id(aggregate_previous(shuffled))
    assert out.loc[1, "PREV_LAST_STATUS"] == "Canceled"


def test_previous_does_not_change_its_input():
    df = small_previous()
    before = df.copy()
    aggregate_previous(df)
    pd.testing.assert_frame_equal(df, before)


# ----------------------------------------------------------------------------- join


def test_join_keeps_every_applicant_in_order():
    app = small_application()
    joined = join_features(app, {"bureau": aggregate_bureau(small_bureau())})
    assert joined["SK_ID_CURR"].tolist() == [3, 1, 2, 5, 4]
    pd.testing.assert_frame_equal(joined[app.columns], app)


def test_join_fills_counts_with_zero_and_leaves_averages_missing():
    joined = by_id(join_features(small_application(),
                                 {"bureau": aggregate_bureau(small_bureau())}))
    assert joined.loc[3, "BUREAU_COUNT"] == 0
    assert joined.loc[5, "BUREAU_ACTIVE_COUNT"] == 0
    assert np.isnan(joined.loc[3, "BUREAU_AMT_CREDIT_SUM_MEAN"])
    assert joined.loc[1, "BUREAU_COUNT"] == 3


def test_join_two_tables():
    tables = {"bureau": aggregate_bureau(small_bureau()),
              "previous_application": aggregate_previous(small_previous())}
    joined = by_id(join_features(small_application(), tables))
    assert len(joined) == 5
    assert joined.loc[4, "PREV_COUNT"] == 0
    assert joined.loc[1, "PREV_LAST_STATUS"] == "Canceled"


def test_join_rejects_a_table_that_was_not_aggregated():
    with pytest.raises(ValueError, match="bureau"):
        join_features(small_application(), {"bureau": small_bureau()})  # raw: 3 rows for id 1


def test_join_rejects_duplicate_feature_names():
    table = aggregate_bureau(small_bureau())
    with pytest.raises(ValueError):
        join_features(small_application(), {"bureau": table, "bureau_again": table})


def test_join_does_not_change_its_inputs():
    app, table = small_application(), aggregate_bureau(small_bureau())
    app_before, table_before = app.copy(), table.copy()
    join_features(app, {"bureau": table})
    pd.testing.assert_frame_equal(app, app_before)
    pd.testing.assert_frame_equal(table, table_before)


def test_assert_one_row_accepts_a_good_table():
    app = small_application()
    assert_one_row_per_applicant(app, app["SK_ID_CURR"])


@pytest.mark.parametrize("ids", [
    [3, 1, 2, 5],            # one applicant missing
    [3, 1, 2, 5, 4, 4],      # one applicant twice
    [3, 1, 2, 5, 4, 99],     # an applicant who should not be there
    [3, 1, 2, 5, 5],         # same row count, but 4 is missing and 5 is doubled
])
def test_assert_one_row_rejects_bad_tables(ids):
    expected = small_application()["SK_ID_CURR"]
    with pytest.raises(ValueError):
        assert_one_row_per_applicant(pd.DataFrame({"SK_ID_CURR": ids}), expected)


# ----------------------------------------------------------------------------- comparison


def test_compare_application_set_matches_week1_baseline(tmp_path):
    config = make_week2_project(tmp_path)
    week1 = baseline.run_baseline(config)
    report = compare.run_comparison(config)
    rows = {r["feature_set"]: r for r in report["results"]}
    assert rows["application"]["roc_auc"] == pytest.approx(week1["roc_auc"])
    assert rows["application"]["delta_roc_auc"] == 0.0
    both = rows["application+bureau+previous"]
    assert both["n_features"] > rows["application"]["n_features"]
    low, high = both["roc_auc_ci"]
    assert low <= both["roc_auc"] <= high
    saved = json.loads(config["week2"]["paths"]["comparison_file"].read_text())
    assert [r["feature_set"] for r in saved["results"]] == list(rows)


# ----------------------------------------------------------------------------- carry-over


def test_bureau_balance_two_stage_values():
    out = by_id(aggregate_bureau_balance(small_bureau_balance(), small_bureau()))
    assert sorted(out.columns) == sorted(BUREAU_BALANCE_FEATURES)
    assert sorted(out.index) == [1, 2, 4]  # credit 13 has no history; applicant 3 none at all
    one = out.loc[1]
    assert one["BB_CREDIT_COUNT"] == 2       # credits 11 and 12, not 13
    assert one["BB_MONTHS_COUNT"] == 5
    assert one["BB_LATE_MONTHS_COUNT"] == 1
    assert one["BB_WORST_STATUS"] == 1


def test_bureau_balance_late_share_is_pooled_over_months():
    out = by_id(aggregate_bureau_balance(small_bureau_balance(), small_bureau()))
    # 1 late month out of 5. The mean of per-credit shares would give (1/3 + 0/2) / 2.
    assert out.loc[1, "BB_LATE_SHARE"] == pytest.approx(0.2)
    assert out.loc[2, "BB_LATE_SHARE"] == pytest.approx(0.5)
    assert out.loc[2, "BB_WORST_STATUS"] == 5


def test_bureau_balance_c_and_x_only_gives_missing_worst_status():
    out = by_id(aggregate_bureau_balance(small_bureau_balance(), small_bureau()))
    assert out.loc[4, "BB_LATE_SHARE"] == 0
    assert np.isnan(out.loc[4, "BB_WORST_STATUS"])


def test_bureau_balance_rejects_a_credit_listed_twice_in_bureau():
    bureau = pd.concat([small_bureau(), small_bureau().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):  # pandas' MergeError is a ValueError
        aggregate_bureau_balance(small_bureau_balance(), bureau)


def test_pos_cash_values():
    out = by_id(aggregate_pos_cash(small_pos_cash()))
    assert sorted(out.columns) == sorted(POS_CASH_FEATURES)
    one = out.loc[1]
    assert one["POS_LOAN_COUNT"] == 2
    assert one["POS_MONTHS_COUNT"] == 5
    assert one["POS_DPD_MAX"] == 5
    assert one["POS_DPD_MEAN"] == pytest.approx(1.0)
    assert one["POS_LATE_SHARE"] == pytest.approx(0.2)       # 1 of 5 months
    assert one["POS_COMPLETED_SHARE"] == pytest.approx(0.5)  # 1 of 2 loans
    assert out.loc[2, "POS_COMPLETED_SHARE"] == 0
