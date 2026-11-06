"""Unit tests for YOUR feature functions, on small hand-made tables.

A unit test checks one function on an input small enough to work out the right answer by
hand. They need no data and no model, so they run everywhere, including in CI (Wednesday).

The Week 2 tests checked that each function gives the right numbers. These test
**properties** your functions must keep whatever the data: one applicant's history never
leaks into another's, row order does not matter, nobody is dropped, nothing is infinite.

Write each TODO test: build a small table (or reuse one from tests/week2_fixtures.py), call
your function, and assert. Replace the ``pytest.fail`` line. Keep the test's name.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (for you to use)
import pandas as pd
import pytest

from src.features.bureau import aggregate_bureau
from src.features.join import join_features  # noqa: F401
from src.features.previous import aggregate_previous  # noqa: F401
from src.features.ratios import add_ratio_features  # noqa: F401
from tests.week2_fixtures import (  # noqa: F401  (hand-made tables you may reuse)
    small_application,
    small_application_full,
    small_bureau,
    small_previous,
)


# PROVIDED: the worked example. One applicant's credits never change another's features.
def test_bureau_features_depend_only_on_own_credits() -> None:
    bureau = small_bureau()
    before = aggregate_bureau(bureau).set_index("SK_ID_CURR")

    # Give applicant 2 an extra active credit. Applicant 1's row must not move.
    extra = pd.DataFrame({
        "SK_ID_CURR": [2], "SK_ID_BUREAU": [22], "CREDIT_ACTIVE": ["Active"],
        "AMT_CREDIT_SUM": [5000.0], "AMT_CREDIT_SUM_DEBT": [0.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0], "CREDIT_DAY_OVERDUE": [0], "DAYS_CREDIT": [-10],
    })
    after = aggregate_bureau(pd.concat([bureau, extra], ignore_index=True)).set_index("SK_ID_CURR")

    pd.testing.assert_series_equal(before.loc[1], after.loc[1])
    assert after.loc[2, "BUREAU_COUNT"] == before.loc[2, "BUREAU_COUNT"] + 1


# TODO(student)
def test_previous_features_ignore_row_order() -> None:
    """Shuffling the rows of previous_application gives exactly the same features.

    Data files come in any order, so your aggregation must not depend on it (remember the
    tie-break on DAYS_DECISION from Week 2).

    Hints:
        - ``shuffled = prev.sample(frac=1, random_state=0)``.
        - Compare both results sorted by SK_ID_CURR with ``pd.testing.assert_frame_equal``
          (``check_like=True`` ignores column order).
    """

    prev = small_previous()

    original = aggregate_previous(prev).sort_values("SK_ID_CURR").reset_index(drop=True)

    shuffled = prev.sample(frac=1, random_state=0)
    shuffled_result = (
        aggregate_previous(shuffled)
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(original, shuffled_result, check_like=True)

    # pytest.fail("TODO(student): write this test")


# TODO(student)
def test_join_keeps_applicants_without_history() -> None:
    """An applicant with no bureau credits stays in the table, with count 0 and no average.

    Join ``aggregate_bureau(small_bureau())`` onto ``small_application()``. Check that the
    joined table has exactly as many rows as the application table, and that an applicant
    who has no bureau rows gets BUREAU_COUNT 0 and a missing (NaN) BUREAU_AMT_CREDIT_SUM_MEAN.

    Hints:
        - Find an id in ``small_application()["SK_ID_CURR"]`` that is not in
          ``small_bureau()["SK_ID_CURR"]``.
        - ``join_features(app, {"bureau": aggregates})``.
    """

    app = small_application()
    bureau = small_bureau()
    aggregates = aggregate_bureau(bureau)

    joined = join_features(app, {"bureau": aggregates})

    applicant_without_history = app.loc[
        ~app["SK_ID_CURR"].isin(bureau["SK_ID_CURR"]),
        "SK_ID_CURR"
    ].iloc[0]

    assert len(joined) == len(app)

    applicant = joined.loc[
        joined["SK_ID_CURR"] == applicant_without_history
    ].iloc[0]

    assert applicant["BUREAU_COUNT"] == 0
    assert pd.isna(applicant["BUREAU_AMT_CREDIT_SUM_MEAN"])

    # pytest.fail("TODO(student): write this test")


# TODO(student)
def test_join_refuses_a_table_that_was_not_aggregated() -> None:
    """Joining a raw child table (many rows per applicant) raises a ValueError.

    This is the guard that stops rows from multiplying (Week 2: 127,657 rows for 5,000
    applicants, and nothing crashed).

    Hints:
        - ``with pytest.raises(ValueError):`` around ``join_features(app, {"bureau":
          small_bureau()})``.
    """

    app = small_application()
    bureau = small_bureau()

    with pytest.raises(ValueError):
        join_features(app, {"bureau": bureau})

    # pytest.fail("TODO(student): write this test")


# TODO(student)
def test_ratios_are_never_infinite() -> None:
    """An income or credit of 0 gives a missing ratio, never infinity.

    Take ``small_application_full()``, set AMT_INCOME_TOTAL to 0 for one applicant and
    AMT_CREDIT to 0 for another, and check that no ratio column contains ``inf``, and that
    the zero-income applicant's CREDIT_INCOME_RATIO is NaN.

    Hints:
        - ``np.isinf(out[cols].to_numpy(dtype=float)).any()`` is True if any value is inf.
    """

    app = small_application_full().copy()

    first_id = app.iloc[0]["SK_ID_CURR"]
    second_id = app.iloc[1]["SK_ID_CURR"]

    app.loc[app["SK_ID_CURR"] == first_id, "AMT_INCOME_TOTAL"] = 0
    app.loc[app["SK_ID_CURR"] == second_id, "AMT_CREDIT"] = 0

    out = add_ratio_features(app)

    cols = [col for col in out.columns if col.endswith("_RATIO")]

    assert len(cols) > 0
    assert not np.isinf(out[cols].to_numpy(dtype=float)).any()

    zero_income = out.loc[out["SK_ID_CURR"] == first_id].iloc[0]

    assert pd.isna(zero_income["CREDIT_INCOME_RATIO"])

    # pytest.fail("TODO(student): write this test")
