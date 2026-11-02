"""Data tests: does the real feature table look the way the model assumes?

The Week 2 validation runs once, when the feature table is built. These tests check the
finished table every time the tests run, so a change anywhere upstream (a new sample, a
changed aggregation, a teammate's "small fix") that breaks an assumption is caught.

Each test checks one assumption and says in its message what was wrong. Choose every limit
by LOOKING at the data first (``features["DAYS_BIRTH"].describe()``), and write down why
you chose it: a limit that is too loose catches nothing, one that is too tight fails on
good data.

They are skipped where the data is missing (a fresh clone, or CI).
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (for you to use)
import pandas as pd
import pytest

N_APPLICANTS = 5000


# PROVIDED: the worked example.
def test_one_row_per_applicant(features: pd.DataFrame) -> None:
    assert len(features) == N_APPLICANTS, f"{len(features):,} rows, expected {N_APPLICANTS:,}"
    duplicated = int(features["SK_ID_CURR"].duplicated().sum())
    assert duplicated == 0, f"{duplicated} applicant ids appear more than once"


# TODO(student)
def test_schema(features: pd.DataFrame) -> None:
    """The columns the model and the rest of the project rely on exist, with the right kind.

    Check at least: SK_ID_CURR and TARGET exist; TARGET only holds 0 and 1; AMT_INCOME_TOTAL,
    AMT_CREDIT, DAYS_BIRTH, EXT_SOURCE_2 and the four *_RATIO columns are numeric.

    Hints:
        - ``set(features["TARGET"].unique()) <= {0, 1}``.
        - ``pd.api.types.is_numeric_dtype(features[col])``.
    """

    required_columns = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "DAYS_BIRTH",
        "EXT_SOURCE_2",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "ANNUITY_CREDIT_RATIO",
    ]

    missing_columns = set(required_columns) - set(features.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    assert set(features["TARGET"].dropna().unique()) <= {0, 1}, (
        "TARGET contains values other than 0 and 1"
    )

    numeric_columns = [
        "SK_ID_CURR",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "DAYS_BIRTH",
        "EXT_SOURCE_2",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "ANNUITY_CREDIT_RATIO",
    ]

    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(features[col]), (
            f"{col} must be numeric"
        )

    #pytest.fail("TODO(student): write this test")


# TODO(student)
def test_value_ranges(features: pd.DataFrame) -> None:
    """Values that cannot be true never appear.

    Check at least: applicants are between 18 and 70 years old (DAYS_BIRTH counts days
    before the application, so it is negative: 18 years is about -6,570 days); income is
    above 0; every *_RATE and *_SHARE column is between 0 and 1 (missing values allowed).

    Hints:
        - ``features["DAYS_BIRTH"].between(low, high).all()``.
        - ``[c for c in features.columns if c.endswith(("_RATE", "_SHARE"))]``.
        - ``s.dropna().between(0, 1).all()`` ignores missing values.
    """

    assert features["DAYS_BIRTH"].between(-25550, -6570).all(), (
        "DAYS_BIRTH contains ages outside the 18-70 year range"
    )

    assert (features["AMT_INCOME_TOTAL"] > 0).all(), (
        "AMT_INCOME_TOTAL contains zero or negative income"
    )

    rate_columns = [
        c for c in features.columns
        if c.endswith(("_RATE", "_SHARE"))
    ]

    for col in rate_columns:
        values = features[col].dropna()
        assert values.between(0, 1).all(), (
            f"{col} contains values outside the 0-1 range"
        )

    #pytest.fail("TODO(student): write this test")


# TODO(student)
def test_missing_values(features: pd.DataFrame) -> None:
    """Missing values stay where we expect them, and nowhere else.

    Check at least: AMT_INCOME_TOTAL, AMT_CREDIT, DAYS_BIRTH and TARGET are never missing;
    EXT_SOURCE_2 (the strongest feature) is missing for less than 1% of applicants; no
    column is missing for every applicant.

    Hints:
        - ``features[col].isna().mean()`` is the share missing.
        - Look at ``features.isna().mean().sort_values()`` before you pick limits.
    """

    required_columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "DAYS_BIRTH",
        "TARGET",
    ]

    for col in required_columns:
        assert features[col].notna().all(), (
            f"{col} contains missing values"
        )

    ext_source_missing = features["EXT_SOURCE_2"].isna().mean()

    assert ext_source_missing < 0.01, (
        f"EXT_SOURCE_2 is {ext_source_missing:.2%} missing; expected less than 1%"
    )

    completely_missing = features.columns[features.isna().all()].tolist()

    assert not completely_missing, (
        f"Columns completely missing: {completely_missing}"
    )

    #pytest.fail("TODO(student): write this test")


# TODO(student)
def test_default_rate(features: pd.DataFrame) -> None:
    """The share of defaults is about 8%, as in the full data.

    A rate far from 8% means the sample or the labels changed: every model, threshold, and
    cost would then be wrong. Pick a band (for example 6% to 10%) and explain the choice.
    """

    default_rate = features["TARGET"].mean()

    # A 6%-10% band allows reasonable sampling variation around the expected 8%.
    assert 0.06 <= default_rate <= 0.10, (
        f"Default rate is {default_rate:.2%}; expected between 6% and 10%"
    )

    #pytest.fail("TODO(student): write this test")
