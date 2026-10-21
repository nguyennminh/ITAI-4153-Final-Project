"""Data validation: check the raw tables before building features, and the feature table after.

A pipeline that crashes with ``KeyError`` deep inside an aggregation has failed, but not
*loudly*: whoever runs it has to read code to learn what is wrong with the data. Worse, some
bad data does not crash at all (a status spelled "active" instead of "Active" is simply never
counted). Validation turns both into one plain-language list of problems, and stops the
pipeline before a bad feature table is ever written.

We use pandera: a schema lists the columns a table must have and the rules each must pass.
``validate(..., lazy=True)`` collects every failure instead of stopping at the first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.features.bureau import BUREAU_BALANCE_COLUMNS, BUREAU_COLUMNS  # noqa: F401
from src.features.payments import CREDIT_CARD_COLUMNS, INSTALLMENTS_COLUMNS  # noqa: F401
from src.features.previous import POS_CASH_COLUMNS, PREVIOUS_COLUMNS  # noqa: F401

# PROVIDED
DAYS_EMPLOYED_SENTINEL = 365243
BUREAU_STATUSES = ["Active", "Closed", "Sold", "Bad debt"]
BUREAU_BALANCE_STATUSES = ["0", "1", "2", "3", "4", "5", "C", "X"]
PREVIOUS_STATUSES = ["Approved", "Refused", "Canceled", "Unused offer"]


# PROVIDED
class DataValidationError(ValueError):
    """Raised when data breaks its schema. ``problems`` holds one readable line per issue."""

    def __init__(self, stage: str, problems: list[str]):
        self.stage = stage
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{stage} validation failed with {len(problems)} problem(s):\n{lines}")


# PROVIDED
def required_columns(columns: list[str]) -> dict[str, pa.Column]:
    """A schema entry for each column that must exist, with no rule on its values yet."""
    return {col: pa.Column(nullable=True) for col in columns}


# PROVIDED
def max_null_share(limit: float) -> pa.Check:
    """A check that fails when more than ``limit`` (0 to 1) of a column's values are missing.

    Use it with ``nullable=True`` on the column, so missing values are allowed at all.
    """
    return pa.Check(lambda s: s.isna().mean() <= limit, element_wise=False, ignore_na=False,
                    error=f"more than {limit:.0%} missing")


# PROVIDED
def is_finite() -> pa.Check:
    """A check that fails on inf or -inf (missing values are allowed)."""
    return pa.Check(lambda s: ~np.isinf(s.astype(float)), error="is infinite")


# PROVIDED
def describe_failures(failure_cases: pd.DataFrame, table: str) -> list[str]:
    """Turn pandera's failure table into one readable line per (column, rule)."""
    lines = []
    cases = failure_cases.assign(column=failure_cases["column"].fillna(""))
    for (column, check), group in cases.groupby(["column", "check"], sort=False):
        where = f"{table}.{column}" if column else table
        values = group["failure_case"].dropna()
        examples = ", ".join(map(str, values.unique()[:3]))
        if check == "column_in_dataframe":
            lines.append(f"{table}: missing column(s) {', '.join(map(str, values))}")
        elif check == "not_nullable":
            lines.append(f"{where}: {len(group):,} missing value(s), none allowed")
        elif check == "field_uniqueness":
            lines.append(f"{where}: {values.nunique():,} value(s) appear more than once "
                         f"(e.g. {examples})")
        elif group["index"].isna().all():
            lines.append(f"{where}: {check}")
        else:
            lines.append(f"{where}: {len(group):,} row(s) fail '{check}' (e.g. {examples})")
    return lines


# PROVIDED
def run_schema(schema: pa.DataFrameSchema, df: pd.DataFrame, table: str) -> list[str]:
    """Validate ``df`` against ``schema``; return the problems found (empty list if none)."""
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        return describe_failures(err.failure_cases, table)
    return []


# PROVIDED: the worked example. Every other schema follows this pattern.
def bureau_schema() -> pa.DataFrameSchema:
    """Rules for the raw bureau table.

    - every column ``aggregate_bureau`` reads must exist (``required_columns``)
    - the two id columns are never missing, and each bureau credit appears once
    - CREDIT_ACTIVE only takes the values we know, so a renamed status cannot silently
      drop out of BUREAU_ACTIVE_COUNT
    - amounts and days overdue are never negative (missing is allowed)
    """
    columns = required_columns(BUREAU_COLUMNS)
    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False),
        "SK_ID_BUREAU": pa.Column(nullable=False, unique=True),
        "CREDIT_ACTIVE": pa.Column(checks=pa.Check.isin(BUREAU_STATUSES), nullable=False),
        "AMT_CREDIT_SUM": pa.Column(checks=pa.Check.ge(0), nullable=True),
        "AMT_CREDIT_SUM_OVERDUE": pa.Column(checks=pa.Check.ge(0), nullable=True),
        "CREDIT_DAY_OVERDUE": pa.Column(checks=pa.Check.ge(0), nullable=True),
    })
    return pa.DataFrameSchema(columns)


# TODO(student)
def application_schema() -> pa.DataFrameSchema:
    """Rules for the raw application_train table.

    Columns and rules (every column listed must exist):

    - SK_ID_CURR: never missing, unique
    - TARGET: 0 or 1, never missing
    - AMT_INCOME_TOTAL, AMT_CREDIT: above 0, never missing
    - AMT_ANNUITY, AMT_GOODS_PRICE: above 0, may be missing
    - DAYS_BIRTH: below 0 (born before applying), never missing
    - DAYS_EMPLOYED: 0 or below, or exactly DAYS_EMPLOYED_SENTINEL (Week 1's "no job" code),
      never missing

    Other columns may exist and are not checked.

    Hints:
        - Copy the pattern of ``bureau_schema``.
        - ``pa.Check.gt``, ``pa.Check.lt``, ``pa.Check.le``, and ``pa.Check.isin`` cover
          most rules. For DAYS_EMPLOYED write your own:
          ``pa.Check(lambda s: ..., error="...")`` where the lambda returns True/False per row.
    """

    columns = required_columns([
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ])

    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False, unique=True),
        "TARGET": pa.Column(
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
        "AMT_INCOME_TOTAL": pa.Column(
            checks=pa.Check.gt(0),
            nullable=False,
        ),
        "AMT_CREDIT": pa.Column(
            checks=pa.Check.gt(0),
            nullable=False,
        ),
        "AMT_ANNUITY": pa.Column(
            checks=pa.Check.gt(0),
            nullable=True,
        ),
        "AMT_GOODS_PRICE": pa.Column(
            checks=pa.Check.gt(0),
            nullable=True,
        ),
        "DAYS_BIRTH": pa.Column(
            checks=pa.Check.lt(0),
            nullable=False,
        ),
        "DAYS_EMPLOYED": pa.Column(
            checks=pa.Check(
                lambda s: (s <= 0) | (s == DAYS_EMPLOYED_SENTINEL),
                error="must be <= 0 or the DAYS_EMPLOYED_SENTINEL",
            ),
            nullable=False,
        ),
    })

    return pa.DataFrameSchema(columns)


    raise NotImplementedError("TODO(student): write the application_train schema")


# TODO(student)
def child_schemas() -> dict[str, pa.DataFrameSchema]:
    """Rules for the six child tables, as {table name: schema}.

    For every table: each column its aggregation reads must exist (the ``*_COLUMNS`` lists
    imported at the top), and its id columns (SK_ID_CURR, plus SK_ID_BUREAU or SK_ID_PREV)
    are never missing. Then these value rules:

    - bureau: use ``bureau_schema()`` as it is
    - bureau_balance: STATUS in BUREAU_BALANCE_STATUSES (STATUS is text, "0" not 0)
    - previous_application: NAME_CONTRACT_STATUS in PREVIOUS_STATUSES
    - POS_CASH_balance: SK_DPD 0 or above
    - installments_payments: AMT_INSTALMENT and AMT_PAYMENT 0 or above (may be missing)
    - credit_card_balance: AMT_CREDIT_LIMIT_ACTUAL and SK_DPD 0 or above

    Hints:
        - bureau_balance has no SK_ID_CURR; its id is SK_ID_BUREAU.
        - SK_ID_PREV repeats in the monthly tables (one row per month), so it is not unique.
    """

    schemas = {
        "bureau": bureau_schema(),
    }

    # Bureau balance
    columns = required_columns(BUREAU_BALANCE_COLUMNS)
    columns.update({
        "SK_ID_BUREAU": pa.Column(nullable=False),
        "STATUS": pa.Column(
            checks=pa.Check.isin(BUREAU_BALANCE_STATUSES),
            nullable=False,
        ),
    })
    schemas["bureau_balance"] = pa.DataFrameSchema(columns)

    # Previous applications
    columns = required_columns(PREVIOUS_COLUMNS)
    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False),
        "SK_ID_PREV": pa.Column(nullable=False),
        "NAME_CONTRACT_STATUS": pa.Column(
            checks=pa.Check.isin(PREVIOUS_STATUSES),
            nullable=False,
        ),
    })
    schemas["previous_application"] = pa.DataFrameSchema(columns)

    # POS cash balance
    columns = required_columns(POS_CASH_COLUMNS)
    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False),
        "SK_ID_PREV": pa.Column(nullable=False),
        "SK_DPD": pa.Column(
            checks=pa.Check.ge(0),
            nullable=True,
        ),
    })
    schemas["POS_CASH_balance"] = pa.DataFrameSchema(columns)

    # Installment payments
    columns = required_columns(INSTALLMENTS_COLUMNS)
    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False),
        "SK_ID_PREV": pa.Column(nullable=False),
        "AMT_INSTALMENT": pa.Column(
            checks=pa.Check.ge(0),
            nullable=True,
        ),
        "AMT_PAYMENT": pa.Column(
            checks=pa.Check.ge(0),
            nullable=True,
        ),
    })
    schemas["installments_payments"] = pa.DataFrameSchema(columns)

    # Credit card balance
    columns = required_columns(CREDIT_CARD_COLUMNS)
    columns.update({
        "SK_ID_CURR": pa.Column(nullable=False),
        "SK_ID_PREV": pa.Column(nullable=False),
        "AMT_CREDIT_LIMIT_ACTUAL": pa.Column(
            checks=pa.Check.ge(0),
            nullable=True,
        ),
        "SK_DPD": pa.Column(
            checks=pa.Check.ge(0),
            nullable=True,
        ),
    })
    schemas["credit_card_balance"] = pa.DataFrameSchema(columns)

    return schemas


    raise NotImplementedError("TODO(student): write the child table schemas")


# TODO(student)
def feature_schema(
    expected_features: list[str], n_applicants: int, null_limits: dict[str, float]
) -> pa.DataFrameSchema:
    """Rules for the finished feature table (the output of ``build_feature_table``).

    Args:
        expected_features: Every engineered feature column the table must have.
        n_applicants: How many rows the table must have (one per applicant).
        null_limits: Largest allowed share of missing values, by column name, with a
            ``"default"`` entry for every expected feature not listed by name. Columns listed
            by name are checked even if they are not engineered features (for example
            AMT_CREDIT).

    Rules:

    - SK_ID_CURR: never missing, unique
    - every expected feature exists
    - names ending in ``_COUNT``: 0 or above, never missing (the join fills 0)
    - names ending in ``_SHARE`` or ``_RATE``: between 0 and 1 (may be missing)
    - names ending in ``_RATIO``: 0 or above and finite (no inf)
    - every other expected feature, and every column named in ``null_limits``: its missing
      share is at most its limit (``max_null_share``)
    - the whole table: exactly ``n_applicants`` rows (a ``pa.Check`` on the DataFrameSchema
      itself, through its ``checks=`` argument)

    Hints:
        - Build a dict {column: pa.Column(...)} in a loop over ``expected_features``, then
          add the columns that are only in ``null_limits``.
        - A column can have several checks: ``checks=[pa.Check.ge(0), is_finite()]``.
        - Apply the missing-share limit to the suffix columns too (except ``_COUNT``).
    """

    columns = {
        "SK_ID_CURR": pa.Column(
            nullable=False,
            unique=True,
        )
    }

    # Validate every expected feature.
    for feature in expected_features:
        if feature == "SK_ID_CURR":
            continue

        if feature.endswith("_COUNT"):
            checks = [pa.Check.ge(0)]
            nullable = False

        elif feature.endswith("_SHARE") or feature.endswith("_RATE"):
            checks = [
                pa.Check.ge(0),
                pa.Check.le(1),
            ]
            nullable = True

        elif feature.endswith("_RATIO"):
            checks = [
                pa.Check.ge(0),
                is_finite(),
            ]
            nullable = True

        else:
            checks = []
            nullable = True

        # Apply missing-value limits to every feature except COUNT.
        if not feature.endswith("_COUNT"):
            limit = null_limits.get(
                feature,
                null_limits["default"],
            )
            checks.append(max_null_share(limit))

        columns[feature] = pa.Column(
            checks=checks if checks else None,
            nullable=nullable,
        )

    # Also validate columns named explicitly in null_limits,
    # even when they are not engineered features.
    for column, limit in null_limits.items():
        if column == "default" or column in columns:
            continue

        columns[column] = pa.Column(
            checks=max_null_share(limit),
            nullable=True,
        )

    return pa.DataFrameSchema(
        columns,
        checks=pa.Check(
            lambda df: len(df) == n_applicants,
            error=f"expected exactly {n_applicants} rows",
        ),
    )


    raise NotImplementedError("TODO(student): write the feature table schema")


# PROVIDED
def validate_inputs(tables: dict[str, pd.DataFrame]) -> None:
    """Check all seven raw tables; raise one DataValidationError listing every problem."""
    schemas = {"application_train": application_schema(), **child_schemas()}
    problems = []
    for name, schema in schemas.items():
        if name not in tables:
            problems.append(f"{name}: table not found")
            continue
        problems += run_schema(schema, tables[name], name)
    if problems:
        raise DataValidationError("input", problems)


# PROVIDED
def validate_features(
    features: pd.DataFrame, expected_features: list[str], n_applicants: int,
    config: dict[str, Any],
) -> None:
    """Check the feature table; raise a DataValidationError listing every problem."""
    null_limits = config["week2"]["validation"]["max_null_share"]
    schema = feature_schema(expected_features, n_applicants, null_limits)
    problems = run_schema(schema, features, "features")
    if problems:
        raise DataValidationError("feature", problems)
