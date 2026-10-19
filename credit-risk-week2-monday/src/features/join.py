"""Joining per-applicant feature tables onto the application table.

The application table has exactly one row per applicant, and so must the result. A join that
silently adds rows (because a child table was not aggregated) or drops rows (because of an
inner join) still runs without an error, and the model then trains on the wrong data. The
functions here make that mistake fail loudly instead.
"""

from __future__ import annotations

import pandas as pd

# PROVIDED: a feature whose name ends with this is a count, so "no records" means 0.
COUNT_SUFFIX = "_COUNT"


# TODO(student)
def assert_one_row_per_applicant(
    df: pd.DataFrame, expected_ids: pd.Series, id_col: str = "SK_ID_CURR"
) -> None:
    """Raise a ValueError unless ``df`` has exactly one row for each id in ``expected_ids``.

    Args:
        df: The table to check (for example the joined feature table).
        expected_ids: The applicant ids that must each appear exactly once.
        id_col: The applicant id column.

    Raises:
        ValueError: if the row count differs, any id appears twice, or the set of ids is not
            exactly ``expected_ids``. The message must say which problem it found and give
            the numbers (for example "24,395 rows for 5,000 applicants").

    Hints:
        - Equal row counts are not enough on their own: one duplicated id plus one missing id
          keeps the count the same.
    """
    raise NotImplementedError("TODO(student): check one row per applicant")


# TODO(student)
def join_features(
    app: pd.DataFrame, feature_tables: dict[str, pd.DataFrame], id_col: str = "SK_ID_CURR"
) -> pd.DataFrame:
    """Left-join each per-applicant feature table onto the application table.

    Args:
        app: The application table, one row per applicant.
        feature_tables: Table name -> aggregated table (the output of an ``aggregate_*``
            function), for example ``{"bureau": aggregate_bureau(bureau)}``.
        id_col: The applicant id column.

    Returns:
        A new DataFrame with the same rows in the same order as ``app``, every column of
        ``app`` unchanged, plus every feature column. For applicants missing from a feature
        table, columns ending in ``_COUNT`` are 0 and every other new column is NaN.

    Raises:
        ValueError: before joining, if a feature table has the same id twice (name the table:
            it was probably not aggregated), or if a feature column name is already taken.
            After joining, through ``assert_one_row_per_applicant``.

    Hints:
        - Which ``how=`` keeps every applicant, including the ones with no history at all?
        - ``pd.merge`` has a ``validate=`` argument that can raise on duplicate keys for you.
        - Why 0 for counts but NaN for averages? Be ready to explain it at the checkpoint.
    """
    raise NotImplementedError("TODO(student): left-join the feature tables")
