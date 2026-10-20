"""Credit Bureau features: what other lenders report about the applicant.

bureau has one row per credit the applicant holds (or held) at another lender, so an applicant
can have 0, 1, or 30 rows. bureau_balance has one row per credit per month and links to bureau
through SK_ID_BUREAU only: it has no SK_ID_CURR column.

The model needs one row per applicant, so each function here returns exactly one row per
applicant that has records. ``aggregate_bureau`` is the worked example: read it before you
write your own.
"""

from __future__ import annotations

import pandas as pd

from src.features.common import count_where, require_columns

# PROVIDED
BUREAU_COLUMNS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
]
BUREAU_BALANCE_COLUMNS = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
BUREAU_BALANCE_FEATURES = [
    "BB_CREDIT_COUNT", "BB_MONTHS_COUNT", "BB_LATE_MONTHS_COUNT", "BB_LATE_SHARE",
    "BB_WORST_STATUS",
]


# PROVIDED: the worked example. Every other aggregation follows this pattern.
def aggregate_bureau(bureau: pd.DataFrame, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Summarize each applicant's credits at other lenders into one row.

    Args:
        bureau: The bureau table, one row per credit.
        id_col: The applicant id column.

    Returns:
        One row per applicant that appears in ``bureau``, with ``id_col`` and these columns:

        - BUREAU_COUNT: credits reported by other lenders
        - BUREAU_ACTIVE_COUNT / BUREAU_CLOSED_COUNT: credits still open / already closed
        - BUREAU_AMT_CREDIT_SUM_MEAN / _MAX: average and largest credit amount
        - BUREAU_AMT_DEBT_MEAN: average debt still owed (missing values ignored)
        - BUREAU_AMT_OVERDUE_SUM: total amount overdue right now
        - BUREAU_DAYS_OVERDUE_MAX: the worst current number of days overdue
        - BUREAU_DAYS_CREDIT_MAX: days before this application that the newest credit opened
          (closest to 0 = most recent)

        Applicants with no bureau records are not in the output. ``join_features`` gives
        them their row, with 0 for the counts and NaN for everything else.
    """
    require_columns(bureau, BUREAU_COLUMNS, "bureau")

    # Step 1: plain summaries with named aggregation, NEW_NAME=(input column, function).
    # The prefix BUREAU_ says which table a feature came from, and names ending in _COUNT
    # tell join_features to fill 0 for applicants without records.
    out = bureau.groupby(id_col).agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_AMT_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
    )

    # Step 2: counts of a condition. Build a True/False column, then count the Trues per
    # applicant. The results are indexed by applicant id, so they line up with `out`.
    keys = bureau[id_col]
    out["BUREAU_ACTIVE_COUNT"] = count_where(bureau["CREDIT_ACTIVE"].eq("Active"), keys)
    out["BUREAU_CLOSED_COUNT"] = count_where(bureau["CREDIT_ACTIVE"].eq("Closed"), keys)

    # Step 3: the id goes back to being a normal column, one row per applicant.
    return out.reset_index()


# TODO(student)  (carry-over: finish before the next class)
def aggregate_bureau_balance(
    bureau_balance: pd.DataFrame, bureau: pd.DataFrame, id_col: str = "SK_ID_CURR"
) -> pd.DataFrame:
    """Summarize the monthly status history of each applicant's bureau credits.

    bureau_balance links to applicants only through bureau, so this works in two stages:
    first one row per bureau credit (SK_ID_BUREAU), then one row per applicant.

    STATUS is text: "0" means no days past due that month, "1" to "5" mean increasingly late
    (1-30 days up to 120+ days or written off), "C" means closed, and "X" means unknown.

    Args:
        bureau_balance: One row per bureau credit per month.
        bureau: The bureau table, used only to find which applicant owns each credit.
        id_col: The applicant id column.

    Returns:
        One row per applicant with at least one credit in ``bureau_balance``, with ``id_col``
        and exactly these columns (``BUREAU_BALANCE_FEATURES``):

        - BB_CREDIT_COUNT: the applicant's bureau credits that have monthly history
        - BB_MONTHS_COUNT: monthly records across those credits
        - BB_LATE_MONTHS_COUNT: months with STATUS "1" to "5"
        - BB_LATE_SHARE: BB_LATE_MONTHS_COUNT / BB_MONTHS_COUNT
        - BB_WORST_STATUS: the highest numeric STATUS (0 to 5) in any month, as a number.
          "C" and "X" do not count; NaN if the applicant has only "C" and "X" months.

    Hints:
        - Call ``require_columns`` on both tables first, like the worked example.
        - ``pd.to_numeric(..., errors="coerce")`` turns "C" and "X" into NaN.
        - Stage 1: group by SK_ID_BUREAU. Stage 2: bring in SK_ID_CURR from bureau with a
          merge, then group by the applicant.
        - Pass ``validate="many_to_one"`` to that merge, so pandas raises if bureau ever
          lists the same credit twice instead of silently duplicating rows.
        - Compute BB_LATE_SHARE from the applicant's totals, not as the mean of each credit's
          share: a credit with 2 months must not weigh as much as one with 60.
        - Credits in bureau without any monthly history must not be counted.
    """

    # Step 1: Check that both tables contain the required columns.
    require_columns(bureau_balance, BUREAU_BALANCE_COLUMNS, "bureau_balance")
    require_columns(bureau, ["SK_ID_BUREAU", id_col], "bureau")

    # Step 2: Convert STATUS to numbers.
    # Statuses "C" and "X" become NaN and are ignored in numeric calculations.
    status_numeric = pd.to_numeric(
        bureau_balance["STATUS"], errors="coerce"
    )

    # Step 3: Aggregate monthly records for each bureau credit.
    credit_history = bureau_balance.assign(
        NUMERIC_STATUS=status_numeric,
        IS_LATE=bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"])
    ).groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("STATUS", "count"),
        BB_LATE_MONTHS_COUNT=("IS_LATE", "sum"),
        BB_WORST_STATUS=("NUMERIC_STATUS", "max"),
    )

    # Step 4: Attach each bureau credit to its applicant.
    # Each credit should belong to only one applicant.
    credit_history = credit_history.reset_index().merge(
        bureau[["SK_ID_BUREAU", id_col]],
        on="SK_ID_BUREAU",
        how="inner",
        validate="many_to_one",
    )

    # Step 5: Aggregate the credit histories for each applicant.
    applicant_history = credit_history.groupby(id_col).agg(
        BB_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BB_MONTHS_COUNT=("BB_MONTHS_COUNT", "sum"),
        BB_LATE_MONTHS_COUNT=("BB_LATE_MONTHS_COUNT", "sum"),
        BB_WORST_STATUS=("BB_WORST_STATUS", "max"),
    )

    # Step 6: Calculate the overall late-month share for each applicant.
    applicant_history["BB_LATE_SHARE"] = (
        applicant_history["BB_LATE_MONTHS_COUNT"]
        / applicant_history["BB_MONTHS_COUNT"]
    )

    # Step 7: Return exactly the required columns, with applicant ID as a column.
    return applicant_history.reset_index()[
        [id_col] + BUREAU_BALANCE_FEATURES
    ]

    raise NotImplementedError("TODO(student): aggregate bureau_balance in two stages")
