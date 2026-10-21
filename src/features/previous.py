"""Home Credit's own history with the applicant: earlier applications and their loans.

previous_application has one row per earlier application at Home Credit, approved or not.
POS_CASH_balance has one row per month of each earlier point-of-sale or cash loan. Both carry
SK_ID_CURR, so each can be grouped by applicant directly.

Follow the pattern of ``aggregate_bureau`` in src/features/bureau.py.
"""

from __future__ import annotations

import pandas as pd

from src.features.common import (  # noqa: F401  (helpers for you to use)
    SENTINEL,
    count_where,
    replace_sentinel,
    require_columns,
    share_where,
)

# PROVIDED
PREVIOUS_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
    "DAYS_DECISION", "DAYS_TERMINATION",
]
# Columns of previous_application that use SENTINEL for "no date".
PREVIOUS_DAY_COLUMNS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]
PREVIOUS_FEATURES = [
    "PREV_COUNT", "PREV_APPROVED_RATE", "PREV_REFUSED_RATE", "PREV_AMT_APPLICATION_MEAN",
    "PREV_AMT_CREDIT_MEAN", "PREV_DAYS_DECISION_MAX", "PREV_DAYS_TERMINATION_MEAN",
    "PREV_LAST_STATUS",
]
POS_CASH_COLUMNS = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "SK_DPD"]
POS_CASH_FEATURES = [
    "POS_LOAN_COUNT", "POS_MONTHS_COUNT", "POS_DPD_MAX", "POS_DPD_MEAN", "POS_LATE_SHARE",
    "POS_COMPLETED_SHARE",
]


def aggregate_previous(previous: pd.DataFrame, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Summarize each applicant's earlier applications at Home Credit into one row.

    Args:
        previous: The previous_application table, one row per earlier application.
        id_col: The applicant id column.

    Returns:
        One row per applicant that appears in ``previous``, with ``id_col`` and at least
        these columns (``PREVIOUS_FEATURES``). You may add more, with the PREV_ prefix.

        - PREV_COUNT: earlier applications
        - PREV_APPROVED_RATE: share of them with NAME_CONTRACT_STATUS "Approved" (0 to 1)
        - PREV_REFUSED_RATE: share with NAME_CONTRACT_STATUS "Refused" (0 to 1)
        - PREV_AMT_APPLICATION_MEAN: average amount the applicant asked for
        - PREV_AMT_CREDIT_MEAN: average amount Home Credit offered
        - PREV_DAYS_DECISION_MAX: days before this application of the most recent decision
        - PREV_DAYS_TERMINATION_MEAN: average DAYS_TERMINATION, ignoring "no date" values
        - PREV_LAST_STATUS: NAME_CONTRACT_STATUS of the most recent application, as text.
          Most recent means the largest DAYS_DECISION; if two applications share it, take
          the one with the larger SK_ID_PREV.

    Hints:
        - Start with ``require_columns``, then ``replace_sentinel`` on PREVIOUS_DAY_COLUMNS.
          A single 365243 left in DAYS_TERMINATION drags an average up by hundreds of days.
        - There are four statuses (Approved, Refused, Canceled, Unused offer), so the two
          rates do not add up to 1. ``share_where`` gives a rate per applicant.
        - For PREV_LAST_STATUS: sort, then keep one row per applicant. In the real sample
          about 4,700 applications tie on DAYS_DECISION with another application of the
          same applicant, so the tie-break matters if you want the same answer every run.
        - Do not change the input DataFrame.
    """
    required = [id_col if col == "SK_ID_CURR" else col for col in PREVIOUS_COLUMNS]
    require_columns(previous, required, "previous_application")
    cleaned = replace_sentinel(previous[required], PREVIOUS_DAY_COLUMNS)
    out = cleaned.groupby(id_col).agg(
        PREV_COUNT=("SK_ID_PREV", "size"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_DAYS_TERMINATION_MEAN=("DAYS_TERMINATION", "mean"),
    )
    keys = cleaned[id_col]
    status = cleaned["NAME_CONTRACT_STATUS"]
    out["PREV_APPROVED_RATE"] = share_where(status.eq("Approved"), keys)
    out["PREV_REFUSED_RATE"] = share_where(status.eq("Refused"), keys)
    latest = cleaned.sort_values(
        ["DAYS_DECISION", "SK_ID_PREV"], na_position="first"
    ).drop_duplicates(id_col, keep="last")
    out["PREV_LAST_STATUS"] = latest.set_index(id_col)["NAME_CONTRACT_STATUS"]
    return out.reset_index()[[id_col, *PREVIOUS_FEATURES]]


# TODO(student)  (carry-over: finish before the next class)
def aggregate_pos_cash(pos_cash: pd.DataFrame, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Summarize the monthly history of each applicant's earlier POS and cash loans.

    Args:
        pos_cash: The POS_CASH_balance table, one row per earlier loan (SK_ID_PREV) per month.
        id_col: The applicant id column.

    Returns:
        One row per applicant that appears in ``pos_cash``, with ``id_col`` and exactly these
        columns (``POS_CASH_FEATURES``):

        - POS_LOAN_COUNT: distinct earlier loans (SK_ID_PREV)
        - POS_MONTHS_COUNT: monthly records across those loans
        - POS_DPD_MAX / POS_DPD_MEAN: largest and average SK_DPD (days past due) in a month
        - POS_LATE_SHARE: share of months with SK_DPD above 0
        - POS_COMPLETED_SHARE: share of the applicant's loans with at least one month whose
          NAME_CONTRACT_STATUS is "Completed"

    Hints:
        - Most columns are a single groupby; POS_COMPLETED_SHARE is per loan first, then per
          applicant (the same two-stage idea as bureau_balance).
        - Watch the denominators: POS_LATE_SHARE is over months, POS_COMPLETED_SHARE over loans.
    """

    # Step 1: Check that all required columns exist.
    required = [
        id_col if col == "SK_ID_CURR" else col
        for col in POS_CASH_COLUMNS
    ]
    require_columns(pos_cash, required, "POS_CASH_balance")

    # Step 2: Aggregate monthly records for each applicant.
    out = pos_cash.groupby(id_col).agg(
        POS_MONTHS_COUNT=("SK_ID_PREV", "size"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
    )

    # Step 3: Count the months when the applicant was late.
    keys = pos_cash[id_col]
    late = pos_cash["SK_DPD"].gt(0)
    out["POS_LATE_SHARE"] = share_where(late, keys)

    # Step 4: Summarize each individual loan.
    # A loan is completed if at least one monthly record says "Completed".
    loan_history = pos_cash.groupby(
        [id_col, "SK_ID_PREV"]
    ).agg(
        LOAN_COMPLETED=("NAME_CONTRACT_STATUS",
                       lambda statuses: statuses.eq("Completed").any())
    )

    # Step 5: Count distinct loans and calculate the completed-loan share.
    applicant_loans = loan_history.groupby(level=0).agg(
        POS_LOAN_COUNT=("LOAN_COMPLETED", "size"),
        COMPLETED_LOAN_COUNT=("LOAN_COMPLETED", "sum"),
    )

    applicant_loans["POS_COMPLETED_SHARE"] = (
        applicant_loans["COMPLETED_LOAN_COUNT"]
        / applicant_loans["POS_LOAN_COUNT"]
    )

    # Step 6: Combine the loan counts and monthly statistics.
    out = out.join(
        applicant_loans[
            ["POS_LOAN_COUNT", "POS_COMPLETED_SHARE"]
        ]
    )

    # Step 7: Return one row per applicant with exactly the required columns.
    return out.reset_index()[[id_col, *POS_CASH_FEATURES]]

    raise NotImplementedError("TODO(student): aggregate POS_CASH_balance per applicant")
