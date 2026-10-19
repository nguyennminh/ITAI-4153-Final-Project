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


# TODO(student)
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
    raise NotImplementedError("TODO(student): aggregate previous_application per applicant")


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
    raise NotImplementedError("TODO(student): aggregate POS_CASH_balance per applicant")
