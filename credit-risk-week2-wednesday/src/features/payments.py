"""Repayment behavior at Home Credit: installment payments and credit card months.

installments_payments has one row per payment made (or due) on an earlier loan: when it was
due (DAYS_INSTALMENT), when it was paid (DAYS_ENTRY_PAYMENT), what was due (AMT_INSTALMENT),
and what was paid (AMT_PAYMENT). credit_card_balance has one row per card per month. Both
carry SK_ID_CURR, so each groups by applicant directly.

These tables answer the lender's favorite question: does this person pay on time? Follow the
pattern of ``aggregate_bureau`` in src/features/bureau.py.
"""

from __future__ import annotations

import pandas as pd

from src.features.common import (  # noqa: F401  (helpers for you to use)
    count_where,
    require_columns,
    safe_divide,
    share_where,
)

# PROVIDED
INSTALLMENTS_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
    "AMT_PAYMENT",
]
INSTALLMENTS_FEATURES = [
    "INST_COUNT", "INST_LATE_SHARE", "INST_DAYS_LATE_MAX", "INST_UNDERPAID_SHARE",
    "INST_PAYMENT_DIFF_MEAN",
]
CREDIT_CARD_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_CURRENT", "SK_DPD",
]
CREDIT_CARD_FEATURES = [
    "CC_MONTHS_COUNT", "CC_BALANCE_MEAN", "CC_UTILIZATION_MEAN", "CC_UTILIZATION_MAX",
    "CC_DRAWINGS_MEAN", "CC_DPD_MAX",
]


# TODO(student)
def aggregate_installments(
    installments: pd.DataFrame, id_col: str = "SK_ID_CURR"
) -> pd.DataFrame:
    """Summarize each applicant's payment history on earlier loans into one row.

    Args:
        installments: The installments_payments table, one row per payment.
        id_col: The applicant id column.

    Returns:
        One row per applicant that appears in ``installments``, with ``id_col`` and exactly
        these columns (``INSTALLMENTS_FEATURES``):

        - INST_COUNT: payment rows
        - INST_LATE_SHARE: share of payments made after the due date
          (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)
        - INST_DAYS_LATE_MAX: the most days any payment was late; 0 if never late
        - INST_UNDERPAID_SHARE: share of payments smaller than the amount due
          (AMT_PAYMENT < AMT_INSTALMENT)
        - INST_PAYMENT_DIFF_MEAN: average of AMT_PAYMENT - AMT_INSTALMENT (negative means the
          applicant tends to pay less than due)

        Rows with no payment recorded (DAYS_ENTRY_PAYMENT or AMT_PAYMENT missing) are left out
        of the two shares, the late maximum, and the average, but still count in INST_COUNT.

    Hints:
        - Both DAYS_* columns count days before this application (negative), so "later" means
          a larger number.
        - A comparison with NaN gives False, which would quietly count a missing payment as
          "on time". Make the missing rows NaN before taking a share, so they drop out.
        - ``share_where`` uses every row as the denominator. Here you need a different one.
    """
    raise NotImplementedError("TODO(student): aggregate installments_payments per applicant")


# TODO(student)
def aggregate_credit_card(
    credit_card: pd.DataFrame, id_col: str = "SK_ID_CURR"
) -> pd.DataFrame:
    """Summarize each applicant's monthly credit card history into one row.

    Args:
        credit_card: The credit_card_balance table, one row per card (SK_ID_PREV) per month.
        id_col: The applicant id column.

    Returns:
        One row per applicant that appears in ``credit_card``, with ``id_col`` and exactly
        these columns (``CREDIT_CARD_FEATURES``):

        - CC_MONTHS_COUNT: card-month rows
        - CC_BALANCE_MEAN: average AMT_BALANCE
        - CC_UTILIZATION_MEAN / _MAX: average and largest AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
          over the months whose limit is above 0; NaN if no month has a limit
        - CC_DRAWINGS_MEAN: average AMT_DRAWINGS_CURRENT (money taken from the card that month)
        - CC_DPD_MAX: the largest SK_DPD (days past due) in any month

    Hints:
        - In our sample about 1 in 5 card months has a limit of 0. ``safe_divide`` gives NaN
          there instead of inf, and pandas' mean and max skip NaN.
        - A balance can be slightly negative (the customer overpaid). Keep it.
    """
    raise NotImplementedError("TODO(student): aggregate credit_card_balance per applicant")
