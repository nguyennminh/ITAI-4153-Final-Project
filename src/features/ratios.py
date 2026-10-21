"""Ratio features from the application table.

A loan of 500,000 means something very different to someone earning 50,000 a year than to
someone earning 1,000,000. Ratios put amounts in proportion to the applicant's situation,
which is how a loan officer reads them.
"""

from __future__ import annotations

import pandas as pd

from src.features.common import require_columns, safe_divide  # noqa: F401  (for you to use)

# PROVIDED
RATIO_INPUTS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
RATIO_FEATURES = [
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO", "ANNUITY_CREDIT_RATIO",
]


# TODO(student)
def add_ratio_features(app: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``app`` with four ratio columns added. The input is not changed.

    - CREDIT_INCOME_RATIO: AMT_CREDIT / AMT_INCOME_TOTAL (loan size in years of income)
    - ANNUITY_INCOME_RATIO: AMT_ANNUITY / AMT_INCOME_TOTAL (share of income spent on the
      yearly repayment)
    - CREDIT_GOODS_RATIO: AMT_CREDIT / AMT_GOODS_PRICE (above 1 means borrowing more than the
      price of what is being bought)
    - ANNUITY_CREDIT_RATIO: AMT_ANNUITY / AMT_CREDIT (yearly repayment as a share of the loan;
      a rough inverse of the loan's length)

    A zero or missing denominator gives NaN, never inf. Every other column is unchanged.

    Hints:
        - ``require_columns(app, RATIO_INPUTS, "application_train")`` first.
        - ``safe_divide`` already handles the zero and missing cases.
    """

    require_columns(app, RATIO_INPUTS, "application_train")

    result = app.copy()

    result["CREDIT_INCOME_RATIO"] = safe_divide(
        result["AMT_CREDIT"],
        result["AMT_INCOME_TOTAL"],
    )

    result["ANNUITY_INCOME_RATIO"] = safe_divide(
        result["AMT_ANNUITY"],
        result["AMT_INCOME_TOTAL"],
    )

    result["CREDIT_GOODS_RATIO"] = safe_divide(
        result["AMT_CREDIT"],
        result["AMT_GOODS_PRICE"],
    )

    result["ANNUITY_CREDIT_RATIO"] = safe_divide(
        result["AMT_ANNUITY"],
        result["AMT_CREDIT"],
    )

    return result

    raise NotImplementedError("TODO(student): add the four ratio features")
