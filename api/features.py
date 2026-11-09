"""Turn one request into the one row of features the model needs.

The model was trained on 164 features. Most of them (bureau credits, earlier applications,
payment history...) are aggregates of many rows in the child tables: no request can carry
them. So the service keeps them ready in a **feature store**: one precomputed row per
applicant, keyed by SK_ID_CURR, built from the same Week 2 pipeline the model was trained on.
A request then only needs the applicant's id, plus any amounts that changed since.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas import PredictRequest

# The amounts a request may update, and the ratio features computed from them.
OVERRIDABLE = ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY")
RATIOS = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
    "ANNUITY_CREDIT_RATIO": ("AMT_ANNUITY", "AMT_CREDIT"),
}


class UnknownApplicant(LookupError):
    """The requested SK_ID_CURR is not in the feature store."""


# PROVIDED
def recompute_ratios(row: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute the four ratio features from the row's amounts, exactly as in Week 2.

    A zero or missing denominator gives NaN, never infinity.
    """
    for ratio, (top, bottom) in RATIOS.items():
        if top in row.columns and bottom in row.columns:
            denominator = row[bottom].astype(float)
            row[ratio] = row[top].astype(float) / denominator.where(denominator != 0)
    return row.replace([np.inf, -np.inf], np.nan)


# TODO(student)
def build_features(request: PredictRequest, store: pd.DataFrame,
                   features: list[str]) -> pd.DataFrame:
    """The model's input for this request: one row, with exactly the columns ``features``.

    1. Look up the applicant's row in ``store`` (indexed by SK_ID_CURR). If the id is not
       there, raise ``UnknownApplicant`` with a message that names the id.
    2. Take a COPY of that row as a one-row DataFrame. Never change ``store`` itself: it is
       shared by every request the service will ever answer.
    3. For every field in OVERRIDABLE that the request sets (not None), replace the value.
       If any was replaced, recompute the ratio features with ``recompute_ratios``.
    4. Return the row with its columns in the order of ``features``.

    Args:
        request: The validated request.
        store: The feature store, indexed by SK_ID_CURR.
        features: The model's input columns, in training order.

    Returns:
        A one-row DataFrame.

    Hints:
        - ``store.loc[[request.SK_ID_CURR]]`` (double brackets) gives a one-row DataFrame;
          check ``request.SK_ID_CURR in store.index`` first.
        - ``getattr(request, name)`` reads a field by its name.
        - Why recompute the ratios? Remember Week 4's ``change_income``: a new income with
          the old CREDIT_INCOME_RATIO describes an applicant who cannot exist.
    """
    raise NotImplementedError("TODO(student): look up the applicant and apply the overrides")
