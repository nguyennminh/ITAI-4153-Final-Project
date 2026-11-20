"""Week 6 Wednesday: plain-language reasons, and the per-group fairness table.

# DO NOT EDIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import explain_applicant
from src.fairness import add_intervals, group_table, wilson

FEATURES = ["EXT_SOURCE_3", "DAYS_BIRTH", "AMT_CREDIT", "CODE_GENDER", "OWN_CAR_AGE"]
VALUES = pd.Series({"EXT_SOURCE_3": 0.12, "DAYS_BIRTH": -9500, "AMT_CREDIT": 900_000.0,
                    "CODE_GENDER": "M", "OWN_CAR_AGE": np.nan})
TYPICAL = pd.Series({"EXT_SOURCE_3": 0.54, "DAYS_BIRTH": -15800, "AMT_CREDIT": 510_000.0,
                     "CODE_GENDER": "F", "OWN_CAR_AGE": 9.0})


# DO NOT EDIT
def test_reasons_are_the_largest_positive_contributions() -> None:
    contributions = pd.Series({"EXT_SOURCE_3": 0.9, "DAYS_BIRTH": 0.3, "AMT_CREDIT": 0.5,
                               "CODE_GENDER": 0.1, "OWN_CAR_AGE": -0.7})
    reasons = explain_applicant(contributions, VALUES, TYPICAL)
    assert isinstance(reasons, list) and len(reasons) == 3
    assert "external credit score 3" in reasons[0] and "0.12" in reasons[0]
    assert "the loan amount" in reasons[1] and "900,000" in reasons[1]
    assert "age" in reasons[2] and "26.0 years" in reasons[2]
    assert "0.90" in reasons[0], "say how much the feature raised the score"


# DO NOT EDIT
def test_reasons_never_include_what_lowered_the_risk() -> None:
    contributions = pd.Series({"EXT_SOURCE_3": 0.4, "DAYS_BIRTH": -0.2, "AMT_CREDIT": -0.1,
                               "CODE_GENDER": 0.0, "OWN_CAR_AGE": -0.3})
    reasons = explain_applicant(contributions, VALUES, TYPICAL)
    assert len(reasons) == 1 and "external credit score 3" in reasons[0]


# DO NOT EDIT
def test_reasons_respect_n_and_missing_values() -> None:
    contributions = pd.Series({"EXT_SOURCE_3": 0.1, "DAYS_BIRTH": 0.2, "AMT_CREDIT": 0.3,
                               "CODE_GENDER": 0.4, "OWN_CAR_AGE": 0.5})
    reasons = explain_applicant(contributions, VALUES, TYPICAL, n=2)
    assert len(reasons) == 2
    assert "missing" in reasons[0] and "gender is M" in reasons[1]


def _frame() -> pd.DataFrame:
    # Group A: 10 applicants, 4 defaulters (3 rejected), 6 good (1 rejected).
    # Group B: 5 applicants, no defaulters, 5 good (0 rejected).
    return pd.DataFrame({
        "g": ["A"] * 10 + ["B"] * 5,
        "y": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [0] * 5,
        "p": [0.9, 0.8, 0.7, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.05] * 5,
        "reject": [True, True, True, False, True, False, False, False, False, False]
        + [False] * 5,
    })


# DO NOT EDIT
def test_group_table_counts_and_rates() -> None:
    table = group_table(_frame(), "g")
    assert list(table["group"]) == ["A", "B"]
    a = table.iloc[0]
    assert (a["n"], a["n_default"], a["n_good"]) == (10, 4, 6)
    assert a["approval_rate"] == pytest.approx(0.6)
    assert a["default_rate"] == pytest.approx(0.4)
    assert a["mean_score"] == pytest.approx(0.36)
    assert a["tpr"] == pytest.approx(0.75)
    assert a["fpr"] == pytest.approx(1 / 6)


# DO NOT EDIT
def test_group_table_without_defaulters() -> None:
    b = group_table(_frame(), "g").iloc[1]
    assert b["approval_rate"] == pytest.approx(1.0)
    assert np.isnan(b["tpr"]), "no defaulters: the TPR is unknown (NaN), not 0"
    assert b["fpr"] == pytest.approx(0.0)


# DO NOT EDIT
def test_intervals_widen_for_small_groups() -> None:
    small, large = wilson(6, 10), wilson(600, 1000)
    assert small[0] < 0.6 < small[1] and large[0] < 0.6 < large[1]
    assert (small[1] - small[0]) > 3 * (large[1] - large[0])
    table = add_intervals(group_table(_frame(), "g"))
    assert "too small" in table.iloc[1]["note"]
