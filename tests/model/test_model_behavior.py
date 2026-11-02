"""Model tests: does the model behave the way a lender needs it to?

Every test calls the model ONLY through ``score`` (that is, ``predict_proba``), so the same
tests work for any model you register: boosting, a tree, LightGBM, anything.

- a minimum-quality test (provided): the model must rank better than a floor;
- invariance tests: changing something that must not matter leaves every score unchanged;
- a directional test: changing something in a known direction moves the risk the right way.

If one of your tests FAILS, do not change the test to make it pass. Find out why, and write
down what you found in docs/test_findings.md: a real failure is a real finding about your
model. Then decide with your team: fix the model, or keep the test and mark it as a known
failure with ``@pytest.mark.xfail(strict=True, reason="see docs/test_findings.md")``.

They are skipped where the data is missing (a fresh clone, or CI).
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (for you to use)
import pytest  # noqa: F401  (for you to use)
from sklearn.metrics import roc_auc_score

from src.model_io import ProductionModel, change_income, score  # noqa: F401

# The floor every reasonable model on this data clears: the Week 1 logistic regression on
# the application table alone scored 0.710, and a model that cannot beat that is broken.
MIN_AUC = 0.70
# The largest change in a probability we treat as "no change" (floating-point noise).
TOLERANCE = 0.001


# PROVIDED: the worked example.
def test_minimum_auc(production: ProductionModel) -> None:
    data = production.data
    auc = roc_auc_score(data.y_valid, score(production.model, data.X_valid))
    assert auc >= MIN_AUC, f"validation AUC {auc:.3f} is below the floor {MIN_AUC}"


# TODO(student)
def test_application_weekday_does_not_change_the_score(production: ProductionModel) -> None:
    """The day of the week someone applied on must not change their risk.

    Set WEEKDAY_APPR_PROCESS_START to the same day (for example "MONDAY") for every
    validation applicant, score again, and check that no score moved by more than TOLERANCE.

    Hints:
        - ``X = production.data.X_valid`` and ``changed = X.copy()``.
        - ``np.abs(score(model, changed) - score(model, X)).max()``.
        - Put the size of the largest change in the assert message.
    """

    model = production.model
    X = production.data.X_valid

    original_scores = score(model, X)

    changed = X.copy()
    changed["WEEKDAY_APPR_PROCESS_START"] = "MONDAY"

    changed_scores = score(model, changed)

    largest_change = np.abs(changed_scores - original_scores).max()

    assert largest_change <= TOLERANCE, (
        f"Changing application weekday changed scores by up to "
        f"{largest_change:.6f}"
    )

    #pytest.fail("TODO(student): write this test")


# TODO(student)
def test_gender_does_not_change_the_score(production: ProductionModel) -> None:
    """Two applicants who differ only in CODE_GENDER must get the same score.

    Lending law treats gender as a protected attribute: the lender's policy is that it must
    not change a decision. Score every validation applicant as "F" and again as "M", and
    check that no score differs by more than 0.01.
    """

    model = production.model
    X = production.data.X_valid

    female = X.copy()
    male = X.copy()

    female["CODE_GENDER"] = "F"
    male["CODE_GENDER"] = "M"

    female_scores = score(model, female)
    male_scores = score(model, male)

    largest_change = np.abs(female_scores - male_scores).max()

    assert largest_change <= 0.01, (
        f"Changing gender changed scores by up to {largest_change:.6f}"
    )

    #pytest.fail("TODO(student): write this test")


# TODO(student)
def test_higher_income_does_not_raise_risk(production: ProductionModel) -> None:
    """With everything else equal, earning 50% more must not make an applicant riskier.

    Use the provided ``change_income(X, 1.5)``: it raises every income by 50% and also
    recomputes the ratio features that depend on income, as real life would. Check that no
    applicant's score goes UP by more than TOLERANCE.

    Hints:
        - ``increase = score(model, change_income(X, 1.5)) - score(model, X)``.
        - Report how many applicants got riskier, and the largest increase, in the message.
    """

    model = production.model
    X = production.data.X_valid

    original_scores = score(model, X)
    increased_scores = score(model, change_income(X, 1.5))

    increase = increased_scores - original_scores

    riskier_count = int((increase > TOLERANCE).sum())
    largest_increase = float(increase.max())

    assert largest_increase <= TOLERANCE, (
        f"{riskier_count} applicants became riskier; "
        f"largest score increase was {largest_increase:.6f}"
    )

    #pytest.fail("TODO(student): write this test")
