"""What a request to the API must look like, and what the API answers.

FastAPI checks every request against ``PredictRequest`` before your code runs. A request
that does not fit (a missing id, a negative income, a field nobody asked for, text where a
number belongs) gets an automatic **422** answer that says what is wrong, and never reaches
the model. That is how "invalid input returns an error" works: you describe valid input,
and FastAPI refuses everything else.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# TODO(student): complete PredictRequest.
class PredictRequest(BaseModel):
    """One credit decision request.

    The applicant is identified by ``SK_ID_CURR`` (given below). Add three OPTIONAL fields
    that let the caller send updated amounts from the application form, which then replace
    the stored values:

    - ``AMT_INCOME_TOTAL``: yearly income, must be greater than 0
    - ``AMT_CREDIT``: the loan amount, must be greater than 0
    - ``AMT_ANNUITY``: the yearly repayment, must be greater than 0

    Each is a ``float | None`` with default ``None`` (None means "use the stored value").

    Hints:
        - ``AMT_INCOME_TOTAL: float | None = Field(default=None, gt=0, description="...")``
        - ``model_config`` below already refuses any field that is not declared (``extra``
          is "forbid"), so a typo like ``AMT_INCOM_TOTAL`` gets a 422 instead of being
          silently ignored.
    """
    
    model_config = ConfigDict(extra="forbid")

    SK_ID_CURR: int = Field(gt=0, description="The applicant's id")

    AMT_INCOME_TOTAL: float | None = Field(
        default=None,
        gt=0,
        description="Updated yearly income; uses the stored value if omitted",
    )
    AMT_CREDIT: float | None = Field(
        default=None,
        gt=0,
        description="Updated loan amount; uses the stored value if omitted",
    )
    AMT_ANNUITY: float | None = Field(
        default=None,
        gt=0,
        description="Updated yearly repayment; uses the stored value if omitted",
    )    


# PROVIDED
class PredictResponse(BaseModel):
    """The decision for one applicant."""

    SK_ID_CURR: int
    probability: float = Field(description="Estimated probability of default, 0 to 1")
    decision: str = Field(description='"reject" if probability >= threshold, else "approve"')
    threshold: float
    model_version: str


# PROVIDED
class ModelInfo(BaseModel):
    """What model is serving, so anyone can trace an answer back to a registered version."""

    name: str
    version: str
    threshold: float
    n_features: int
    valid_auc: float
    data_version: str
    exported: str
