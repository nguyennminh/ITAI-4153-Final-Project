"""API tests: the service answers correctly, and refuses what it must refuse.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.

They start the real app on a small made-up bundle (tests/api/conftest.py), so they need no
data and also run in CI on every pull request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.features import UnknownApplicant, build_features, recompute_ratios
from api.schemas import PredictRequest
from src.features.ratios import RATIO_FEATURES, add_ratio_features
from tests.api.conftest import FEATURES, FIRST_ID, make_store
from tests.week2_fixtures import small_application_full

KNOWN = FIRST_ID + 10


# ---------------------------------------------------------------- provided code (passes now)

def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "7"}


def test_service_ratios_match_week2() -> None:
    """The service must compute the ratios exactly as the training pipeline did."""
    app = small_application_full()
    expected = add_ratio_features(app)[RATIO_FEATURES]
    got = recompute_ratios(app.copy())[RATIO_FEATURES]
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


@pytest.mark.parametrize("payload", [
    {},                                         # no id
    {"SK_ID_CURR": "abc"},                      # not a number
    {"SK_ID_CURR": -5},                         # not a valid id
    {"SK_ID_CURR": KNOWN, "AMT_INCOM_TOTAL": 1},  # a typo: unknown fields are refused
])
def test_invalid_requests_get_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


# ---------------------------------------------------------------- PredictRequest

@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"])
def test_request_accepts_updated_amounts(field: str) -> None:
    request = PredictRequest(SK_ID_CURR=KNOWN, **{field: 1234.5})
    assert getattr(request, field) == 1234.5
    assert getattr(PredictRequest(SK_ID_CURR=KNOWN), field) is None, "default None"


@pytest.mark.parametrize("payload", [
    {"SK_ID_CURR": KNOWN, "AMT_INCOME_TOTAL": -1000},
    {"SK_ID_CURR": KNOWN, "AMT_CREDIT": 0},
    {"SK_ID_CURR": KNOWN, "AMT_ANNUITY": "lots"},
])
def test_impossible_amounts_get_422(client: TestClient, payload: dict) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "AMT_" in response.text, "the error names the field that is wrong"


# ---------------------------------------------------------------- build_features

def test_build_features_returns_one_row_in_model_order() -> None:
    store = make_store().set_index("SK_ID_CURR")
    row = build_features(PredictRequest(SK_ID_CURR=KNOWN), store, FEATURES)
    assert list(row.columns) == FEATURES
    assert len(row) == 1
    pd.testing.assert_series_equal(row.iloc[0], store.loc[KNOWN, FEATURES], check_names=False)


def test_build_features_applies_overrides_and_recomputes_ratios() -> None:
    store = make_store().set_index("SK_ID_CURR")
    row = build_features(PredictRequest(SK_ID_CURR=KNOWN, AMT_INCOME_TOTAL=50_000.0),
                         store, FEATURES)
    assert row["AMT_INCOME_TOTAL"].iloc[0] == 50_000.0
    expected = store.loc[KNOWN, "AMT_CREDIT"] / 50_000.0
    assert row["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(expected), \
        "a new income must update the ratios that divide by it"


def test_build_features_never_changes_the_store() -> None:
    store = make_store().set_index("SK_ID_CURR")
    before = store.copy()
    build_features(PredictRequest(SK_ID_CURR=KNOWN, AMT_CREDIT=1.0), store, FEATURES)
    pd.testing.assert_frame_equal(store, before)


def test_build_features_unknown_applicant() -> None:
    store = make_store().set_index("SK_ID_CURR")
    with pytest.raises(UnknownApplicant, match="999"):
        build_features(PredictRequest(SK_ID_CURR=999), store, FEATURES)


# ---------------------------------------------------------------- the endpoints

def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    info = response.json()
    assert info["version"] == "7" and info["name"] == "credit-risk"
    assert info["threshold"] == 0.12 and info["n_features"] == len(FEATURES)
    assert info["valid_auc"] == 0.75 and info["data_version"] == "abc123"


def test_predict_answers_with_a_decision(client: TestClient) -> None:
    response = client.post("/predict", json={"SK_ID_CURR": KNOWN})
    assert response.status_code == 200
    body = response.json()
    assert body["SK_ID_CURR"] == KNOWN and body["model_version"] == "7"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.12
    expected = "reject" if body["probability"] >= 0.12 else "approve"
    assert body["decision"] == expected


def test_predict_is_repeatable(client: TestClient) -> None:
    first = client.post("/predict", json={"SK_ID_CURR": KNOWN}).json()
    second = client.post("/predict", json={"SK_ID_CURR": KNOWN}).json()
    assert first == second


def test_predict_both_decisions_happen(client: TestClient) -> None:
    decisions = {client.post("/predict", json={"SK_ID_CURR": i}).json()["decision"]
                 for i in range(FIRST_ID, FIRST_ID + 60)}
    assert decisions == {"approve", "reject"}, "a threshold that never rejects is suspicious"


def test_predict_unknown_applicant_gets_404(client: TestClient) -> None:
    response = client.post("/predict", json={"SK_ID_CURR": 999})
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_predict_with_updated_income(client: TestClient) -> None:
    response = client.post("/predict", json={"SK_ID_CURR": KNOWN, "AMT_INCOME_TOTAL": 40000})
    assert response.status_code == 200
    assert np.isfinite(response.json()["probability"])
