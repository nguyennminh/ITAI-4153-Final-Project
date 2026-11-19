"""Prediction logging (Week 6 Monday): build_record, and /predict writing one record per answer.

# DO NOT EDIT
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.prediction_log import RECORD_KEYS, build_record
from api.schemas import PredictRequest, PredictResponse
from tests.api.conftest import FEATURES, FIRST_ID

MISSING_ID = FIRST_ID + 3  # its EXT_SOURCE_2 is missing in the made-up store


def _row(**values: object) -> pd.DataFrame:
    base = {f: 1.0 for f in FEATURES} | {"CODE_GENDER": "F"}
    return pd.DataFrame([base | values])


def _response(**values: object) -> PredictResponse:
    base = dict(SK_ID_CURR=FIRST_ID, probability=0.2, decision="reject", threshold=0.12,
                model_version="7")
    return PredictResponse(**(base | values))


# DO NOT EDIT
def test_build_record_has_every_key() -> None:
    record = build_record(PredictRequest(SK_ID_CURR=FIRST_ID), _row(), _response())
    assert set(record) == set(RECORD_KEYS)
    assert record["event"] == "prediction" and record["severity"] == "INFO"
    assert record["SK_ID_CURR"] == FIRST_ID and record["model_version"] == "7"
    assert record["probability"] == pytest.approx(0.2) and record["decision"] == "reject"
    assert record["threshold"] == pytest.approx(0.12)


# DO NOT EDIT
def test_build_record_ids_and_time() -> None:
    a = build_record(PredictRequest(SK_ID_CURR=FIRST_ID), _row(), _response())
    b = build_record(PredictRequest(SK_ID_CURR=FIRST_ID), _row(), _response())
    assert a["request_id"] != b["request_id"], "every request needs its own new id"
    assert len(a["request_id"]) >= 16
    assert pd.Timestamp(a["timestamp"]).tzinfo is not None, "the timestamp needs a time zone"


# DO NOT EDIT
def test_build_record_features_are_valid_json() -> None:
    row = _row(EXT_SOURCE_2=np.nan, AMT_CREDIT=np.float64(5e5))
    record = build_record(PredictRequest(SK_ID_CURR=FIRST_ID), row, _response())
    assert set(record["features"]) == set(FEATURES)
    assert record["features"]["EXT_SOURCE_2"] is None, "a missing value must be None"
    assert record["features"]["AMT_CREDIT"] == 5e5
    json.dumps(record, allow_nan=False)  # fails if any NaN or numpy value is left


# DO NOT EDIT
def test_build_record_overrides() -> None:
    plain = build_record(PredictRequest(SK_ID_CURR=FIRST_ID), _row(), _response())
    assert plain["overrides"] == {}
    changed = build_record(PredictRequest(SK_ID_CURR=FIRST_ID, AMT_INCOME_TOTAL=50_000),
                           _row(), _response())
    assert changed["overrides"] == {"AMT_INCOME_TOTAL": 50_000}


def _logged(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_FILE", str(log))
    response = client.post("/predict", json=payload)
    lines = log.read_text().splitlines() if log.exists() else []
    return response, [json.loads(line) for line in lines]


# DO NOT EDIT
def test_predict_logs_one_record(client, tmp_path, monkeypatch) -> None:
    response, records = _logged(client, tmp_path, monkeypatch, {"SK_ID_CURR": FIRST_ID})
    assert response.status_code == 200
    assert len(records) == 1, ("/predict must call log_prediction(request, row, response) "
                               "once, just before it returns")
    record, body = records[0], response.json()
    assert record["SK_ID_CURR"] == FIRST_ID
    assert record["probability"] == pytest.approx(body["probability"])
    assert record["decision"] == body["decision"]
    assert record["model_version"] == body["model_version"]


# DO NOT EDIT
def test_predict_logs_what_the_model_saw(client, tmp_path, monkeypatch) -> None:
    _, records = _logged(client, tmp_path, monkeypatch,
                         {"SK_ID_CURR": MISSING_ID, "AMT_INCOME_TOTAL": 40_000})
    assert len(records) == 1, "/predict must call log_prediction(request, row, response)"
    features = records[0]["features"]
    assert features["AMT_INCOME_TOTAL"] == 40_000, "log the row after the overrides"
    assert features["EXT_SOURCE_2"] is None
    assert features["CREDIT_INCOME_RATIO"] == pytest.approx(features["AMT_CREDIT"] / 40_000)
    assert records[0]["overrides"] == {"AMT_INCOME_TOTAL": 40_000}


# DO NOT EDIT
def test_errors_are_not_logged(client, tmp_path, monkeypatch) -> None:
    response, records = _logged(client, tmp_path, monkeypatch, {"SK_ID_CURR": 1})
    assert response.status_code == 404 and records == []
    response, records = _logged(client, tmp_path, monkeypatch,
                                {"SK_ID_CURR": FIRST_ID, "AMT_CREDIT": -1})
    assert response.status_code == 422 and records == []


# DO NOT EDIT
def test_logging_failure_does_not_fail_predict(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PREDICTION_LOG_FILE", str(tmp_path / "no_such_folder" / "log.jsonl"))
    response = client.post("/predict", json={"SK_ID_CURR": FIRST_ID})
    assert response.status_code == 200, "a logging problem must never fail the prediction"
