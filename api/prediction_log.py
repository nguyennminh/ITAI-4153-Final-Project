"""Prediction logging: one record per decision, so you can monitor what the model really saw.

Every answer from /predict is logged as ONE line of JSON on standard output. On Cloud Run,
anything a service prints goes to **Cloud Logging**, and a line that is valid JSON is stored
as a structured entry (its fields appear under ``jsonPayload``), so you can later download
every prediction with one ``gcloud logging read`` command. On your laptop, set the
environment variable ``PREDICTION_LOG_FILE`` and the same lines are also appended to that
file.

A record answers, for any decision, months later: which request, when, which model version,
what went in (the features the model actually received, after the feature store lookup and
any overrides), and what came out. Without it you cannot see drift, and you cannot explain
a decision to a customer or a regulator.

Logging must never break a prediction: if writing the log fails, the customer still gets
their answer, and a warning goes to the error log instead.
"""

from __future__ import annotations

import json
import math
import os
import sys
import uuid  # noqa: F401  (for you to use)
from datetime import UTC, datetime  # noqa: F401  (for you to use)
from typing import Any

import pandas as pd

from api.features import OVERRIDABLE  # noqa: F401  (for you to use)
from api.schemas import PredictRequest, PredictResponse

EVENT = "prediction"
RECORD_KEYS = ("event", "severity", "request_id", "timestamp", "model_version", "SK_ID_CURR",
               "overrides", "features", "probability", "decision", "threshold")


# PROVIDED
def json_safe(value: Any) -> Any:
    """A value JSON can store: NaN and missing become None, numpy numbers become Python ones."""
    if hasattr(value, "item"):  # numpy scalars
        value = value.item()
    if value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


# TODO(student)
def build_record(request: PredictRequest, row: pd.DataFrame,
                 response: PredictResponse) -> dict[str, Any]:
    """The log record for one answered request.

    Args:
        request: what the caller sent (the id, and any amounts they overrode).
        row: the one-row DataFrame the model scored (from ``build_features``).
        response: what /predict answers.

    Returns:
        A dict with exactly the keys in ``RECORD_KEYS``:

        - ``event``: the constant ``EVENT`` ("prediction"); ``severity``: "INFO"
          (Cloud Logging reads it as the entry's severity)
        - ``request_id``: a new random id for this request: ``uuid.uuid4().hex``
        - ``timestamp``: now, in UTC: ``datetime.now(UTC).isoformat()``
        - ``model_version``, ``SK_ID_CURR``, ``probability``, ``decision``, ``threshold``:
          from ``response``
        - ``overrides``: a dict of the amounts the caller sent, only those that are not
          None, for each name in ``OVERRIDABLE`` (an empty dict if none)
        - ``features``: the row's values as a dict {feature name: value}, every value passed
          through ``json_safe`` (a missing value must be None, not NaN, or the line is not
          valid JSON)

    Hints:
        - ``row.iloc[0].to_dict()`` is {column: value} for the one row.
        - ``getattr(request, name)`` reads a request field by its name.
    """
    raise NotImplementedError("TODO(student): build the prediction log record")


# PROVIDED
def emit(record: dict[str, Any]) -> None:
    """Write the record as one JSON line to stdout (Cloud Logging), and to the file named by
    the PREDICTION_LOG_FILE environment variable, if it is set."""
    line = json.dumps(record, allow_nan=False, separators=(",", ":"))
    print(line, flush=True)
    path = os.environ.get("PREDICTION_LOG_FILE")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# PROVIDED
def log_prediction(request: PredictRequest, row: pd.DataFrame,
                   response: PredictResponse) -> None:
    """Build and write the record; never let a logging problem fail the request."""
    try:
        emit(build_record(request, row, response))
    except Exception as err:  # noqa: BLE001  (any logging error: report it, keep serving)
        print(f"WARNING: prediction not logged: {type(err).__name__}: {err}",
              file=sys.stderr, flush=True)
