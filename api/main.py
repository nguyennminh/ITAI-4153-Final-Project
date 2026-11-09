"""The credit-risk scoring service.

Run it on your laptop with ``python run.py serve``, then open http://127.0.0.1:8000/docs:
FastAPI writes an interactive page for every endpoint, where you can send test requests.

    GET  /health       is the service up, and is a model loaded?
    GET  /model-info   which model is serving (version, threshold, ...)
    POST /predict      a decision for one applicant

The model is loaded ONCE, when the service starts (``lifespan`` below), not on every
request: loading takes about a second, a prediction a few milliseconds.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException  # noqa: F401  (HTTPException is for you to use)

from api.bundle import Bundle, load_bundle
from api.features import UnknownApplicant, build_features  # noqa: F401
from api.schemas import ModelInfo, PredictRequest, PredictResponse  # noqa: F401

STATE: dict[str, Bundle] = {}


# PROVIDED
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the bundle when the service starts; drop it when it stops."""
    STATE["bundle"] = load_bundle()
    yield
    STATE.clear()


app = FastAPI(title="Credit-risk scoring service", version="1.0", lifespan=lifespan)


# PROVIDED
def bundle() -> Bundle:
    """The loaded bundle (the service refuses to answer before it is loaded)."""
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return STATE["bundle"]


# PROVIDED: the worked example.
@app.get("/health")
def health() -> dict[str, object]:
    """Liveness check: Cloud Run and the monitoring use it. Cheap, never touches the model."""
    loaded = "bundle" in STATE
    return {"status": "ok" if loaded else "starting", "model_loaded": loaded,
            "model_version": STATE["bundle"].info["version"] if loaded else None}


# TODO(student)
@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Which model is serving.

    Build a ``ModelInfo`` from ``bundle().info`` (the model_info.json written by
    ``python run.py export``): name, version, threshold, n_features (the number of
    features), valid_auc, data_version, exported. Open models/model_info.json to see the
    keys.
    """
    b = bundle()
    info = b.info

    return ModelInfo(
        name=info["name"],
        version=info["version"],
        threshold=info["threshold"],
        n_features=len(b.features),
        valid_auc=info["valid_auc"],
        data_version=info["data_version"],
        exported=info["exported"],
    )

    # raise NotImplementedError("TODO(student): describe the serving model")


# TODO(student)
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """A credit decision for one applicant.

    Start with ``b = bundle()``: the loaded model, its info, and the feature store.

    1. Build the model's input with ``build_features(request, b.store, b.features)``.
       If it raises ``UnknownApplicant``, answer 404 with
       ``raise HTTPException(status_code=404, detail=...)``.
    2. The probability of default is ``b.model.predict_proba(row)[0, 1]`` (as a float).
    3. The decision is "reject" if the probability is at least the threshold
       (``b.info["threshold"]``), else "approve". Same rule as Week 3.
    4. Return a ``PredictResponse`` with the id, probability, decision, threshold, and
       ``b.info["version"]`` as model_version.

    You do not need to check the input yourself: FastAPI already refused anything that does
    not fit ``PredictRequest`` with a 422.
    """

    b = bundle()

    # Build the applicant's feature row.
    try:
        row = build_features(request, b.store, b.features)
    except UnknownApplicant as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Calculate the probability of default.
    probability = float(b.model.predict_proba(row)[0, 1])

    # Determine the credit decision.
    threshold = float(b.info["threshold"])
    decision = "reject" if probability >= threshold else "approve"

    # Return the prediction response.
    return PredictResponse(
        SK_ID_CURR=request.SK_ID_CURR,
        probability=probability,
        decision=decision,
        threshold=threshold,
        model_version=b.info["version"],
    )

    # raise NotImplementedError("TODO(student): score the applicant and decide")
