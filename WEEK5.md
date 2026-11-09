# Week 5: Serving and deployment (Monday)

<!-- PROVIDED -->

Your model is registered. This week anyone on the internet will be able to ask it for a
decision: by Sunday it runs behind a public API on **Google Cloud Run**. Today you build the
API and run it on your laptop; on Wednesday you put it in a container and deploy it.

```
caller ──POST /predict {"SK_ID_CURR": 100021}──▶ API ──▶ feature store: the applicant's row
                                                  │
                                                  ▼
caller ◀── {"probability": 0.07, "decision": "approve", "model_version": "1"} ── model
```

## Three endpoints

| Endpoint | Answers | Why it exists |
|---|---|---|
| `GET /health` | is the service up, is a model loaded? | Cloud Run and the monitoring (Week 6) check it |
| `GET /model-info` | which model version, threshold, features | every answer must be traceable to a registered model |
| `POST /predict` | probability, decision, threshold, model version | the decision itself |

## Where do the 164 features come from?

A request is for one applicant. The model needs 164 features, and about 40 of them summarize
many rows of the child tables (bureau credits, earlier applications, payments). Nobody calling
the API can send those. Think about this before you read on: how would you get them?

This week's answer is a **feature store**: one precomputed row per applicant, keyed by
`SK_ID_CURR`, built by the same Week 2 pipeline and the same cleaning the model was trained
on. A request sends the applicant's id, plus any amounts from the form that changed (income,
loan amount, yearly repayment). Those replace the stored values, and the ratio features that
depend on them are recomputed, exactly as in Week 2. You write that lookup today
(`build_features`), and you write down the design in `docs/api_design.md`.

## The model bundle

Cloud Run cannot reach the MLflow database on your laptop. So `python run.py export` copies
everything the service needs into `models/` (never committed):

| File | What |
|---|---|
| `models/model.joblib` | your registered model, `models:/credit-risk@staging` |
| `models/model_info.json` | its version, threshold, features, validation AUC |
| `models/feature_store.parquet` | one row of model features per applicant |

It also writes `requirements-api.txt` (committed): the service's packages, **pinned** to the
exact versions installed on your laptop. A model saved with one scikit-learn version must be
loaded with the same one.

On your laptop the service reads `models/`. On Cloud Run it downloads the same three files
from a **private** Cloud Storage bucket when it starts (the environment variable
`MODEL_BUCKET`). The feature store is derived from the Kaggle data: the bucket must never be
public, and `models/` never goes to GitHub.

## Validation for free

FastAPI checks every request against `PredictRequest` (api/schemas.py) before your code
runs. A missing id, a negative income, text where a number belongs, or a field nobody asked
for gets an automatic **422** that says what is wrong. You describe valid input; FastAPI
refuses everything else. An applicant who is not in the feature store is a different error:
the request was valid, but there is nothing to score, so your code answers **404**.

## Commands

```
python run.py export              # models/ bundle + requirements-api.txt
python run.py serve               # the API on http://127.0.0.1:8000 (Ctrl+C to stop)
python run.py test tests/api      # the API tests (no data needed: they also run in CI)
python run.py check --stage mon   # the Monday checkpoint
```

While `serve` runs, open **http://127.0.0.1:8000/docs**: FastAPI's interactive page, where
you can send requests with "Try it out".

## The files today

| File | Status | What it is |
|---|---|---|
| `api/main.py` | MIXED | `/health` (worked example); **`/model-info`**, **`/predict`** (yours) |
| `api/schemas.py` | MIXED | **`PredictRequest`** (yours to complete); the response models |
| `api/features.py` | MIXED | `recompute_ratios` (provided); **`build_features`** (yours) |
| `api/bundle.py` | PROVIDED | loads the bundle from `models/` or from the bucket |
| `src/export_model.py` | PROVIDED | `python run.py export` |
| `tests/api/` | DO NOT EDIT | 22 API tests on a small made-up bundle (9 pass now) |
| `scripts/check_week5.py` | DO NOT EDIT | the checkpoint checks |
| `docs/api_design_template.md` | PROVIDED | copy to `docs/api_design.md` |
| `pyproject.toml` | PROVIDED (replaced) | hides a warning from inside FastAPI's test client |

## Rules this week

- `main` is protected (Week 4): copy the zip and work on a branch, and merge through a pull
  request. After adding code under `src/`, run `python run.py evaluate` before you commit,
  or your own gate says "stale metrics".
- Never commit `models/`, `data/`, or any key. Never make the bucket public.
- Do not edit `scripts/check_week5.py` or the tests.
