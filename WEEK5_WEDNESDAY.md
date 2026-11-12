# Week 5: Serving and deployment (Wednesday)

<!-- PROVIDED -->

On Monday your API ran on your laptop. Today it goes live: in a **container**, on **Google
Cloud Run**, at a public address that anyone can call, and it stays up until the end of the
course.

```
Dockerfile ──docker build──▶ image ──docker push──▶ Artifact Registry
                                                         │ gcloud run deploy
                                                         ▼
       private bucket (model bundle) ──at startup──▶ Cloud Run service ◀── https://...run.app
```

## Why a container?

"It works on my laptop" is not a deployment. A **container image** packs the code, the
exact package versions, and the command that starts the service into one file that runs the
same way everywhere: on a teammate's laptop, in Cloud Shell, on Cloud Run. The **Dockerfile**
is the recipe:

| Line | Why |
|---|---|
| `FROM python:3.11-slim` | a small Linux with Python; *slim* leaves out what a service never needs. Same Python minor version as your laptop |
| `COPY requirements-api.txt` + `RUN pip install -r` | the pinned versions, installed BEFORE the code is copied, so Docker reuses this slow step when only the code changes |
| `COPY api/ api/` | only the service's code: no data, no model, no tests (`.dockerignore`) |
| `USER appuser` | not root: if someone breaks into the service, they are not all-powerful in the container |
| `CMD ... uvicorn ... --port ${PORT}` | start the service on the port Cloud Run gives it |

## Where things run today

Everything in the cloud runs from **Cloud Shell**, the terminal built into the Google Cloud
Console (the **>_** icon). It has Docker and `gcloud` and is logged in as you: nothing to
install, and it builds images for the processor Cloud Run uses (an image built on an
Apple-Silicon Mac would not start there). Follow `docs/deploy_cloud_run.md` step by step.

## Cloud Run in three settings

- **min instances 0:** nothing runs (and nothing is charged) when nobody calls. The first
  call after a quiet period waits for a **cold start**: a new container starts and loads the
  model. You will measure it.
- **max instances 2:** a traffic spike can never run up your credits.
- **unauthenticated:** anyone can call it. Fine for a course demo with no real customers.

## Measuring response times

`python run.py latency --url https://...run.app` sends one request on its own (the cold start
if no instance was running), then 100 more, and reports **p50** (the typical wait) and
**p95** (what the slowest 1 in 20 waits). Users notice the slow ones: p95 matters more than
the average. Writing `summarize` is today's code TODO.

## Commands

```
python run.py latency                         # against your laptop (python run.py serve)
python run.py latency --url https://...run.app   # against the live service
python run.py check --stage wed               # the Wednesday checkpoint (calls your live URL)
```

## The files today

| File | Status | What it is |
|---|---|---|
| `Dockerfile` | MIXED: base image and code copy provided; **3 TODO parts** | the image recipe |
| `.dockerignore` | PROVIDED | what never goes into the image |
| `scripts/measure_latency.py` | MIXED: **`summarize`** (yours) | `python run.py latency` |
| `docs/deploy_cloud_run.md` | PROVIDED | the Cloud Shell checklist, step by step |
| `docs/deployment_template.md`, `docs/api_template.md` | PROVIDED | copy to `docs/deployment.md` and `docs/api.md` |
| `scripts/check_week5.py` | DO NOT EDIT (replaced) | adds `--stage wed` and the submission checks |
| `tests/test_week5_wed.py` | DO NOT EDIT | 7 tests: the latency summary and the Dockerfile |

## Rules today

- Never commit a key or a service-account file; nothing here needs one.
- The bucket stays private. The service stays up until the end of the course.
- Work on a branch and merge through a pull request; your CI (Week 4) still runs on every
  pull request, and it now also runs the API tests.
