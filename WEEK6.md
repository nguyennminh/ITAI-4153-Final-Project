# Week 6: Monitoring and responsible AI (Monday)

<!-- PROVIDED -->

Your model is live. Nobody is watching it. This week you make it observable: every
decision is logged, production traffic arrives, and you check whether what the model sees
in production still looks like what it learned from. On Wednesday you explain its decisions
and check them for fairness.

```
send batch ──▶ live API ──▶ decision ──▶ caller
                  │
                  └─▶ one JSON log line per decision ──▶ Cloud Logging
                                                             │ gcloud logging read
                                                             ▼
                 training data (X_train) ──▶ python run.py monitor ◀── data/logs/predictions.json
                                                 │
                                                 ▼
                         reports/monitoring/: PSI, KS, Evidently, plots
```

## 1. Prediction logging

`api/prediction_log.py` turns each answered request into one record:

| Field | Why it is there |
|---|---|
| `request_id`, `timestamp` | find one decision later; order them in time |
| `model_version` | which model made it: after a new model ships, you compare old and new |
| `SK_ID_CURR`, `overrides` | who, and which amounts the caller changed |
| `features` | what the model **actually received** (after the feature store and the overrides): this is what you monitor |
| `probability`, `decision`, `threshold` | what came out |

The service prints the record as one JSON line. On Cloud Run, a JSON line on standard output
becomes a structured entry in **Cloud Logging**, no extra library or permission needed. You
write `build_record`, and you add the call to your `/predict`:

```python
from api.prediction_log import log_prediction          # at the top of api/main.py

    response = PredictResponse(...)                     # what you return today
    log_prediction(request, row, response)              # one record per answer
    return response
```

Only answered requests are logged: a 404 or 422 never reaches that line. And logging never
breaks a prediction: if it fails, the customer still gets an answer and a warning goes to the
error log (`log_prediction` already does this).

On your laptop you can also write the same lines to a file, to try everything before
deploying:

- Mac / Linux: `PREDICTION_LOG_FILE=data/logs/local.jsonl python run.py serve`
- Windows (PowerShell): `$env:PREDICTION_LOG_FILE="data/logs/local.jsonl"; python run.py serve`

(create the `data/logs` folder first). The checkpoint needs the **live** log, though.

## 2. Production traffic (simulated)

The Home Credit data has no dates, so production is simulated with applicants your model has
never seen. `python run.py pool` takes **3,000 applicants that are not in your sample**
from your full download (`data/raw`), builds their features with **your** Week 2 pipeline,
and splits them into three batches of 1,000:

| Batch | When | What |
|---|---|---|
| 1 | "this week" | new applicants, nothing changed |
| 2 | "a week later" | something changed upstream. **Your job: find out what**, with numbers |
| 3 | Week 7 | kept for later; do not send it |

The service can only score applicants in its feature store, so `python run.py export` now
adds the 3,000 to it (8,000 applicants), and you upload it and redeploy
(`docs/cloud_shell_week6.md`). `data/production/labels.parquet` says who really defaulted:
do not use it this week. In production you would not know for months.

## 3. Drift: did the inputs change?

`python run.py monitor` compares each logged batch with the model's training data,
feature by feature:

- **PSI** (population stability index), which you write: split the training values into 10
  bins of equal size plus one bin for missing values, then compare the share of production
  values in each bin. 0 is identical; below 0.1 means no real change, 0.1 to 0.25 a moderate
  change, above 0.25 a major one.
- **KS test**: the largest gap between the two cumulative distributions, with a p-value.
  With thousands of rows even harmless differences get tiny p-values: read the statistic.
- **Evidently**: an open-source monitoring library; its interactive report per batch is in
  `reports/monitoring/evidently/` (open it in a browser). Compare its verdict with yours.
- The **scores** and the **approval rate** per batch, compared with the validation set.

## Commands

```
python run.py test tests/api tests/test_week6_mon.py   # logging and PSI tests (no data needed)
python run.py pool                 # data/production/: 3 batches + the store extension
python run.py export               # the bundle, now with 8,000 applicants
python run.py evaluate             # the zip added code under src/: fresh gate metrics
python run.py send --batch 1 --url https://...run.app     # then --batch 2
python run.py monitor              # reads data/logs/predictions.json
python run.py check --stage mon    # the Monday checkpoint
```

## The files today

| File | Status | What it is |
|---|---|---|
| `api/prediction_log.py` | MIXED | `json_safe`, `emit`, `log_prediction` (provided); **`build_record`** (yours) |
| `api/main.py` | yours (Week 5) | **add the `log_prediction` call** (above) |
| `src/monitoring.py` | MIXED | **`psi`** (yours); the rest of `python run.py monitor` (provided) |
| `src/production_pool.py` | PROVIDED | `python run.py pool` |
| `src/export_model.py` | PROVIDED (replaced) | adds the production applicants to the feature store |
| `scripts/send_batch.py` | PROVIDED | `python run.py send` |
| `tests/api/test_logging.py`, `tests/test_week6_mon.py` | DO NOT EDIT | 8 logging tests and 7 PSI tests |
| `scripts/check_week6.py` | DO NOT EDIT | the checkpoint checks |
| `docs/cloud_shell_week6.md` | PROVIDED | redeploy, then download the log |
| `docs/monitoring_report_template.md` | PROVIDED | copy to `docs/monitoring_report.md` |
| `.gitignore`, `run.py` | PROVIDED (replaced) | Evidently's large HTML files stay local; new commands |

## Rules this week

- `main` is protected: work on a branch, merge through a pull request. After adding code
  under `src/`, run `python run.py evaluate` before you commit.
- Never commit `data/` (the production batches and the logs hold Kaggle-derived data),
  `models/`, or any key. The bucket stays private.
- Do not edit `scripts/check_week6.py` or the tests.
