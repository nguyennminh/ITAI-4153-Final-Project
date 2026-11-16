<!-- PROVIDED: copy this file to docs/api.md and fill in every placeholder.
     This is the API's README: someone who has never seen your code must be able to call it. -->

# Credit-risk scoring API

Team: Group 4 (Krish Iyer, Minh Nguyen, Calvin Zheng)

Base URL: https://credit-risk-api-196591733163.us-central1.run.app (also in
docs/deployment.md). Interactive documentation:
https://credit-risk-api-196591733163.us-central1.run.app/docs.

The service gives a loan decision for one applicant of our Home Credit sample. It looks up
the applicant's 164 features in the feature store, scores them with our registered model
(`credit-risk@staging`, version 1), and rejects the application when the probability of
default is at least 0.08. The answers below come from the live service. Set `URL` first:

```bash
URL=https://credit-risk-api-196591733163.us-central1.run.app
```

## Endpoints

### GET /health

Is the service up, and has it loaded the model? Cloud Run and monitoring call it. It never
runs the model, so it is cheap.

```bash
curl -s $URL/health
```

```json
{"status": "ok", "model_loaded": true, "model_version": "1"}
```

### GET /model-info

Which model is serving: its version, decision threshold, number of features, validation AUC,
and the data and export it came from.

```bash
curl -s $URL/model-info
```

```json
{
  "name": "credit-risk",
  "version": "1",
  "threshold": 0.08,
  "n_features": 164,
  "valid_auc": 0.7471218044304732,
  "data_version": "9fb9955ad0ed",
  "exported": "2026-11-12T20:23:52+00:00"
}
```

(`exported` is when the bundle in the bucket was exported.)

### POST /predict

A decision for one applicant. The body needs `SK_ID_CURR`, an id from the feature store. It
may also send updated amounts, each greater than 0: `AMT_INCOME_TOTAL`, `AMT_CREDIT`,
`AMT_ANNUITY`. The service then uses them instead of the stored values and recomputes the
ratio features. Any other field is refused.

```bash
curl -s -X POST $URL/predict -H "Content-Type: application/json" \
    -d '{"SK_ID_CURR": 100021}'
```

```json
{
  "SK_ID_CURR": 100021,
  "probability": 0.1282051282051282,
  "decision": "reject",
  "threshold": 0.08,
  "model_version": "1"
}
```

`probability` is the predicted chance of default. `decision` is `reject` when it is at least
`threshold`, otherwise `approve`.

## Errors

**422: the request does not fit.** A missing or non-integer `SK_ID_CURR`, an amount of 0 or
less, or an unknown field. FastAPI refuses it before the model runs and says which field is
wrong:

```bash
curl -s -X POST $URL/predict -H "Content-Type: application/json" \
    -d '{"SK_ID_CURR": 100021, "AMT_INCOME_TOTAL": -1000}'
```

```json
{"detail": [{"type": "greater_than", "loc": ["body", "AMT_INCOME_TOTAL"],
  "msg": "Input should be greater than 0", "input": -1000, "ctx": {"gt": 0.0}}]}
```

**404: unknown applicant.** The request is valid, but the id is not in the feature store, so
there are no features to score:

```bash
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d '{"SK_ID_CURR": 1}'
```

```json
{"detail": "Applicant with SK_ID_CURR 1 was not found."}
```

## Response times

Measured on 2026-11-16 at 17:44 UTC from a laptop in North Carolina with
`python run.py latency --url <base URL>`, after the service had received no requests for 22
minutes (reports/latency.json):

| | |
|---|---|
| First request (cold start) | 20,308 ms |
| Requests after it | 100, 0 errors |
| p50 | 88 ms |
| p95 | 191 ms |
| Slowest | 243 ms |
| Mean | 104 ms |

Once an instance is running, the person on the loan screen typically gets the decision in
under a tenth of a second (p50), and 19 requests in 20 finish within about 0.2 seconds (p95).
Both feel instant. These times include the network trip from the laptop to us-central1, not
only the model. The problem is the first request after a quiet period. With min instances at
0, Cloud Run has shut the service down, so a new container has to start, download the model
bundle from the bucket and load it before it can answer. The applicant who triggers that
waits about 20 seconds, which is long enough to think the page is broken. Setting min
instances to 1 would remove the wait for the cost of one instance kept running all the time.
For a class project we keep 0. For a real lender taking applications during the day, 1
instance during business hours would be worth it.

## Online or batch scoring?

Online scoring, which this API does, scores one applicant at the moment they apply and
answers in milliseconds. Batch scoring scores many applicants at once on a schedule, for
example every night, and stores the results. Batch is cheaper per applicant and simpler to
run, since one job reads the data once and nothing has to stay up waiting for requests. It
suits work that can wait, such as re-scoring every existing customer for a monthly risk
report. A credit decision cannot wait: the applicant is on the loan screen, and they may
change the amount they ask for, which changes the ratio features. A batch score from last
night would not know about that change, and a new applicant would have no score at all. So
online scoring suits our decision, at the cost of a service that must stay up and a cold
start when it has scaled to zero. Batch still has a place behind it: rebuilding the feature
store on a schedule keeps applicants' histories current.
