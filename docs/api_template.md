<!-- PROVIDED: copy this file to docs/api.md and fill in every placeholder.
     This is the API's README: someone who has never seen your code must be able to call it. -->

# Credit-risk scoring API

Team: TODO

Base URL: TODO (https://...run.app). Interactive documentation: `<base URL>/docs`.

## Endpoints

TODO: For each of /health, /model-info and /predict: the method, what it is for, an example
request (curl) and its answer.

## Errors

TODO: What a caller gets for a request that does not fit (422, with an example), and for an
applicant who is not in the feature store (404, with an example).

## Response times

TODO: From reports/latency.json: how many requests, p50, p95, the slowest, and the first
request (cold start), measured against the live service on <date>. What do p50 and p95 mean
for the person waiting on the loan screen?

## Online or batch scoring?

TODO: A short note (5-10 sentences). Online scoring answers one applicant at a time, at the
moment they apply (this API). Batch scoring scores many applicants at once on a schedule
(for example every night). When is each the better choice, what does each cost, and which
suits a credit decision, and why?
