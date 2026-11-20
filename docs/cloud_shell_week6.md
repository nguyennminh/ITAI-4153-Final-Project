# Week 6 in Cloud Shell: redeploy, then download the prediction log

<!-- PROVIDED -->

**One member** (the one who deployed in Week 5) runs these steps in **Cloud Shell** (the
**>_** icon at the top right of https://console.cloud.google.com). Copy each block, check what
it prints, then go on. Text in `<angle brackets>` is yours to fill in.

Before you start, on your laptop: `python run.py pool`, `python run.py export`, the logging
tests pass, and your pull request (with `api/prediction_log.py` and the logging call in
`api/main.py`) is **merged into main**. Cloud Shell builds the image from GitHub's main.

## Part A: the new feature store and the new code (about 10 minutes)

### A1. Upload the new bundle (Console, not Cloud Shell)

`python run.py export` wrote a new `models/feature_store.parquet`: 8,000 applicants (your
5,000 plus the 3,000 production applicants). The service only knows applicants that are in
it, so the bucket needs the new file.

1. Open https://console.cloud.google.com/storage/browser and click your bucket
   (`credit-risk-models-...`).
2. **Upload → Upload files**, select all three files in your repository's `models/` folder.
3. When asked whether to overwrite the existing files, choose **Overwrite**. Wait until the
   upload panel says **Complete** for all three.
4. Check the **Last modified** time of `feature_store.parquet`: it is today, a minute ago.

### A2. Settings for this terminal (as in Week 5, with a new image tag)

```bash
gcloud config set project <your-project-id>
PROJECT=$(gcloud config get-value project)
REGION=us-central1
BUCKET=<your bucket's name>
IMAGE=$REGION-docker.pkg.dev/$PROJECT/credit-risk/api:v2
SA=credit-risk-api@$PROJECT.iam.gserviceaccount.com
```

`v2`: a new tag for the new code, so `v1` stays available if you ever need to go back.

### A3. Get the merged code, build, push, deploy (5-8 minutes)

```bash
cd ~ && rm -rf <repo> && git clone https://github.com/<your-account>/<repo>.git
cd <repo>
grep -c log_prediction api/main.py
```

The last line must print `2` (the import and the call). If it prints `0`, your pull request
is not merged yet: merge it, then run this block again.

```bash
docker build -t $IMAGE .
docker push $IMAGE
gcloud run deploy credit-risk-api --image=$IMAGE --region=$REGION \
    --service-account=$SA --set-env-vars=MODEL_BUCKET=gs://$BUCKET \
    --allow-unauthenticated --min-instances=0 --max-instances=2 \
    --memory=1Gi --cpu=1 --port=8080
```

Same command as Week 5, with the new image. The URL does not change.

### A4. Check the live service knows a production applicant

```bash
URL=$(gcloud run services describe credit-risk-api --region=$REGION --format='value(status.url)')
curl -s $URL/health
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d '{"SK_ID_CURR": <first id in data/production/batch3.jsonl>}'
```

A decision (not a 404) means the new feature store is live. Batch 3 is for Week 7, so one
test request from it does not disturb anything.

## Part B: send the batches (on your laptop)

```
python run.py send --batch 1 --url <your service URL>
python run.py send --batch 2 --url <your service URL>
```

Each takes about 30 seconds and must end with `1,000 of 1,000 answered 200`. Send each batch
**once**.

## Part C: download the prediction log (Cloud Shell, 2 minutes)

Wait a minute after sending (log entries take a few seconds to arrive), then:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="credit-risk-api" AND jsonPayload.event="prediction"' \
  --freshness=7d --limit=5000 --format=json > predictions.json
python3 -c "import json; print(len(json.load(open('predictions.json'))), 'prediction records')"
```

It prints at least 2,000 records (a few more from your tests). Then:

```bash
cloudshell download predictions.json
```

Your browser saves `predictions.json` (about 10-15 MB). Move it into your repository as
**`data/logs/predictions.json`** (create the `data/logs` folder). It is under `data/`, so
git never commits it: it holds applicants' features, derived from the Kaggle data.

You can also look at the entries in the Console: **Logging → Logs Explorer**, paste the
query in quotes above into the query box, **Run query**, and open one entry: your fields are
under `jsonPayload`.

## When something goes wrong

| What you see | Why, and what to do |
|---|---|
| `send` gives 404 for every request | The live feature store is the old one: A1 (upload with Overwrite), then redeploy (A3) so the service reads it again |
| `docker build` sits at `FROM ... python:3.11-slim` for minutes | Docker Hub is slow from Cloud Shell: Ctrl+C, `docker pull mirror.gcr.io/library/python:3.11-slim`, `docker tag mirror.gcr.io/library/python:3.11-slim python:3.11-slim`, build again |
| `grep -c log_prediction` prints 0 | The pull request is not merged, or you cloned before merging |
| `predictions.json` holds 0 records | The image is the old one (no logging): check A3. Or the logs have not arrived yet: wait a minute |
| Fewer than 2,000 records | `--freshness` too short (you sent more than 7 days ago), or a batch was not sent completely |
| `cloudshell download` does nothing | Use the **⋮** menu of Cloud Shell → **Download**, and type `predictions.json` |
| `monitor` says "built from a local file" | You ran it on a file written by `PREDICTION_LOG_FILE` on your laptop; the checkpoint needs the live log |
