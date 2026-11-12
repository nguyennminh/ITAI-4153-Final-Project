# Deploying to Cloud Run (the checklist)

<!-- PROVIDED -->

Everything below runs in **Cloud Shell**, the terminal built into the Google Cloud Console:
click the **>_** icon at the top right of https://console.cloud.google.com. It already has
`gcloud` and Docker, is logged in as you, and builds images for the right processor (an
image built on an Apple-Silicon Mac does not run on Cloud Run). Nothing to install.

**One member** runs these steps (the project owner from Week 4). Copy each block, check what
it prints, then go on. Text in `<angle brackets>` is yours to fill in.

## 0. Settings for this terminal (every time you open Cloud Shell)

```bash
gcloud config set project <your-project-id>
PROJECT=$(gcloud config get-value project)
REGION=us-central1
BUCKET=<your bucket's name, for example credit-risk-models-alex>
IMAGE=$REGION-docker.pkg.dev/$PROJECT/credit-risk/api:v1
SA=credit-risk-api@$PROJECT.iam.gserviceaccount.com
```

## 1. Turn on the services (once, 1 minute)

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## 2. The model bucket (once; skip it if you did it on Monday)

A **private** bucket for the bundle (`python run.py export` wrote it into models/ on your
laptop). It holds data derived from the Kaggle data, so public access stays blocked. In the
Console (not Cloud Shell):

1. Open https://console.cloud.google.com/storage/browser (check the project picker at the
   top shows your project) and click **+ Create**.
2. **Name:** `credit-risk-models-<your-name-or-team>` (names are unique across all of Google;
   if it is taken, add a number). **Continue**.
3. **Location type:** Region, **us-central1 (Iowa)**. **Continue**.
4. **Storage class:** Standard. **Continue**.
5. **Access control:** keep **"Enforce public access prevention on this bucket"** ticked and
   **Uniform** access. **Continue**, then **Create**. If asked "Public access will be
   prevented", click **Confirm**.
6. In the new, empty bucket: **Upload → Upload files**, go to your repository's `models/`
   folder, select all three files (`feature_store.parquet`, `model.joblib`,
   `model_info.json`) and open them. Wait until the upload panel says **Complete** for all three.
7. Check: the three files are at the **top level** of the bucket (not in a folder), and the
   bucket's **Public access** says **Not public**.

Then, back in Cloud Shell, set `BUCKET` (step 0) to that name and check:

```bash
gcloud storage ls gs://$BUCKET
```

It lists the three files. After every `python run.py export` (a new model), upload them again.

## 3. The service's own account (once)

The running service needs to read the bucket, and nothing else:

```bash
gcloud iam service-accounts create credit-risk-api --display-name="Credit-risk API"
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
    --member=serviceAccount:$SA --role=roles/storage.objectViewer
```

## 4. A place for your images (once)

```bash
gcloud artifacts repositories create credit-risk --repository-format=docker \
    --location=$REGION --description="Credit-risk API images"
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
```

## 5. Get your code (every new deploy)

```bash
cd ~ && rm -rf <repo> && git clone https://github.com/<your-account>/<repo>.git
cd <repo>
```

Your repository must be public (Week 4), and `requirements-api.txt` must be committed.

## 6. Build the image (2-3 minutes)

```bash
docker build -t $IMAGE .
docker images $IMAGE
```

The second command shows the image's size (expect a few hundred MB).

## 7. Try it in Cloud Shell before deploying

```bash
mkdir -p models && gcloud storage cp "gs://$BUCKET/*" models/
docker run --rm -p 8080:8080 -v "$PWD/models:/app/models" $IMAGE
```

Wait for `Uvicorn running on http://0.0.0.0:8080`. Open a **second Cloud Shell tab** (the +
next to the tab name), and:

```bash
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -H "Content-Type: application/json" \
    -d '{"SK_ID_CURR": <an id from your feature store>}'
```

Then stop the container with Ctrl+C in the first tab.

## 8. Push and deploy (3-5 minutes)

```bash
docker push $IMAGE
gcloud run deploy credit-risk-api --image=$IMAGE --region=$REGION \
    --service-account=$SA --set-env-vars=MODEL_BUCKET=gs://$BUCKET \
    --allow-unauthenticated --min-instances=0 --max-instances=2 \
    --memory=1Gi --cpu=1 --port=8080
```

- `--allow-unauthenticated`: anyone can call it (it is a course demo; no real customers).
- `--min-instances=0`: no instance runs (and costs nothing) when nobody calls it; the first
  call after a quiet period waits for a **cold start**.
- `--max-instances=2`: a traffic spike can never run up your credits.

It ends with `Service URL: https://credit-risk-api-....run.app`. Write it in
`docs/deployment.md`.

## 9. Call the live service

```bash
URL=$(gcloud run services describe credit-risk-api --region=$REGION --format='value(status.url)')
curl -s $URL/health
curl -s $URL/model-info
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d '{"SK_ID_CURR": <an id>}'
```

## When something goes wrong

| What you see | Why, and what to do |
|---|---|
| `Permission denied` on `buckets create` / `services enable` | Wrong project: run step 0 again with your project id |
| `docker build` fails at `pip install` with "No matching distribution" | The FROM line's Python version differs from requirements-api.txt's: match them |
| Deploy says "container failed to start and listen on the port" | Read the log: **Cloud Run → credit-risk-api → Logs**. Most often the bundle could not be read (step 3, or the files are not at the top of the bucket) |
| `403 ... storage.objects.get` in the log | Step 3 was skipped, or the deploy did not use `--service-account=$SA` |
| `/predict` answers 404 for every id | You are sending ids that are not in the feature store: take one from it |
| The first call takes several seconds | A cold start (min instances 0): expected. Measure it, don't hide it |
| `allUsers` / unauthenticated access refused | Your account's organization forbids public services: tell the instructor |

## Never

- Never commit a key or a service-account JSON file. Nothing here needs one.
- Never make the bucket public.
- Keep the service up until the end of the course: it is graded in Weeks 6 and 7.
