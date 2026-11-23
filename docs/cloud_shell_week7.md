# Week 7 in Cloud Shell: one folder per model version, and switching between them

<!-- PROVIDED -->

From today, every model version lives in its own folder of your bucket, and the live
service is told which folder to load:

```
gs://credit-risk-models-<name>/
├── models/v1/   model.joblib  model_info.json  feature_store.parquet   ← the champion today
├── models/v2/   ... the challenger's bundle (if you promote it)
└── (the three old top-level files from Weeks 5-6; leave them)
```

**Promoting or rolling back = pointing the service at another folder.** No new image, no
rebuild: Cloud Run starts a new revision that downloads that folder's bundle (about 1
minute). That is why a rollback is fast and safe.

**One member** runs these steps in **Cloud Shell** (the **>_** icon at the top right of
https://console.cloud.google.com). Copy each block, check what it prints, then go on.

## 0. Settings (every time Cloud Shell opens or reconnects)

```bash
gcloud config set project <your-project-id>
PROJECT=$(gcloud config get-value project)
REGION=us-central1
BUCKET=<your bucket's name>
URL=$(gcloud run services describe credit-risk-api --region=$REGION --format='value(status.url)')
echo $BUCKET $URL
```

**Check:** it prints your bucket's name and your service URL. If a later command complains
about a missing argument, Cloud Shell reconnected: run this block again.

## 1. Give the current champion its folder (once, 1 minute)

The three top-level files are the bundle your service runs now (version 1, exported in
Week 6 with 8,000 applicants). Copy them into `models/v1/`:

```bash
gcloud storage cp gs://$BUCKET/model.joblib gs://$BUCKET/model_info.json \
    gs://$BUCKET/feature_store.parquet gs://$BUCKET/models/v1/
gcloud storage ls gs://$BUCKET/models/v1/
```

**Check:** three files listed under `models/v1/`.

## 2. Point the service at that folder (once, 1-2 minutes)

```bash
gcloud run services update credit-risk-api --region=$REGION \
    --update-env-vars=MODEL_BUCKET=gs://$BUCKET/models/v1
curl -s $URL/model-info
```

**Check:** it ends with `Service [credit-risk-api] revision [...] has been deployed`, and
`/model-info` says `"version":"1"`. Nothing changed for the callers: same model, same URL;
only where it is loaded from.

## 3. If you promote: the challenger's folder, then the switch

On your laptop, `python run.py promote` moved the alias and wrote the challenger's bundle into
`models/`. Upload it into its own folder (Console, not Cloud Shell):

1. **Cloud Storage → Buckets →** your bucket → open the folder **models**.
2. **Create folder** → name it `v2` (the challenger's version number) → **Create**.
3. Open `v2` → **Upload → Upload files** → select the three files in your repository's
   `models/` folder → wait for **Complete**.

Check that `models/v2/model_info.json` says version 2:

```bash
gcloud storage cat gs://$BUCKET/models/v2/model_info.json | grep '"version"'
```

Then switch the service:

```bash
gcloud run services update credit-risk-api --region=$REGION \
    --update-env-vars=MODEL_BUCKET=gs://$BUCKET/models/v2
curl -s $URL/model-info
```

**Check:** `"version":"2"` and the challenger's threshold.

## 4. Roll back (practise it once, even if you kept the champion)

On your laptop: `python run.py rollback --version 1` (the alias and `models/` follow). In
Cloud Shell:

```bash
gcloud run services update credit-risk-api --region=$REGION \
    --update-env-vars=MODEL_BUCKET=gs://$BUCKET/models/v1
curl -s $URL/model-info
```

**Check:** `"version":"1"`. Time it: that is your rollback time for the runbook.

If you kept the champion, there is no `models/v2/` yet: practise the switch with the same
command and `models/v1` (the service restarts on the same model), and on Wednesday with the
version your automated pipeline registers.

## When something goes wrong

| What you see | Why, and what to do |
|---|---|
| `update` ends with "container failed to start" | The folder is missing a file, or the files are in a sub-folder: `gcloud storage ls gs://$BUCKET/models/vN/` must list exactly the three files |
| `403 ... storage.objects.get` in the service log | Should not happen (the service account can read the whole bucket); check the bucket name in the command |
| `/model-info` still shows the old version | You are reading the old revision for a few seconds: wait 30 seconds and try again |
| `URL` is empty | The service name or region differs: `gcloud run services list` |
| `cp: ... No URLs matched` in step 1 | The top-level files have other names in your bucket: `gcloud storage ls gs://$BUCKET` |
