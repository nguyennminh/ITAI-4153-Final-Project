<!-- PROVIDED: copy this file to docs/deployment.md and fill in every placeholder.
     The check reads the service address from this file and calls it. -->

# Deployment (Week 5)

Team: Group 4 (Krish Iyer, Minh Nguyen, Calvin Zheng)

| | |
|---|---|
| Service URL | TODO (the Service URL printed at the end of `gcloud run deploy`) |
| Project id / region | credit-risk-group4-513720 / us-central1 |
| Image | TODO (us-central1-docker.pkg.dev/credit-risk-group4-513720/credit-risk/api:v1, confirm after the push) |
| Model bucket | TODO (gs://credit-risk-models-credit-risk-group4-513720 unless the name was taken; private) |
| Model version served | 1 (`credit-risk@staging`; confirm with /model-info on the live service) |
| Instances | min 0, max 2 |
| Memory / CPU | 1 GiB / 1 CPU |
| Service account | credit-risk-api@credit-risk-group4-513720.iam.gserviceaccount.com; role: Storage Object Viewer on the model bucket only |

## How to deploy a new model

1. Register the new model on the laptop that holds our MLflow database: `python run.py final`.
2. On the same laptop, export the bundle: `python run.py export`. It rewrites `models/` and
   `requirements-api.txt`. If the Python version on the second line of
   `requirements-api.txt` changed, change the Dockerfile's `FROM python:<version>-slim` to
   match, or the saved model may not load.
3. Run `python run.py evaluate` (the gate needs fresh metrics), commit
   `requirements-api.txt`, `reports/gate_metrics.json` and any Dockerfile change on a branch,
   open a pull request, and merge it once the Cloud Build check is green.
4. Upload the three `models/` files to the top level of the bucket, replacing the old ones
   (Console: Cloud Storage → the bucket → Upload files, or
   `gcloud storage cp models/* gs://<bucket>/`). The bucket stays private.
5. If only the model changed (same packages), restart the service so it reads the new bundle
   when it starts: in Cloud Shell, `gcloud run services update credit-risk-api
   --region=us-central1 --update-env-vars=BUNDLE_UPDATED=<date>` starts a new revision.
6. If the code, the packages, or the Dockerfile changed, rebuild and redeploy instead:
   `docs/deploy_cloud_run.md` steps 0, 5, 6, 7 and 8 (use a new image tag, for example `v2`).
7. Check the live service: `curl <service URL>/model-info` shows the new version, then
   `python run.py check`.
