<!-- PROVIDED: copy this file to docs/deployment.md and fill in every placeholder.
     The check reads the service address from this file and calls it. -->

# Deployment (Week 5)

Team: TODO

| | |
|---|---|
| Service URL | TODO (https://...run.app) |
| Project id / region | TODO / us-central1 |
| Image | TODO (us-central1-docker.pkg.dev/.../credit-risk/api:v1) |
| Model bucket | TODO (gs://credit-risk-models-...; private) |
| Model version served | TODO (from /model-info) |
| Instances | min 0, max TODO |
| Memory / CPU | TODO |
| Service account | TODO (credit-risk-api@...; role: Storage Object Viewer on the bucket only) |

## How to deploy a new model

TODO: The steps your team follows when a new model is registered (export, upload the bundle,
redeploy or restart the service), so that someone else on the team could do it.
