<!-- PROVIDED: copy this file to docs/ci_report.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# CI and the quality gate (Week 4)

Team: TODO

## The pipeline

Cloud Build project: TODO (project id). Trigger: TODO (trigger name), on pull requests to
`main`. Branch protection on `main` requires the check: TODO (the check's name as GitHub
shows it).

| Step | What it runs | Why it is there |
|---|---|---|
| install | `pip install -r requirements.txt` | the same packages as on our laptops |
| lint | TODO | TODO |
| test | TODO | TODO (which tests run in CI, which are skipped, and why) |
| gate | TODO | TODO |

## Gate thresholds, and why

For each threshold in `config/gate.yaml`: the value, the number it is compared with today,
and why this value (what you looked at, and what passing or failing it would mean).

- **min_auc** = TODO.
- **max_cost** = TODO.
- **max_auc_drop** = TODO.
- **allow_failing_model_tests** = TODO.

## Evidence

- A green Cloud Build run on a pull request: TODO (paste the link to the build page: it
  starts with https://console.cloud.google.com/cloud-build/)
- A red run, and what made it red: TODO (the link, and the gate's message)
- How the red pull request was handled: TODO
